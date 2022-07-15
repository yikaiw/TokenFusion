import torch
import torch.nn as nn
import sys
import os
import torch.nn.functional as F

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)

from .pointnet import Pointnet2Backbone
from .transformer import TransformerDecoderLayer
from .modules import PointsObjClsModule, FPSModule, GeneralSamplingModule, PositionEmbeddingLearned, PredictHead, \
    ClsAgnosticPredictHead
from .vit import *

MAX_HEIGHT = 530
MAX_WIDTH = 730
PATCH_SIZE = 16


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            #print(layer, x.shape)
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class PredictorLG(nn.Module):
    """ From DydamicVit
    """
    def __init__(self, embed_dim=384):
        super().__init__()
        self.score_nets = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim // 2),
            nn.GELU(),
            nn.Linear(embed_dim // 2, embed_dim // 4),
            nn.GELU(),
            nn.Linear(embed_dim // 4, 2),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x):
        x = self.score_nets(x)
        return x


class GroupFreeDetector(nn.Module):
    r"""
        A Group-Free detector for 3D object detection via Transformer.

        Parameters
        ----------
        num_class: int
            Number of semantics classes to predict over -- size of softmax classifier
        num_heading_bin: int
        num_size_cluster: int
        input_feature_dim: (default: 0)
            Input dim in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        width: (default: 1)
            PointNet backbone width ratio
        num_proposal: int (default: 128)
            Number of proposals/detections generated from the network. Each proposal is a 3D OBB with a semantic class.
        sampling: (default: kps)
            Initial object candidate sampling method
    """

    def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
                 input_feature_dim=0, width=1, bn_momentum=0.1, sync_bn=False, num_proposal=128, sampling='kps',
                 dropout=0.1, activation="relu", nhead=8, num_decoder_layers=6, dim_feedforward=2048,
                 self_position_embedding='xyz_learned', cross_position_embedding='xyz_learned',
                 size_cls_agnostic=False, img_backbone='tiny'):
        super().__init__()

        self.num_class = num_class
        self.num_heading_bin = num_heading_bin
        self.num_size_cluster = num_size_cluster
        self.mean_size_arr = mean_size_arr
        assert (mean_size_arr.shape[0] == self.num_size_cluster)
        self.input_feature_dim = input_feature_dim
        self.num_proposal = num_proposal
        self.bn_momentum = bn_momentum
        self.sync_bn = sync_bn
        self.width = width
        self.nhead = nhead
        self.sampling = sampling
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward
        self.self_position_embedding = self_position_embedding
        self.cross_position_embedding = cross_position_embedding
        self.size_cls_agnostic = size_cls_agnostic

        pre_trained = None
        det_token_num = 100
        if img_backbone == 'tiny':
            init_pe_size = [800, 1333]
            mid_pe_size = None
        elif img_backbone == 'small':
            init_pe_size = [512, 864]
            mid_pe_size = [512, 864]
        elif img_backbone == 'base':
            init_pe_size = [800, 1344]
            mid_pe_size = [800, 1344]
        use_checkpoint = False
        self.has_mid_pe = mid_pe_size
        use_checkpoint = False

        # Backbone point feature learning
        self.backbone_net = Pointnet2Backbone(input_feature_dim=self.input_feature_dim, width=self.width)
        if img_backbone == 'tiny':
            # self.img_backbone, hidden_dim = tiny(pretrained='ckpt/yolos_ti.pth')
            self.img_backbone, hidden_dim = tiny(pretrained='ckpt/deit_tiny_patch16_224-a1311bcf.pth')
        elif img_backbone == 'small':
            self.img_backbone, hidden_dim = small(pretrained=pre_trained)
        elif img_backbone == 'base':
            self.img_backbone, hidden_dim = base(pretrained=pre_trained)
        elif img_backbone == 'small_dWr':
            self.img_backbone, hidden_dim = small_dWr(pretrained=pre_trained)
        else:
            raise ValueError(f'backbone {img_backbone} not supported')

        self.img_backbone.finetune_det(
            det_token_num=det_token_num, img_size=init_pe_size, 
            mid_pe_size=mid_pe_size, use_checkpoint=use_checkpoint)
        self.img_class_embed = FFN(hidden_dim, hidden_dim, num_class + 1, 3)
        self.img_bbox_embed = FFN(hidden_dim, hidden_dim, 4, 3)

        if self.sampling == 'fps':
            self.fps_module = FPSModule(num_proposal)
        elif self.sampling == 'kps':
            self.points_obj_cls = PointsObjClsModule(288)
            self.gsample_module = GeneralSamplingModule()
        else:
            raise NotImplementedError
        # Proposal
        if self.size_cls_agnostic:
            self.proposal_head = ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288)
        else:
            self.proposal_head = PredictHead(num_class, num_heading_bin, num_size_cluster,
                                             mean_size_arr, num_proposal, 288)
        if self.num_decoder_layers <= 0:
            # stop building if has no decoder layer
            return

        # Transformer Decoder Projection
        self.decoder_key_proj = nn.Conv1d(288, 288, kernel_size=1)
        self.decoder_query_proj = nn.Conv1d(288, 288, kernel_size=1)

        # Position Embedding for Self-Attention
        if self.self_position_embedding == 'none':
            self.decoder_self_posembeds = [None for i in range(num_decoder_layers)]
        elif self.self_position_embedding == 'xyz_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(3, 288))
        elif self.self_position_embedding == 'loc_learned':
            self.decoder_self_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_self_posembeds.append(PositionEmbeddingLearned(6, 288))
        else:
            raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            self.decoder_cross_posembeds = [None for i in range(num_decoder_layers)]
        elif self.cross_position_embedding == 'xyz_learned':
            self.decoder_cross_posembeds = nn.ModuleList()
            for i in range(self.num_decoder_layers):
                self.decoder_cross_posembeds.append(PositionEmbeddingLearned(3, 288))
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        # Transformer decoder layers
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    288, nhead, dim_feedforward, dropout, activation,
                    self_posembed=self.decoder_self_posembeds[i],
                    cross_posembed=self.decoder_cross_posembeds[i],
                ))

        # Image2point projection
        self.img2point_proj = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.img2point_proj.append(FFN(hidden_dim, hidden_dim, 288, 3))
            #self.img2point_proj.append(nn.Conv1d(hidden_dim, 288, kernel_size=1))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            if self.size_cls_agnostic:
                self.prediction_heads.append(ClsAgnosticPredictHead(num_class, num_heading_bin, num_proposal, 288))
            else:
                self.prediction_heads.append(PredictHead(
                    num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, 288))

        predictor_list = [PredictorLG(288) for i in range(len(self.num_decoder_layers))]
        self.score_predictor = nn.ModuleList(predictor_list)

        # Init
        self.init_weights()
        self.init_bn_momentum()
        if self.sync_bn:
            nn.SyncBatchNorm.convert_sync_batchnorm(self)

    def forward(self, inputs):
        """ Forward pass of the network

        Args:
            inputs: dict
                {point_clouds}

                point_clouds: Variable(torch.cuda.FloatTensor)
                    (B, N, 3 + input_channels) tensor
                    Point cloud to run predicts on
                    Each point in the point-cloud MUST
                    be formated as (x, y, z, features...)
        Returns:
            end_points: dict
        """

        # if isinstance(inputs['full_img'], (list, torch.Tensor)):
        #     samples = nested_tensor_from_tensor_list(inputs['full_img'])
        # x = self.img_backbone(samples.tensors)
        # print(inputs['full_img'].shape)
        end_points = {}
        end_points = self.backbone_net(inputs['point_clouds'], end_points)
        # Query Points Generation
        points_xyz = end_points['fp2_xyz']
        points_features = end_points['fp2_features']
        xyz = end_points['fp2_xyz']
        features = end_points['fp2_features']
        end_points['seed_inds'] = end_points['fp2_inds']
        end_points['seed_xyz'] = xyz
        end_points['seed_features'] = features
        if self.sampling == 'fps':
            xyz, features, sample_inds = self.fps_module(xyz, features)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        elif self.sampling == 'kps':
            points_obj_cls_logits = self.points_obj_cls(features)  # (batch_size, 1, num_seed)
            end_points['seeds_obj_cls_logits'] = points_obj_cls_logits
            points_obj_cls_scores = torch.sigmoid(points_obj_cls_logits).squeeze(1)
            sample_inds = torch.topk(points_obj_cls_scores, self.num_proposal)[1].int()
            xyz, features, sample_inds = self.gsample_module(xyz, features, sample_inds)
            cluster_feature = features
            cluster_xyz = xyz
            end_points['query_points_xyz'] = xyz  # (batch_size, num_proposal, 3)
            end_points['query_points_feature'] = features  # (batch_size, C, num_proposal)
            end_points['query_points_sample_inds'] = sample_inds  # (bsz, num_proposal) # should be 0,1,...,num_proposal
        else:
            raise NotImplementedError

        # inputs['calib_Rtilt']: [8, 3, 3], points_xyz: [8, 1024, 3]
        xyz2 = torch.matmul(inputs['calib_Rtilt'].transpose(2, 1), (1 / (inputs['scale'] ** 2)) \
            .unsqueeze(-1).unsqueeze(-1) * points_xyz.transpose(2, 1))
        xyz2 = xyz2.transpose(2, 1)
        xyz2[:, :, [0, 1, 2]] = xyz2[:, :, [0, 2, 1]]
        xyz2[:, :, 1] *= -1
        end_points['xyz_camera_coord'] = xyz2
        uv = torch.matmul(xyz2, inputs['calib_K'].transpose(2, 1)).detach()
        uv[:, :, 0] /= uv[:, :, 2]
        uv[:, :, 1] /= uv[:, :, 2]
        # uv: [8, 1024, 3]
        u, v = (uv[:, :, 0] - 1).round(), (uv[:, :, 1] - 1).round()  # shape [8, 1024] in [0, 730], [0, 530]
        u, v = torch.floor(u / PATCH_SIZE), torch.floor(v / PATCH_SIZE)  # shape [8, 1024] in [0, 45], [0, 33]
        proj_patch_idx = (v * (MAX_WIDTH // PATCH_SIZE) + u).long()  # for img shape [33, 45] flattened to [33 * 45]
        # shape [8, 1024] in [0, 33 * 45 - 1]
        #torch.save({'point_clouds': inputs['point_clouds'], 'points_xyz': points_xyz, 'u': u, 'v': v, 'proj_patch_idx': proj_patch_idx, 'full_img': inputs['full_img']}, 'project.pth')
        #assert 1==2
        # Proposal
        proposal_center, proposal_size = self.proposal_head(
            cluster_feature, base_xyz=cluster_xyz, end_points=end_points, prefix='proposal_')  # N num_proposal 3

        base_xyz = proposal_center.detach().clone()  # [8, 256]
        base_size = proposal_size.detach().clone()  # [8, 256]

        # Transformer Decoder and Prediction
        if self.num_decoder_layers > 0:
            query = self.decoder_query_proj(cluster_feature)
            key = self.decoder_key_proj(points_features) if self.decoder_key_proj is not None else None
        # points_features: [8, 288, 1024], key: [8, 288, 1024]
        # Position Embedding for Cross-Attention
        if self.cross_position_embedding == 'none':
            key_pos = None
        elif self.cross_position_embedding in ['xyz_learned']:
            key_pos = points_xyz
        else:
            raise NotImplementedError(f"cross_position_embedding not supported {self.cross_position_embedding}")

        img, temp_mid_pos_embed  = self.img_backbone(inputs['full_img'])
        # img: [8, 1586, hidden_dim], 1586 = 1 + 33 * 45 + 100
        proj_patch_idx = torch.cat([proj_patch_idx.unsqueeze(-1)] * img.shape[-1], dim=-1) + 1
        # shape [8, 1024, hidden_dim] in [1, 33 * 45], + 1 for cls token

        key_masks = []
        for i in range(self.num_decoder_layers):
            prefix = 'last_' if (i == self.num_decoder_layers - 1) else f'{i}head_'

            # Position Embedding for Self-Attention
            if self.self_position_embedding == 'none':
                query_pos = None
            elif self.self_position_embedding == 'xyz_learned':
                query_pos = base_xyz
            elif self.self_position_embedding == 'loc_learned':
                query_pos = torch.cat([base_xyz, base_size], -1)
            else:
                raise NotImplementedError(f"self_position_embedding not supported {self.self_position_embedding}")

            img = self.img_backbone.blocks[2 * i](img)
            if self.has_mid_pe:
                if 2 * i < (self.num_decoder_layers * 2 - 1):
                    img = img + temp_mid_pos_embed[2 * i]
            img = self.img_backbone.blocks[2 * i + 1](img)
            # img: [8, 1586, hidden_dim]
            if self.has_mid_pe:
                if 2 * i + 1 < (self.num_decoder_layers * 2 - 1):
                    img = img + temp_mid_pos_embed[2 * i + 1]

            img_proj = torch.gather(img, 1, proj_patch_idx)#.transpose(1, 2)  # [8, hidden_dim, 1024]

            key_score = self.score_predictor[i](key.transpose(1, 2))  # [8, 2, 1024]
            key_mask = F.softmax(key_score, dim=1)[:, 0]  # [8, 1024]
            key_masks.append(key_mask.flatten())

            key_ = key * key_mask.unsqueeze(1)
            img_ = self.img2point_proj[i](img_proj.detach()).transpose(1, 2)

            key_fusion = torch.zeros_like(key)
            mask_threshold = 0.02
            key_fusion[key_mask >= mask_threshold] = key_[key_mask >= mask_threshold]
            key_fusion[key_mask < mask_threshold] = img_[key_mask < mask_threshold]

            # key_fusion = key + self.img2point_proj[i](img_proj.detach()).transpose(1, 2)

            # Transformer Decoder Layer
            query = self.decoder[i](query, key_fusion, query_pos, key_pos)  # [8, 288, 1024]

            # Prediction
            base_xyz, base_size = self.prediction_heads[i](query, base_xyz=cluster_xyz, end_points=end_points, prefix=prefix)

            base_xyz = base_xyz.detach().clone()
            base_size = base_size.detach().clone()

        img = self.img_backbone.norm(img)[:, -self.img_backbone.det_token_num:, :]
        img_outputs_class = self.img_class_embed(img)
        img_outputs_coord = self.img_bbox_embed(img).sigmoid()
        end_points['pred_logits2d'] = img_outputs_class
        end_points['pred_boxes2d'] = img_outputs_coord
        end_points['key_masks'] = key_masks

        return end_points

    def init_weights(self):
        # initialize transformer
        for m in self.decoder.parameters():
            if m.dim() > 1:
                nn.init.xavier_uniform_(m)

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum
