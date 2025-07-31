"""
This is an implementation of ProNet model

"""
#5
from torch_geometric.nn import inits, MessagePassing,GCNConv
from torch_geometric.nn import radius_graph
from scipy.spatial import KDTree
from torch_scatter import scatter_mean
import torch.bin
from .features import d_angle_emb, d_theta_phi_emb

from torch_scatter import scatter
from torch_sparse import matmul
import yaml
import torch
from torch import nn
from torch.nn import Embedding
import torch.nn.functional as F
import numpy as np
from .autogressive_decoder import DecLayer
from easydict import EasyDict
import sys
import os
import plotly.express as px
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MFE.dmasif_encoder.protein_surface_encoder import dMaSIF
from MFE.dmasif_encoder.data_iteration import iterate_Surf
from .protein_decoder import *

class MLPDecoder(nn.Module):
    def __init__(self, hidden_dim, vocab=20):
        super().__init__()
        self.readout1 = nn.Linear(hidden_dim, vocab)
        # self.readout2 = nn.Linear(64, vocab)
        self._init_weights()

    def forward(self, h_V, batch_id=None):
        # min_val = torch.min(h_V)
        # max_val = torch.max(h_V)
        # new_max = 10
        # new_min = -10
        # normalized_tensor = (h_V - min_val)*(new_max-new_min) / (max_val - min_val)+new_min
        logits = self.readout1(h_V)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs
    
    def _init_weights(self):
        # nn.init.kaiming_uniform_(self.readout2.weight)
        nn.init.kaiming_uniform_(self.readout1.weight)

num_aa_type = 26
num_side_chain_embs = 8
num_bb_embs = 6

def swish(x):
    return x * torch.sigmoid(x)


class Linear(torch.nn.Module):
    """
        A linear method encapsulation similar to PyG's

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
        bias (int)
        weight_initializer (string): (glorot or zeros)
    """

    def __init__(self, in_channels, out_channels, bias=True, weight_initializer='glorot'):

        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_initializer = weight_initializer

        self.weight = nn.Parameter(torch.Tensor(out_channels, in_channels))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight_initializer == 'glorot':
            inits.glorot(self.weight)
        elif self.weight_initializer == 'zeros':
            inits.zeros(self.weight)
        if self.bias is not None:
            inits.zeros(self.bias)

    def forward(self, x):
        """"""
        return F.linear(x, self.weight, self.bias)


class TwoLinear(torch.nn.Module):
    """
        A layer with two linear modules

        Parameters
        ----------
        in_channels (int)
        middle_channels (int)
        out_channels (int)
        bias (bool)
        act (bool)
    """

    def __init__(
            self,
            in_channels,
            middle_channels,
            out_channels,
            bias=False,
            act=False
    ):
        super(TwoLinear, self).__init__()
        self.lin1 = Linear(in_channels, middle_channels, bias=bias)
        self.lin2 = Linear(middle_channels, out_channels, bias=bias)
        self.act = act

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, x):
        x = self.lin1(x)
        if self.act:
            x = swish(x)
        x = self.lin2(x)
        if self.act:
            x = swish(x)
        return x


class EdgeGraphConv(MessagePassing):
    """
        Graph convolution similar to PyG's GraphConv(https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html#torch_geometric.nn.conv.GraphConv)

        The difference is that this module performs Hadamard product between node feature and edge feature

        Parameters
        ----------
        in_channels (int)
        out_channels (int)
    """
    def __init__(self, in_channels, out_channels):
        super(EdgeGraphConv, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.lin_l = Linear(in_channels, out_channels)
        self.lin_r = Linear(in_channels, out_channels, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x, edge_index, edge_weight, size=None):
        x = (x, x)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=size)
        out = self.lin_l(out)
        return out + self.lin_r(x[1])

    def message(self, x_j, edge_weight):
        return edge_weight * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x[0], reduce=self.aggr)


class InteractionBlock(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            output_channels,
            num_radial,
            num_spherical,
            num_layers,
            mid_emb,
            act=swish,
            num_pos_emb=16,
            nums_curvature=10,
            dropout=0,
            level='allatom'
    ):
        super(InteractionBlock, self).__init__()
        self.act = act
        self.dropout = nn.Dropout(dropout)
        
        self.conv0 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv1 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv2 = EdgeGraphConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)

        self.lin_feature0 = TwoLinear(num_radial * num_spherical ** 2, mid_emb, hidden_channels)
        if level == 'aminoacid':
            self.lin_feature1 = TwoLinear(num_radial * num_spherical, mid_emb, hidden_channels)
        elif level == 'backbone' or level == 'allatom':
            self.lin_feature1 = TwoLinear(3 * num_radial * num_spherical, mid_emb, hidden_channels)
        self.lin_feature2 = TwoLinear(num_pos_emb, mid_emb, hidden_channels)

        self.lin_feature3 = TwoLinear(nums_curvature, mid_emb, hidden_channels)

        self.lin_1 = Linear(hidden_channels, hidden_channels)
        self.lin_2 = Linear(hidden_channels, hidden_channels)
        self.lin_3 = Linear(hidden_channels, hidden_channels)

        self.lin0 = Linear(hidden_channels, hidden_channels)
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        self.lin3 = Linear(hidden_channels, hidden_channels)

        self.lins_cat = torch.nn.ModuleList()
        self.lins_cat.append(Linear(3*hidden_channels, hidden_channels))
        for _ in range(num_layers-1):
            self.lins_cat.append(Linear(hidden_channels, hidden_channels))

        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers-1):
            self.lins.append(Linear(hidden_channels, hidden_channels))
        self.final = Linear(hidden_channels, output_channels)
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        self.layer_norm = nn.LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.conv0.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

        self.lin_feature0.reset_parameters()
        self.lin_feature1.reset_parameters()
        self.lin_feature2.reset_parameters()

        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()

        self.lin0.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

        for lin in self.lins:
            lin.reset_parameters()
        for lin in self.lins_cat:
            lin.reset_parameters()

        self.final.reset_parameters()


    def forward(self, x, feature0, feature1, pos_emb, edge_index, batch):
        x_lin_1 = self.act(self.lin_1(x))
        x_lin_2 = self.act(self.lin_2(x))
        
        feature0 = self.lin_feature0(feature0)
        h0 = self.conv0(x_lin_1, edge_index, feature0)
        h0 = self.lin0(h0)
        h0 = self.layer_norm(h0)
        h0 = self.act(h0)
        h0 = self.dropout(h0)

        feature1 = self.lin_feature1(feature1)
        h1 = self.conv1(x_lin_1, edge_index, feature1)
        h1 = self.lin1(h1)
        h1 = self.layer_norm(h1)
        h1 = self.act(h1)
        h1 = self.dropout(h1)

        feature2 = self.lin_feature2(pos_emb)
        h2 = self.conv2(x_lin_1, edge_index, feature2)
        h2 = self.lin2(h2)
        h2 = self.layer_norm(h2)
        h2 = self.act(h2)
        h2 = self.dropout(h2)

        #使用GCNConv来提取表面曲率特征
        # surf_feature = self.lin_feature3(curvature)
        # h3 = self.conv3(surf_feature, edge_index)
        # h3 = self.lin3(h3)
        # h3 = self.layer_norm(h3)
        # h3 = self.act(h3)
        # h3 = self.dropout(h3)

        h = torch.cat((h0, h1, h2 ),1)
        for lin in self.lins_cat:
            h = self.act(lin(h)) 

        h = h + x_lin_2

        for lin in self.lins:
            h = self.act(self.layer_norm(lin(h)))
        h = self.final(h)
        
        return h


class ProNet(nn.Module):
    r"""
         The ProNet from the "Learning Protein Representations via Complete 3D Graph Networks" paper.
        
        Args:
            level: (str, optional): The level of protein representations. It could be :obj:`aminoacid`, obj:`backbone`, and :obj:`allatom`. (default: :obj:`aminoacid`)
            num_blocks (int, optional): Number of building blocks. (default: :obj:`4`)
            hidden_channels (int, optional): Hidden embedding size. (default: :obj:`128`)
            out_channels (int, optional): Size of each output sample. (default: :obj:`1`)
            mid_emb (int, optional): Embedding size used for geometric features. (default: :obj:`64`)
            num_radial (int, optional): Number of radial basis functions. (default: :obj:`6`)
            num_spherical (int, optional): Number of spherical harmonics. (default: :obj:`2`)
            cutoff (float, optional): Cutoff distance for interatomic interactions. (default: :obj:`10.0`)
            max_num_neighbors (int, optional): Max number of neighbors during graph construction. (default: :obj:`32`)
            int_emb_layers (int, optional): Number of embedding layers in the interaction block. (default: :obj:`3`)
            out_layers (int, optional): Number of layers for features after interaction blocks. (default: :obj:`2`)
            num_pos_emb (int, optional): Number of positional embeddings. (default: :obj:`16`)
            dropout (float, optional): Dropout. (default: :obj:`0`)
            data_augment_eachlayer (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to the node features before each interaction block. (default: :obj:`False`)
            euler_noise (bool, optional): Data augmentation tricks. If set to :obj:`True`, will add noise to Euler angles. (default: :obj:`False`)
            
    """
    def __init__(
            self,
            level='aminoacid',
            num_blocks=4,
            hidden_channels=128,
            out_channels=1,
            mid_emb=64,
            num_radial=6,
            num_spherical=2,
            cutoff=10.0,
            max_num_neighbors=32,
            int_emb_layers=3,
            out_layers=2,
            num_pos_emb=16,
            dropout=0,
            data_augment_eachlayer=True,
            euler_noise = False,
    ):
        super(ProNet, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_pos_emb = num_pos_emb
        self.data_augment_eachlayer = data_augment_eachlayer
        self.euler_noise = euler_noise
        self.level = level
        self.num_surf = 10
        self.act = swish
        with open("./configs/config.yml", 'r') as f:
            config = EasyDict(yaml.safe_load(f))
        self.protein_surface_encoder = dMaSIF(config.model.dmasif)
        self.feature0 = d_theta_phi_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)
        self.feature1 = d_angle_emb(num_radial=num_radial, num_spherical=num_spherical, cutoff=cutoff)

        if level == 'aminoacid':
            self.embedding = Embedding(num_aa_type, hidden_channels)
        elif level == 'backbone':
            self.embedding = torch.nn.Linear(num_bb_embs, hidden_channels)
        elif level == 'allatom':
            self.embedding = torch.nn.Linear(num_bb_embs + num_side_chain_embs, hidden_channels)
        else:
            print('No supported model!')

        self.interaction_blocks = torch.nn.ModuleList(
            [
                InteractionBlock(
                    hidden_channels=hidden_channels,
                    output_channels=hidden_channels,
                    num_radial=num_radial,
                    num_spherical=num_spherical,
                    num_layers=int_emb_layers,
                    mid_emb=mid_emb,
                    act=self.act,
                    num_pos_emb=num_pos_emb,
                    dropout=dropout,
                    level=level
                )
                for _ in range(num_blocks)
            ]
        )
                
        self.lins_out = torch.nn.ModuleList()
        for _ in range(out_layers-1):
            self.lins_out.append(Linear(hidden_channels, hidden_channels))
        self.lin_out = Linear(hidden_channels, out_channels)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.decoder = DecLayer(hidden_channels)

        self.reset_parameters()
        self.batch_norm = nn.BatchNorm1d(hidden_channels)
        # 学习权重参数
        self.weight_a = nn.Parameter(torch.FloatTensor([0.5]))
        self.weight_b = nn.Parameter(torch.FloatTensor([0.5]))
        # 两个独立的MLP
        self.mlp_a = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )
        
        self.mlp_b = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels)
        )
        
        # 融合MLP
        self.fusion_mlp_2 = nn.Sequential(
            nn.Linear(hidden_channels * 2, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, hidden_channels)
        )
        self.corss_attention = CrossAttentionFusion(hidden_channels, hidden_dim=hidden_channels, output_dim=hidden_channels, num_heads=8, dropout=0.1)
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            nn.Linear(hidden_channels, out_channels)
        )

    # 修改ProNet类添加获取嵌入特征的方法（您需要在pronet.py文件中添加）
    def get_embeddings(self, batch_data):
        """
        提取模型中间层表示，包括结构特征、表面特征和融合特征
        
        Args:
            batch_data: 输入数据批次
            
        Returns:
            dict: 包含不同特征表示的字典
        """
        pos, batch = batch_data.coords_ca, batch_data.batch
        device = pos.device
        surf = batch_data.surface
        struct2surf = batch_data.struc2surf
        
        # 获取表面嵌入
        surf_embed = iterate_Surf(self.protein_surface_encoder, surf, device=device)
        
        pos_n = batch_data.coords_n
        pos_c = batch_data.coords_c
        bb_embs = batch_data.bb_embs
        side_chain_embs = batch_data.side_chain_embs
        
        # 处理输入特征
        if self.level == 'backbone':
            x = self.embedding(bb_embs)
        elif self.level == 'allatom':
            x = torch.cat([bb_embs, side_chain_embs], dim=1)
            x = self.embedding(x)
        else:
            print('No supported model!')
            
        # 创建边索引
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        pos_emb = self.pos_emb(edge_index, self.num_pos_emb)
        j, i = edge_index
        
        # 计算距离
        dist = (pos[i] - pos[j]).norm(dim=1)
        
        num_nodes = len(batch)
        
        # 计算角度 theta 和 phi
        refi0 = (i-1)%num_nodes
        refi1 = (i+1)%num_nodes
        
        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)
        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i]).norm(dim=-1)
        theta = torch.atan2(b, a)
        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i])
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i])
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * (pos[refi0] - pos[i])).sum(dim=-1) / ((pos[refi0] - pos[i]).norm(dim=-1))
        phi = torch.atan2(b, a)
        feature0 = self.feature0(dist, theta, phi)
        
        # 处理特征
        if self.level == 'backbone' or self.level == 'allatom':
            # 计算欧拉角
            Or1_x = pos_n[i] - pos[i]
            Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i]))
            Or1_z_length = Or1_z.norm(dim=1) + 1e-7
            
            Or2_x = pos_n[j] - pos[j]
            Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j]))
            Or2_z_length = Or2_z.norm(dim=1) + 1e-7
            
            Or1_Or2_N = torch.cross(Or1_z, Or2_z)
            
            angle1 = torch.atan2((torch.cross(Or1_x, Or1_Or2_N) * Or1_z).sum(dim=-1)/Or1_z_length, (Or1_x * Or1_Or2_N).sum(dim=-1))
            angle2 = torch.atan2(torch.cross(Or1_z, Or2_z).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1))
            angle3 = torch.atan2((torch.cross(Or1_Or2_N, Or2_x) * Or2_z).sum(dim=-1)/Or2_z_length, (Or1_Or2_N * Or2_x).sum(dim=-1))
            
            if self.euler_noise:
                euler_noise = torch.clip(torch.empty(3,len(angle1)).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
                angle1 += euler_noise[0]
                angle2 += euler_noise[1]
                angle3 += euler_noise[2]
            
            feature1 = torch.cat((self.feature1(dist, angle1), self.feature1(dist, angle2), self.feature1(dist, angle3)),1)
        
        elif self.level == 'aminoacid':
            refi = (i-1)%num_nodes
            
            refj0 = (j-1)%num_nodes
            refj = (j-1)%num_nodes
            refj1 = (j+1)%num_nodes
            
            mask = refi0 == j
            refi[mask] = refi1[mask]
            mask = refj0 == i
            refj[mask] = refj1[mask]
            
            plane1 = torch.cross(pos[j] - pos[i], pos[refi] - pos[i])
            plane2 = torch.cross(pos[j] - pos[i], pos[refj] - pos[j])
            a = (plane1 * plane2).sum(dim=-1) 
            b = (torch.cross(plane1, plane2) * (pos[j] - pos[i])).sum(dim=-1) / dist
            tau = torch.atan2(b, a)
            
            feature1 = self.feature1(dist, tau)
        
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature0, feature1, pos_emb, edge_index, batch)
        # 处理交互
        structure_features = x
        surface_features = self.aggregate_surface_to_structure(surf_embed, structure_features, surf["batch"], batch, struct2surf)
        fused_features = self.corss_attention(structure_features, surface_features)
        
        # 返回所有嵌入特征
        embeddings = {
            'structure_representation': structure_features,  # 结构特征
            'surface_representation': surface_features,      # 表面特征
            'fused_representation': fused_features,          # 融合特征
            'raw_surface_representation': surf_embed           # 原始表面嵌入
        }
        
        return embeddings
    
    def reset_parameters(self):
        self.embedding.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()
        for lin in self.lins_out:
            lin.reset_parameters()
        self.lin_out.reset_parameters()

    def pos_emb(self, edge_index, num_pos_emb=16):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_pos_emb, 2, dtype=torch.float32, device=edge_index.device)
            * -(np.log(10000.0) / num_pos_emb)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def aggregate_surface_to_structure(self, surf_repr, struc_repr, batch_surf, batch_struc, index_list):
        device = surf_repr.device
        batch_size = batch_surf.max().item() + 1  # 假设 batch_surf 和 batch_struc 的值是从0开始连续编号

        # 初始化存储所有聚合后的结构特征
        all_aggregated = torch.zeros_like(struc_repr)

        # 遍历每个批次
        for batch_idx in range(batch_size):
            # 提取当前批次的点索引
            surf_mask = (batch_surf == batch_idx)
            struc_mask = (batch_struc == batch_idx)

            surf_indices_in_batch = torch.where(surf_mask)[0]  # 当前批次表面点索引
            struc_indices_in_batch = torch.where(struc_mask)[0]  # 当前批次结构点索引

            if len(surf_indices_in_batch) == 0 or len(struc_indices_in_batch) == 0:
                continue

            # 当前批次的表面点特征
            surf_points = surf_repr[surf_indices_in_batch]  # [num_surf_points_in_batch, feat_dim]
            if batch_size == 1:
                # 当前批次的邻居索引，shape: [num_struct_points_in_batch, 8]
                adj_indices = index_list[batch_idx]
            else:
                # 如果只有一个批次，直接使用 index_list
                adj_indices = index_list

            # 直接用邻居索引聚合表面特征
            

            # 初始化邻居特征存储
            neighbor_features = torch.zeros((adj_indices.shape[0], adj_indices.shape[1], surf_points.shape[1]),
                                            device=device, dtype=surf_points.dtype)

            # 填充邻居特征
            # 找出有效的邻居索引
            # for i in range(adj_indices.shape[0]):
            #     for j in range(adj_indices.shape[1]):
            #         idx = adj_indices[i, j]
            #         if idx != -1:
            #             neighbor_features[i, j] = surf_points[idx]
            #         # 若为-1，则表示没有邻居，不需做操作
            # 直接用邻居索引批量索引 surf_points
            adj_indices_tensor = torch.tensor(adj_indices, device=device) 

            # 生成掩码
            valid_mask = adj_indices_tensor >= 0

            # 替换无效索引为0
            adj_indices_clamped = adj_indices_tensor.clone()
            adj_indices_clamped[~valid_mask] = 0

            # 用PyTorch索引
            neighbor_features = surf_points[adj_indices_clamped]

            # 将无效邻居的特征置为0
            # valid_mask的shape: [2,8]
            # 需要扩展维度
            mask_expanded = valid_mask.unsqueeze(-1).expand_as(neighbor_features)
            neighbor_features[~mask_expanded] = 0
            
            # 计算邻居的平均特征作为结构点的聚合特征
            aggregated_features = neighbor_features.mean(dim=1)  # shape: [num_struct_points_in_batch, feat_dim]

            # 使用结构点在整体中的索引更新结构特征
            all_aggregated[struc_indices_in_batch] = aggregated_features

        # 最后，将聚合的特征与原结构特征相加（或者其他融合策略）
        return all_aggregated
    
    def forward(self, batch_data):
        # print(batch_data.id)
        pos, batch = batch_data.coords_ca, batch_data.batch
        if batch is None:
            batch = torch.zeros(pos.shape[0], dtype=torch.long, device=pos.device)
        device = pos.device
        surf = batch_data.surface
        struct2surf = batch_data.struc2surf
        surf_embed = iterate_Surf(self.protein_surface_encoder,surf,device=device)
        pos_n = batch_data.coords_n
        pos_c = batch_data.coords_c
        bb_embs = batch_data.bb_embs
        side_chain_embs = batch_data.side_chain_embs
        
        if self.level == 'backbone':
            # x = torch.cat([torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs], dim = 1)
            x = self.embedding(bb_embs)
        elif self.level == 'allatom':
            # x = torch.cat([torch.squeeze(F.one_hot(z, num_classes=num_aa_type).float()), bb_embs, side_chain_embs], dim = 1)
            x = torch.cat([bb_embs, side_chain_embs], dim = 1)
            x = self.embedding(x)
        else:
            print('No supported model!')           
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, max_num_neighbors=self.max_num_neighbors)
        pos_emb = self.pos_emb(edge_index, self.num_pos_emb)
        j, i = edge_index

        # Calculate distances.
        dist = (pos[i] - pos[j]).norm(dim=1)
        
        num_nodes = len(batch)

        # Calculate angles theta and phi.
        refi0 = (i-1)%num_nodes
        refi1 = (i+1)%num_nodes

        a = ((pos[j] - pos[i]) * (pos[refi0] - pos[i])).sum(dim=-1)

        b = torch.cross(pos[j] - pos[i], pos[refi0] - pos[i]).norm(dim=-1)

        theta = torch.atan2(b, a)
        plane1 = torch.cross(pos[refi0] - pos[i], pos[refi1] - pos[i])
        plane2 = torch.cross(pos[refi0] - pos[i], pos[j] - pos[i])
        a = (plane1 * plane2).sum(dim=-1)
        b = (torch.cross(plane1, plane2) * (pos[refi0] - pos[i])).sum(dim=-1) / ((pos[refi0] - pos[i]).norm(dim=-1))
        phi = torch.atan2(b, a)                              
        feature0 = self.feature0(dist, theta, phi)
        if self.level == 'backbone' or self.level == 'allatom':
            # Calculate Euler angles.
            Or1_x = pos_n[i] - pos[i]
            Or1_z = torch.cross(Or1_x, torch.cross(Or1_x, pos_c[i] - pos[i]))
            Or1_z_length = Or1_z.norm(dim=1) + 1e-7
            
            Or2_x = pos_n[j] - pos[j]
            Or2_z = torch.cross(Or2_x, torch.cross(Or2_x, pos_c[j] - pos[j]))
            Or2_z_length = Or2_z.norm(dim=1) + 1e-7

            Or1_Or2_N = torch.cross(Or1_z, Or2_z)
            
            angle1 = torch.atan2((torch.cross(Or1_x, Or1_Or2_N) * Or1_z).sum(dim=-1)/Or1_z_length, (Or1_x * Or1_Or2_N).sum(dim=-1))
            angle2 = torch.atan2(torch.cross(Or1_z, Or2_z).norm(dim=-1), (Or1_z * Or2_z).sum(dim=-1))
            angle3 = torch.atan2((torch.cross(Or1_Or2_N, Or2_x) * Or2_z).sum(dim=-1)/Or2_z_length, (Or1_Or2_N * Or2_x).sum(dim=-1))

            if self.euler_noise:
                euler_noise = torch.clip(torch.empty(3,len(angle1)).to(device).normal_(mean=0.0, std=0.025), min=-0.1, max=0.1)
                angle1 += euler_noise[0]
                angle2 += euler_noise[1]
                angle3 += euler_noise[2]

            feature1 = torch.cat((self.feature1(dist, angle1), self.feature1(dist, angle2), self.feature1(dist, angle3)),1)

        elif self.level == 'aminoacid':
            refi = (i-1)%num_nodes

            refj0 = (j-1)%num_nodes
            refj = (j-1)%num_nodes
            refj1 = (j+1)%num_nodes

            mask = refi0 == j
            refi[mask] = refi1[mask]
            mask = refj0 == i
            refj[mask] = refj1[mask]

            plane1 = torch.cross(pos[j] - pos[i], pos[refi] - pos[i])
            plane2 = torch.cross(pos[j] - pos[i], pos[refj] - pos[j])
            a = (plane1 * plane2).sum(dim=-1) 
            b = (torch.cross(plane1, plane2) * (pos[j] - pos[i])).sum(dim=-1) / dist
            tau = torch.atan2(b, a)

            feature1 = self.feature1(dist, tau)

        # Interaction blocks.
        for interaction_block in self.interaction_blocks:
            x = interaction_block(x, feature0, feature1, pos_emb, edge_index, batch)
        # 区间融合-表面加结构
        x1 = self.aggregate_surface_to_structure(surf_embed,x,surf["batch"],batch,struct2surf)
        fused_features = self.corss_attention(x,x1)
        # fused_features = torch.cat((x,x1),dim=1)

        # y = self.decoder(fused_features)
        # 区间融合-结构
        y = self.decoder(fused_features)
        # fused_features = self.corss_attention(x,x1)
        # # 加权融合
        # weighted_sum = self.weight_a * x1 + self.weight_b * x
        # # 通过MLP处理
        # fused_features = self.fusion_mlp(weighted_sum)
        # p_a = self.mlp_a(x)
        # p_b = self.mlp_b(x1)
        # concat_features = torch.cat((p_a, p_b), dim=1)
        
        #fused_features = self.corss_attention(p_a,p_b)
        # x1 = self.aggregate_surface_to_structure(surf_embed,x,surf["batch"],batch,struct2surf)
        # # 加权融合
        # weighted_sum = self.weight_a * x1 + self.weight_b * x
        # # 通过MLP处理
        # fused_features = self.fusion_mlp(weighted_sum)
        #y = self.decoder(fused_features)
        
        return y
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())
    
def get_lengths_from_batch(batch):
    # batch: (total_len,)
    lengths = torch.bincount(batch).tolist()
    return lengths  # 例如 [5,8,10,5,9,...] 共32个长度

def assign_embedding_to_alphaC(
        surface_points: torch.Tensor, 
        curvature: torch.Tensor,        
        alpha_coords: torch.Tensor,     
        aggregation: str = "mean"       
    ) -> torch.Tensor:
    
    # Step 1: 表面点→alphaC分配
    alpha_tree = KDTree(alpha_coords.cpu().numpy())
    _, nearest_alpha_indices = alpha_tree.query(surface_points.cpu().numpy(), k=1)
    nearest_alpha_indices = torch.tensor(nearest_alpha_indices, device=surface_points.device, dtype=torch.long).squeeze(-1)
    
    # Step 2: alphaC→表面点反向分配（关键新增部分）
    surface_tree = KDTree(surface_points.cpu().numpy())
    _, nearest_surface_indices = surface_tree.query(alpha_coords.cpu().numpy(), k=1)
    nearest_surface_indices = torch.tensor(nearest_surface_indices, device=surface_points.device, dtype=torch.long).squeeze(-1)
    
    # 合并两种分配结果（原始分配 + 反向分配）
    combined_alpha_indices = torch.cat([
        nearest_alpha_indices, 
        torch.arange(len(alpha_coords), device=surface_points.device)  # 每个alphaC对应自己的反向索引
    ])
    combined_curvature = torch.cat([curvature, curvature[nearest_surface_indices]])
    
    # Step 3: 聚合逻辑（使用合并后的索引）
    aggregated_curvature = torch.zeros((alpha_coords.shape[0], curvature.shape[1]), device=curvature.device)
    
    if aggregation == "sum":
        aggregated_curvature.index_add_(0, combined_alpha_indices, combined_curvature)
    elif aggregation == "mean":
        counts = torch.zeros((alpha_coords.shape[0], 1), device=curvature.device)
        counts.index_add_(0, combined_alpha_indices, torch.ones_like(combined_curvature[:, :1]))
        aggregated_curvature.index_add_(0, combined_alpha_indices, combined_curvature)
        aggregated_curvature = aggregated_curvature / counts.clamp(min=1)
    elif aggregation == "max":
        aggregated_curvature.scatter_reduce_(
            0, 
            combined_alpha_indices[:, None].expand(-1, curvature.shape[1]), 
            combined_curvature, 
            reduce="amax", 
            include_self=False
        )
    
    return aggregated_curvature

# class CrossAttentionFusion(nn.Module):
#     def __init__(self, feature_dim, hidden_dim=128, output_dim=128, num_heads=8, dropout=0.1):
#         """
#         交叉注意力特征融合模型
        
#         参数:
#             feature_dim: 输入特征的维度
#             hidden_dim: 隐藏层维度
#             output_dim: 输出特征维度
#             num_heads: 多头注意力的头数
#             dropout: Dropout比率
#         """
#         super(CrossAttentionFusion, self).__init__()
        
#         # 确保hidden_dim可以被num_heads整除
#         assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
#         self.feature_dim = feature_dim
#         self.hidden_dim = hidden_dim
#         self.num_heads = num_heads
#         self.head_dim = hidden_dim // num_heads
        
#         # 特征投影层
#         self.proj_a = nn.Linear(feature_dim, hidden_dim)
#         self.proj_b = nn.Linear(feature_dim, hidden_dim)
        
#         # A注意B的查询、键、值投影
#         self.q_a = nn.Linear(hidden_dim, hidden_dim)
#         self.k_b = nn.Linear(hidden_dim, hidden_dim)
#         self.v_b = nn.Linear(hidden_dim, hidden_dim)
        
#         # B注意A的查询、键、值投影
#         self.q_b = nn.Linear(hidden_dim, hidden_dim)
#         self.k_a = nn.Linear(hidden_dim, hidden_dim)
#         self.v_a = nn.Linear(hidden_dim, hidden_dim)
        
#         # 输出投影
#         self.proj_out_a = nn.Linear(hidden_dim, hidden_dim)
#         self.proj_out_b = nn.Linear(hidden_dim, hidden_dim)
        
#         # 层归一化
#         self.norm_a1 = nn.LayerNorm(hidden_dim)
#         self.norm_a2 = nn.LayerNorm(hidden_dim)
#         self.norm_b1 = nn.LayerNorm(hidden_dim)
#         self.norm_b2 = nn.LayerNorm(hidden_dim)
        
#         # 前馈网络
#         self.ffn_a = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )
        
#         self.ffn_b = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim * 4),
#             nn.ReLU(),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim * 4, hidden_dim)
#         )
        
#         # 最终融合MLP
#         self.fusion_mlp = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.ReLU(),
#             nn.BatchNorm1d(hidden_dim),
#             nn.Dropout(dropout),
#             nn.Linear(hidden_dim, output_dim)
#         )
        
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, feature_a, feature_b):
#         batch_size = feature_a.size(0)
        
#         # 特征投影
#         proj_a = self.proj_a(feature_a)  # [batch_size, feature_dim] -> [batch_size, hidden_dim]
#         proj_b = self.proj_b(feature_b)  # [batch_size, feature_dim] -> [batch_size, hidden_dim]
        
#         # A注意B (A是查询，B是键和值)
#         q_a = self.q_a(proj_a).view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
#         k_b = self.k_b(proj_b).view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
#         v_b = self.v_b(proj_b).view(batch_size, self.num_heads, self.head_dim)  # [batch_size, num_heads, head_dim]
        
#         # 计算注意力分数 (A注意B)
#         attn_scores_a = torch.bmm(q_a.transpose(0, 1), k_b.transpose(0, 1).transpose(1, 2))  # [num_heads, batch_size, batch_size]
#         attn_scores_a = attn_scores_a / (self.head_dim ** 0.5)
#         attn_probs_a = F.softmax(attn_scores_a, dim=-1)
        
#         # 应用注意力权重
#         attn_output_a = torch.bmm(attn_probs_a, v_b.transpose(0, 1))  # [num_heads, batch_size, head_dim]
#         attn_output_a = attn_output_a.transpose(0, 1).contiguous().view(batch_size, self.hidden_dim)  # [batch_size, hidden_dim]
#         attn_output_a = self.proj_out_a(attn_output_a)
        
#         # 残差连接和层归一化
#         proj_a = proj_a + self.dropout(attn_output_a)
#         proj_a = self.norm_a1(proj_a)
        
#         # 前馈网络
#         ffn_output_a = self.ffn_a(proj_a)
#         proj_a = proj_a + self.dropout(ffn_output_a)
#         proj_a = self.norm_a2(proj_a)
        
#         # B注意A (B是查询，A是键和值)
#         q_b = self.q_b(proj_b).view(batch_size, self.num_heads, self.head_dim)
#         k_a = self.k_a(proj_a).view(batch_size, self.num_heads, self.head_dim)
#         v_a = self.v_a(proj_a).view(batch_size, self.num_heads, self.head_dim)
        
#         # 计算注意力分数 (B注意A)
#         attn_scores_b = torch.bmm(q_b.transpose(0, 1), k_a.transpose(0, 1).transpose(1, 2))
#         attn_scores_b = attn_scores_b / (self.head_dim ** 0.5)
#         attn_probs_b = F.softmax(attn_scores_b, dim=-1)
        
#         # 应用注意力权重
#         attn_output_b = torch.bmm(attn_probs_b, v_a.transpose(0, 1))
#         attn_output_b = attn_output_b.transpose(0, 1).contiguous().view(batch_size, self.hidden_dim)
#         attn_output_b = self.proj_out_b(attn_output_b)
        
#         # 残差连接和层归一化
#         proj_b = proj_b + self.dropout(attn_output_b)
#         proj_b = self.norm_b1(proj_b)
        
#         # 前馈网络
#         ffn_output_b = self.ffn_b(proj_b)
#         proj_b = proj_b + self.dropout(ffn_output_b)
#         proj_b = self.norm_b2(proj_b)
        
#         # 拼接增强后的特征
#         concat_features = torch.cat((proj_a, proj_b), dim=1)
        
#         # 最终融合
#         fused_features = self.fusion_mlp(concat_features)
        
#         return fused_features

class CrossAttentionFusion(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, output_dim=128, num_heads=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"
        
        self.proj_a = nn.Linear(feature_dim, hidden_dim)
        self.proj_b = nn.Linear(feature_dim, hidden_dim)
        
        # 交叉注意力模块
        self.mha_a_to_b = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.mha_b_to_a = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        
        # 层归一化
        self.norm_a1 = nn.LayerNorm(hidden_dim)
        self.norm_a2 = nn.LayerNorm(hidden_dim)
        self.norm_b1 = nn.LayerNorm(hidden_dim)
        self.norm_b2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn_a = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        self.ffn_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # 融合 MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, feature_a, feature_b):
        # 投影
        proj_a = self.proj_a(feature_a)  # [batch, hidden_dim]
        proj_b = self.proj_b(feature_b)  # [batch, hidden_dim]
        
        # 添加序列维度，长度=1，方便 mha 输入
        proj_a = proj_a.unsqueeze(1)  # [batch, seq_len=1, hidden_dim]
        proj_b = proj_b.unsqueeze(1)

        # A注意B：A是query，B是key和value
        attn_output_a, _ = self.mha_a_to_b(query=proj_a, key=proj_b, value=proj_b)
        # B注意A：B是query，A是key和value
        attn_output_b, _ = self.mha_b_to_a(query=proj_b, key=proj_a, value=proj_a)
        
        proj_a = proj_a + self.dropout(attn_output_a)
        proj_a = self.norm_a1(proj_a)
        ffn_out_a = self.ffn_a(proj_a)
        proj_a = proj_a + self.dropout(ffn_out_a)
        proj_a = self.norm_a2(proj_a)
        
        
        proj_b = proj_b + self.dropout(attn_output_b)
        proj_b = self.norm_b1(proj_b)
        ffn_out_b = self.ffn_b(proj_b)
        proj_b = proj_b + self.dropout(ffn_out_b)
        proj_b = self.norm_b2(proj_b)
        
        # 去掉序列维度
        proj_a = proj_a.squeeze(1)  # [batch, hidden_dim]
        proj_b = proj_b.squeeze(1)

        # 拼接并融合
        concat = torch.cat([proj_a, proj_b], dim=1)  # [batch, hidden_dim*2]
        fused = self.fusion_mlp(concat)
        return fused

class CrossAttentionFusionWo(nn.Module):
    def __init__(self, feature_dim, hidden_dim=128, 
                 output_dim=128, num_heads=8, dropout=0.1):
        super().__init__()
        assert hidden_dim % num_heads == 0
        
        self.proj_a = nn.Linear(feature_dim, hidden_dim)
        self.proj_b = nn.Linear(feature_dim, hidden_dim)

        self.mha_b_to_a = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True
        )
        self.norm_b1 = nn.LayerNorm(hidden_dim)
        self.norm_b2 = nn.LayerNorm(hidden_dim)
        self.ffn_b = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim*4),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim*4, hidden_dim)
        )

        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim*2, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, feature_a, feature_b):
        # feature_a: [batch, len1, feature_dim]
        # feature_b: [batch, len2, feature_dim]
        proj_a = self.proj_a(feature_a)   # [batch, len1, hidden_dim]
        proj_b = self.proj_b(feature_b)   # [batch, len2, hidden_dim]

        # B->A cross-attention, 输出对齐 len2
        attn_b, _ = self.mha_b_to_a(
            query=proj_b, key=proj_a, value=proj_a
        )                                  # [batch, len2, hidden_dim]

        # 残差 + LayerNorm + FFN
        proj_b = self.norm_b1(proj_b + attn_b)
        proj_b = self.norm_b2(proj_b + self.ffn_b(proj_b))

        # 序列级融合
        concat = torch.cat([proj_b, attn_b], dim=-1)  # [batch, len2, hidden_dim*2]
        fused = self.fusion_mlp(concat)               # [batch, len2, output_dim]
        return fused                                  # or fused.squeeze(0) => [len2, output_dim]


# class CrossAttentionFusion(nn.Module):
#     def __init__(self, embed_dim=128, num_heads=8):
#         super().__init__()
#         # 使用PyTorch内置多头注意力模块
#         self.cross_attn = nn.MultiheadAttention(
#             embed_dim=embed_dim,
#             num_heads=num_heads,
#             batch_first=False  # 输入格式为 (seq_len, batch, embed_dim)
#         )
        
#         # 初始化参数
#         self._reset_parameters()

#     def _reset_parameters(self):
#         # Xavier初始化提升收敛速度
#         nn.init.xavier_uniform_(self.cross_attn.in_proj_weight)
#         nn.init.constant_(self.cross_attn.in_proj_bias, 0.)
#         nn.init.xavier_uniform_(self.cross_attn.out_proj.weight)
#         nn.init.constant_(self.cross_attn.out_proj.bias, 0.)

#     def forward(self, vec_short, vec_long):
#         """
#         输入:
#         vec_short: (len, 128)  需要被增强的短向量
#         vec_long:  (16384, 128) 提供信息的上下文长向量
        
#         输出: 
#         (len, 128) 融合后的结果
#         """
#         # 添加batch维度 (PyTorch要求格式为 [seq_len, batch, embed_dim])
#         query = vec_short.unsqueeze(1)  # (len, 1, 128)
#         key = value = vec_long.unsqueeze(1)  # (16384, 1, 128)

#         # 计算交叉注意力
#         attn_output, _ = self.cross_attn(
#             query=query,
#             key=key,
#             value=value,
#             need_weights=False
#         )
        
#         # 移除batch维度
#         return attn_output.squeeze(1)  # (len, 128)
def visualize_batch_protein(p_xyz, p_normals, p_batch):
    """ 批次式蛋白质表面可视化 """
    # 数据转换
    points = p_xyz.cpu().numpy()
    batch_ids = p_batch.cpu().numpy()
    
    # 创建批次颜色映射（支持最多50个批次）
    max_batch = batch_ids.max()
    colors = px.colors.sample_colorscale("Rainbow", [n/(max_batch+1) for n in batch_ids])
    
    # 生成3D散点图
    fig = px.scatter_3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        color=batch_ids,
        color_continuous_scale="Rainbow",
        labels={'color': 'Batch ID'},
        title=f'Multi-Batch Visualization ({max_batch+1} batches)'
    )
    
    # 优化显示设置
    fig.update_traces(
        marker=dict(
            size=3.5,
            opacity=0.65,
            line=dict(width=0)
        )
    )
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)',
            aspectmode='cube'
        ),
        coloraxis_colorbar=dict(
            title="Batch ID",
            tickvals=np.linspace(0, max_batch, 5),
            ticktext=np.linspace(0, max_batch, 5).astype(int)
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    fig.show()

def visualize_protein_surface(p_xyz, p_normals, p_batch=None):
    """
    蛋白质表面点云可视化函数
    参数：
        p_xyz: (N,3) 点云坐标张量
        p_normals: (N,3) 法向量张量
        p_batch: (N,) 批次索引（可选）
    """
    # 转换为numpy数组
    points = p_xyz.cpu().numpy()
    normals = p_normals.cpu().numpy()
    
    # 修正颜色处理方式
    rgb_colors = np.round((normals + 1)/2 * 255).astype(np.uint8)  # 转换为0-255的RGB值
    color_strs = [f'rgb({r},{g},{b})' for r,g,b in rgb_colors]    # 转换为Plotly可识别的颜色字符串
    if p_batch is not None:
        batch_ids = p_batch.cpu().numpy()
        fig = px.scatter_3d(
            x=points[:,0], y=points[:,1], z=points[:,2],
            color=batch_ids,
            color_continuous_scale='Viridis',
            title='Batch-wise Visualization'
        )
    
    # 创建3D散点图
    fig = px.scatter_3d(
        x=points[:,0], y=points[:,1], z=points[:,2],
        # 修改颜色传递方式 ↓
        color_discrete_sequence=color_strs,  # 直接指定颜色序列
        title='Protein Surface Point Cloud'
    )

    # 调整显示效果
    fig.update_traces(
        marker=dict(
            size=2.5,
            opacity=0.8,
            line=dict(width=0)
        )
    )
    
    # 添加轴标签
    fig.update_layout(
        scene=dict(
            xaxis_title='X (Å)',
            yaxis_title='Y (Å)',
            zaxis_title='Z (Å)'
        ),
        margin=dict(l=0, r=0, b=0, t=30)
    )
    
    # 显示图表（Jupyter中自动显示，脚本中需调用show()）
    fig.show()

# 使用示例（假设已有p_xyz, p_normals数据）：
# visualize_protein_surface(p_xyz, p_normals)