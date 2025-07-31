import os
import os.path as osp
import numpy as np
import warnings
import sys
from Bio.PDB import PDBParser
from tqdm import tqdm
from scipy.spatial import KDTree
import torch 
import torch.nn.functional as F
import random
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from sklearn.preprocessing import LabelEncoder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from MFE.dmasif_encoder.data_iteration import Generate_Surface
#from dMasif.dmasif_encoder.geometry_processing import curvatures, assign_curvature_to_alphaC
#from showCloud import visualize_protein_surface
from sklearn.neighbors import NearestNeighbors
#import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
#dataset
amino_acid_dict = {"ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q", "GLU": "E",
                   "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F",
                   "PRO": "P", "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V"}
# --- 残基物理化学特征---
HYDROPATHY = {"I": 4.5, "V": 4.2, "L": 3.8, "F": 2.8, "C": 2.5, "M": 1.9, 
              "A": 1.8, "W": -0.9, "G": -0.4, "T": -0.7, "S": -0.8, 
              "Y": -1.3, "P": -1.6, "H": -3.2, "N": -3.5, "D": -3.5, 
              "Q": -3.5, "E": -3.5, "K": -3.9, "R": -4.5}
CHARGE = {"R": 1, "K": 1, "D": -1, "E": -1, "H": 0.1}
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')

def find_struc2surf(surface_cloud, structure_points, k=8):
    """
    为每个结构点查找最近的k个表面点云索引
    
    参数:
        surface_cloud (Tensor): 表面点云数据 [N, 3]
        structure_points (Tensor): 结构点数据 [M, 3]
        k (int): 最近邻数量，默认为8
    
    返回:
        Tensor: 包含最近邻索引的LongTensor [M, k]
    """
    # 转换Tensor到CPU并转为numpy数组
    surface_np = surface_cloud.numpy()
    structure_np = structure_points.numpy()
    
    # 构建KDTree
    tree = KDTree(surface_np)
    
    # 批量查询所有结构点的最近邻
    _, indices = tree.query(structure_np, k=k)
    
    # 转换为PyTorch Tensor并保持设备一致
    return indices

def find_surf2struc(surface_points, structure_points, threshold, max_neighbor):
    """
    查找每个表面点周围在阈值范围内且不超过最大邻居数的结构点索引
    
    参数：
        surface_points (np.ndarray): 表面点云数据，形状为[N, 3]
        structure_points (np.ndarray): 结构点数据，形状为[M, 3]
        threshold (float): 距离阈值
        max_neighbor (int): 每个表面点的最大邻居数
    
    返回：
        list: 每个表面点对应的邻居结构点索引列表，按距离排序
    """
    # 构建结构点的KD树
    structure_kdtree = KDTree(structure_points)
    
    # 单线程批量查询所有表面点的邻居
    neighbor_indices = structure_kdtree.query_ball_point(
        surface_points, 
        threshold
    )  # 移除了workers参数
    
    # 处理空结果的情况
    if len(neighbor_indices) == 0:
        return [[] for _ in range(len(surface_points))]
    
    # 计算距离并排序的向量化操作
    surface_indices = np.repeat(
        np.arange(len(surface_points)),
        [len(indices) for indices in neighbor_indices]
    )
    all_diffs = structure_points[np.concatenate(neighbor_indices)] - surface_points[surface_indices]
    dists = np.linalg.norm(all_diffs, axis=1)
    
    # 按距离排序并截断
    result = []
    offset = 0
    for indices in neighbor_indices:
        n = len(indices)
        if n > 0:
            sorted_order = np.argsort(dists[offset:offset+n])
            sorted_indices = np.array(indices)[sorted_order]
            result.append(sorted_indices[:max_neighbor].tolist())
        else:
            result.append([])
        offset += n
    
    return result



def merge_idx(idx):
    a = idx[:, 0] * (10 ** (np.log10(idx[:, 1]).astype(int) + 1)) + idx[:, 1]

    b = a * (10 ** (np.log10(idx[:, 2]).astype(int) + 1)) + idx[:, 2]

    return b

def get_voxel_dict(ids, lines):
    voxel_dict = dict()
    for ind, line in zip(ids, lines):
        if ind not in voxel_dict:
            voxel_dict[ind] = []
        voxel_dict[ind].append(line)
    return voxel_dict

def fill_idx(xyz, x, y, z, empty_index):

    for i in range(len(xyz)):

        for j in range(len(x)):

            if xyz[i,0] >= x[j] and xyz[i,0] <= x[j+1]:

                for k in range(len(y)):

                    if xyz[i,1] >= y[k] and xyz[i,1] <= y[k+1]:

                        for l in range(len(z)):

                            if xyz[i,2] >= z[l] and xyz[i,2] <= z[l+1]:

                                empty_index[i,0] = j+1
                                empty_index[i,1] = k+1
                                empty_index[i,2] = l+1

                                break
                        break
                break
    return empty_index

def octree(xyz):
    xyzmin = np.min(xyz, axis=0)
    xyzmax = np.max(xyz, axis=0)
    n = 0
    idx = np.zeros_like(xyz, dtype=int)

    number = 0
    #: there will be implemented more conditions to stop the split process
    while number < 4:
        x = np.linspace(xyzmin[0], xyzmax[0], n)
        y = np.linspace(xyzmin[1], xyzmax[1], n)
        z = np.linspace(xyzmin[2], xyzmax[2], n)
        idx = fill_idx(xyz, x, y, z, idx)

        n = (2 ** n) + 1
        number += 1
    idx = merge_idx(idx)
    return idx

def atom_to_features(atom_id):
    """
    将原子标识符（如 'OE2_GLU_89'）转换为特征向量
    """
    # 解析原子标识符
    parts = atom_id.split("_")
    residue_name = parts[1]   # e.g., GLU
    # --- 残基特征 ---
    # 获取残基的单字母代码（如 GLU → E）
    aa = amino_acid_dict.get(residue_name, "X")  # 未知残基用 'X' 表示
    residue_hydrophobicity = HYDROPATHY.get(aa, 0.0) / 5.0  # 归一化
    residue_charge = CHARGE.get(aa, 0.0)

    # --- 组合特征 ---
    return [
        residue_hydrophobicity,
        residue_charge,
        # 可添加残基编号的编码（如 residue_num / 100.0 归一化）
    ]

# 示例：'OE2_GLU_89' → [hydrophobicity(E), charge(E), 1, 0, -1]
class CATHdataset(InMemoryDataset):
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 split='train',
                 task = None
                ):

        self.split = split
        self.root = root
        self.task = task

        super(CATHdataset, self).__init__(
            root, transform, pre_transform, pre_filter)
        
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self):
        name = 'processed'
        return osp.join(self.root, name, self.split)

    @property
    def raw_file_names(self):
        name = self.split + '.txt'
        return name

    @property
    def processed_file_names(self):
        if self.task is None:
            return 'CATH_4_2.pt'
        else:
            return 'CATH_4_2_'+self.task+'.pt'
        

    def extract_pdb_data(self, pdb_file):
        """
        从 PDB 文件中提取氨基酸类型、原子所属氨基酸 ID、原子名称和原子坐标。

        参数:
        pdb_file (str): PDB 文件的路径。

        返回:
        dict: 包含 'amino_types', 'atom_amino_id', 'atom_names', 'atom_pos' 的字典。
        """
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('protein_structure', pdb_file)

        amino_types = []
        atom_amino_id = []
        atom_names = []
        atom_pos = []
        target_atoms = {
            b'N', b'CA', b'C', b'CB', b'CG', b'SG', b'OG', b'CG1', b'OG1', 
            b'CD', b'SD', b'CD1', b'OD1', b'ND1', b'CE', b'NE', b'OE1', 
            b'CZ', b'NZ', b'NH1'
        }
        for model in structure:
            for chain in model:
                t = 0
                for residue in chain:
                    # 仅处理标准氨基酸残基
                    if residue.id[0] != ' ':
                        continue
                    # 获取氨基酸类型（使用三字母代码）
                    aa_type = residue.get_resname()
                    amino_types.append(aa_type)
                    res_id = t  # 残基编号
                    t = t + 1
                    for atom in residue:
                        if atom.name.encode() in target_atoms :
                            atom_amino_id.append(res_id)
                            atom_names.append(atom.name)
                            atom_pos.append(atom.coord)  # NumPy 数组

        # 手动创建三字母到单字母的映射字典
        three_to_one = {
            'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G',
            'HIS': 'H', 'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V',
            'TRP': 'W', 'TYR': 'Y'
        }

        # 将氨基酸类型的单字母代码映射为数字
        amino_acid_to_number = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5,
            'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
            'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17,
            'W': 18, 'Y': 19
        }

        # 将氨基酸类型转换为单字母表示
        amino_types_one_letter = [three_to_one.get(aa.upper(), 'X') for aa in amino_types]
        # 将列表转为字符串
        amino_acid_sequence = ''.join(amino_types_one_letter)
        # 将单字母氨基酸类型转换为数字
        amino_types_numeric = [amino_acid_to_number.get(aa, -1) for aa in amino_types_one_letter]

        # 返回值全部转换为 ndarray
        return {
            'amino_types': np.array(amino_types_numeric, dtype=np.int32),
            'atom_amino_id': np.array(atom_amino_id, dtype=np.int32),
            'atom_names': np.array(atom_names, dtype='S'),  # 用字节数组表示原子名称
            'atom_pos': np.array(atom_pos, dtype=np.float32),  # 原子坐标
            'amino_acid_sequence': np.array([amino_acid_sequence], dtype='S')  # 字符串作为 ndarray
        }

    def _normalize(self,tensor, dim=-1):
        '''
        Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
        '''
        return torch.nan_to_num(
            torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))

    def get_atom_pos(self, amino_types, atom_names, atom_amino_id, atom_pos):
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        mask_n = np.char.equal(atom_names, b'N')
        mask_ca = np.char.equal(atom_names, b'CA')
        mask_c = np.char.equal(atom_names, b'C')
        mask_cb = np.char.equal(atom_names, b'CB')
        mask_g = np.char.equal(atom_names, b'CG') | np.char.equal(atom_names, b'SG') | np.char.equal(atom_names, b'OG') | np.char.equal(atom_names, b'CG1') | np.char.equal(atom_names, b'OG1')
        mask_d = np.char.equal(atom_names, b'CD') | np.char.equal(atom_names, b'SD') | np.char.equal(atom_names, b'CD1') | np.char.equal(atom_names, b'OD1') | np.char.equal(atom_names, b'ND1')
        mask_e = np.char.equal(atom_names, b'CE') | np.char.equal(atom_names, b'NE') | np.char.equal(atom_names, b'OE1')
        mask_z = np.char.equal(atom_names, b'CZ') | np.char.equal(atom_names, b'NZ')
        mask_h = np.char.equal(atom_names, b'NH1')

        pos_n = np.full((len(amino_types),3),np.nan)
        temp = atom_pos[mask_n]
        temp2 = atom_amino_id[mask_n]
        pos_n[temp2] = temp

        pos_n = torch.FloatTensor(pos_n)
        flag=0
        pos_ca = np.full((len(amino_types),3),np.nan)
        pos_ca[atom_amino_id[mask_ca]] = atom_pos[mask_ca]
        pos_ca = torch.FloatTensor(pos_ca)
        # 检测 NaN 元素
        nan_mask = torch.isnan(pos_ca)  # size: (n_amino, 3)
        all_nan_rows = torch.all(nan_mask, dim=1)
        nan_row_indices = torch.nonzero(all_nan_rows, as_tuple=True)[0]
        if len(nan_row_indices) > 0:
            print('NaN in CA position:', nan_row_indices)
            # torch.set_printoptions(profile="full")
            # true_count = np.sum(mask_ca)
            # print(pos_ca)
            flag=1
            

        pos_c = np.full((len(amino_types),3),np.nan)
        pos_c[atom_amino_id[mask_c]] = atom_pos[mask_c]
        pos_c = torch.FloatTensor(pos_c)

        # if HomologyTAPE only contain pos_ca, we set the position of C and N as the position of CA
        pos_n[torch.isnan(pos_n)] = pos_ca[torch.isnan(pos_n)]
        pos_c[torch.isnan(pos_c)] = pos_ca[torch.isnan(pos_c)]

        pos_cb = np.full((len(amino_types),3),np.nan)
        pos_cb[atom_amino_id[mask_cb]] = atom_pos[mask_cb]
        pos_cb = torch.FloatTensor(pos_cb)

        pos_g = np.full((len(amino_types),3),np.nan)
        pos_g[atom_amino_id[mask_g]] = atom_pos[mask_g]
        pos_g = torch.FloatTensor(pos_g)

        pos_d = np.full((len(amino_types),3),np.nan)
        pos_d[atom_amino_id[mask_d]] = atom_pos[mask_d]
        pos_d = torch.FloatTensor(pos_d)

        pos_e = np.full((len(amino_types),3),np.nan)
        pos_e[atom_amino_id[mask_e]] = atom_pos[mask_e]
        pos_e = torch.FloatTensor(pos_e)

        pos_z = np.full((len(amino_types),3),np.nan)
        pos_z[atom_amino_id[mask_z]] = atom_pos[mask_z]
        pos_z = torch.FloatTensor(pos_z)

        pos_h = np.full((len(amino_types),3),np.nan)
        pos_h[atom_amino_id[mask_h]] = atom_pos[mask_h]
        pos_h = torch.FloatTensor(pos_h)

        return pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h, flag


    def side_chain_embs(self, pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h):
        v1, v2, v3, v4, v5, v6, v7 = pos_ca - pos_n, pos_cb - pos_ca, pos_g - pos_cb, pos_d - pos_g, pos_e - pos_d, pos_z - pos_e, pos_h - pos_z

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        angle1 = torch.unsqueeze(self.compute_dihedrals(v1, v2, v3),1)
        angle2 = torch.unsqueeze(self.compute_dihedrals(v2, v3, v4),1)
        angle3 = torch.unsqueeze(self.compute_dihedrals(v3, v4, v5),1)
        angle4 = torch.unsqueeze(self.compute_dihedrals(v4, v5, v6),1)
        angle5 = torch.unsqueeze(self.compute_dihedrals(v5, v6, v7),1)

        side_chain_angles = torch.cat((angle1, angle2, angle3, angle4),1)
        side_chain_embs = torch.cat((torch.sin(side_chain_angles), torch.cos(side_chain_angles)),1)
        
        return side_chain_embs

    
    def bb_embs(self, X):   
        # X should be a num_residues x 3 x 3, order N, C-alpha, and C atoms of each residue
        # N coords: X[:,0,:]
        # CA coords: X[:,1,:]
        # C coords: X[:,2,:]
        # return num_residues x 6 
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X, [3 * X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = self._normalize(dX, dim=-1)
        u0 = U[:-2]
        u1 = U[1:-1]
        u2 = U[2:]

        angle = self.compute_dihedrals(u0, u1, u2)
        
        # add phi[0], psi[-1], omega[-1] with value 0
        angle = F.pad(angle, [1, 2]) 
        angle = torch.reshape(angle, [-1, 3])
        angle_features = torch.cat([torch.cos(angle), torch.sin(angle)], 1)
        return angle_features

    
    def compute_dihedrals(self, v1, v2, v3):
        n1 = torch.cross(v1, v2)
        n2 = torch.cross(v2, v3)
        a = (n1 * n2).sum(dim=-1)
        b = torch.nan_to_num((torch.cross(n1, n2) * v2).sum(dim=-1) / v2.norm(dim=1))
        torsion = torch.nan_to_num(torch.atan2(b, a))
        return torsion
    
    def extract_vertix(self,input_file):
        print(input_file)
        import codecs
        with codecs.open(input_file, 'r', encoding='utf-8',
                        errors='ignore') as fdata:
            lines = fdata.readlines()[3: ]
        vertices, atoms = [], []
        for line in lines:
            phrases = line.strip().split()
            aa = phrases[-1].split("_")[1]
            if aa not in amino_acid_dict:
                continue
            atoms.append(phrases[-1])
            vertices.append(np.array([float(phrases[0]), float(phrases[1]), float(phrases[2])]))

        if len(vertices) <= 1:
            return vertices, atoms

        nbrs = NearestNeighbors(n_neighbors=8, algorithm='ball_tree').fit(np.array(vertices))
        distances, indices = nbrs.kneighbors(np.array(vertices))   # [N, 8]
        distances = np.square(distances)
        d = np.max(distances, axis=1, keepdims=True)   # [N, 8]
        probs = np.exp(-distances/d)
        probs = probs / np.sum(probs, axis=1, keepdims=True)

        new_vertices,all_lines = [],[]
        for prob, index in zip(probs, indices):
            neighbors = np.array([vertices[ind] for ind in index])  # [8, 3]
            vertex = np.sum(neighbors * prob.reshape(-1, 1), axis=0)
            new_vertices.append(vertex)


        
        center = np.mean(new_vertices, axis=0)
        max_ = np.max(new_vertices, axis=0)
        min_ = np.min(new_vertices, axis=0)
        length = np.max(max_ - min_)
        vertices = (new_vertices - center) / length

        for vertex,atom in zip(vertices,atoms):
            line = " ".join([str(term) for term in vertex])
            line = line + " " + atom+"\n"
            all_lines.append(line)

        ids = octree(vertices)
        voxel_dict = get_voxel_dict(ids, all_lines)
        total_points = len(all_lines)
        ratios = min(1.0, float(5000) / total_points)
        new_lines = []
        for key in voxel_dict:
            points = voxel_dict[key]
            number = int(len(points) * ratios)
            samples = random.sample(points, number)
            for sample in samples:
                new_lines.append(sample)
        vert_dict = {}
        for line in new_lines:
            phrases = line.strip().split()
            index = int(phrases[-1].split("_")[2])
            vert_dict[line] = index
        new_lines = sorted(vert_dict.items(), key=lambda item: item[1])
        surface_coor,surf_atoms = [],[]
        for item in new_lines:
            line = item[0]
            phrases = line.strip().split()
            coor = [float(phrases[0]), float(phrases[1]), float(phrases[2])]
            surface_coor.append(coor)
            surf_atoms.append(phrases[-1])

        aa_features = [atom_to_features(atom_id) for atom_id in surf_atoms]
        return surface_coor,aa_features
    # def batch_neighbors_kpconv(self, queries, supports, q_batches, s_batches, radius, max_neighbors):
    #     """
    #     Computes neighbors for a batch of queries and supports, apply radius search
    #     :param queries: (N1, 3) the query points
    #     :param supports: (N2, 3) the support points
    #     :param q_batches: (B) the list of lengths of batch elements in queries
    #     :param s_batches: (B)the list of lengths of batch elements in supports
    #     :param radius: float32
    #     :return: neighbors indices
    #     """

    #     neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    #     if max_neighbors > 0:
    #         return torch.from_numpy(neighbors[:, :max_neighbors])
    #     else:
    #         return torch.from_numpy(neighbors)
        
    def protein_to_graph(self, pdb_file_path):
        # 从 PDB 文件中提取数据
        data = Data()
        pdb_data = self.extract_pdb_data(pdb_file_path)

        amino_types = pdb_data['amino_types']  # size: (n_amino,)
        mask = amino_types == -1
        if np.sum(mask) > 0:
            amino_types[mask] = 25  # 对于氨基酸类型，将 -1 的值设置为 25

        atom_amino_id = pdb_data['atom_amino_id']  # size: (n_atom,)
        atom_names = pdb_data['atom_names']  # size: (n_atom,) 原子类型 
        atom_pos = pdb_data['atom_pos']  # size: (n_atom, 3)
        seq = torch.from_numpy(amino_types)
        
        # atoms to compute side chain torsion angles: N, CA, CB, _G/_G1, _D/_D1, _E/_E1, _Z, NH1
        pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h,flag = self.get_atom_pos(amino_types, atom_names, atom_amino_id, atom_pos)
        if flag==1:
            # 找出 pos_ca 中全为 NaN 的行索引
            nan_rows = torch.isnan(pos_ca).all(dim=1)
            nan_row_indices = torch.nonzero(nan_rows, as_tuple=True)[0]  # 获取全为 NaN 的行索引
            # 假设无效的 res_id 以 tensor 形式给出
            invalid_res_ids = nan_row_indices  # 需要删除的 res_id，如 127 和 2

            # 找到 atom_amino_id 中所有属于无效 res_id 的索引
            invalid_indices = torch.isin(torch.tensor(atom_amino_id), invalid_res_ids)

            # 过滤 atom_pos 和 atom_amino_id，删除无效索引对应的行
            atom_pos = np.delete(atom_pos, invalid_indices.numpy(), axis=0)
            atom_amino_id = np.delete(atom_amino_id, invalid_indices.numpy())
            atom_names = np.delete(atom_names, invalid_indices.numpy())
            amino_types = amino_types[~nan_rows.numpy()]

            # 过滤掉全为 NaN 的行，保留其他行
            pos_n = pos_n[~nan_rows]
            pos_ca = pos_ca[~nan_rows]
            pos_c = pos_c[~nan_rows]
            pos_cb = pos_cb[~nan_rows]
            pos_g = pos_g[~nan_rows]
            pos_d = pos_d[~nan_rows]
            pos_e = pos_e[~nan_rows]
            pos_z = pos_z[~nan_rows]
            pos_h = pos_h[~nan_rows]
            seq = seq[~nan_rows.numpy()]
            
        data.seq= seq
        # dmasif 表面生成
        first_letters = [atom.decode('utf-8')[0] for atom in atom_names]
        atom_mapping = {
            'N': [1, 0, 0, 0],
            'C': [0, 1, 0, 0],
            'O': [0, 0, 1, 0],
            'S': [0, 0, 0, 1]
        }
        # 创建一个 one-hot 编码的列表
        atom_types = [atom_mapping[letter] for letter in first_letters]
        atom_pos_tensor = torch.tensor(atom_pos,device=device)
        # 转换为 PyTorch tensor，并移动到 CUDA 设备
        atoms_types_tensor = torch.tensor(atom_types, device=device, dtype=torch.float32)
        data_item = {
                    'atom_xyz': atom_pos_tensor,
                    'atomtypes': atoms_types_tensor
        }
        protein_surface = Generate_Surface(data_item,device)
        surf = {
            'xyz': protein_surface['xyz'].cpu(),
            'normals': protein_surface['normals'].cpu(),
            'batch': protein_surface['batch'].cpu(),
            'atomtypes': atoms_types_tensor.cpu(),
            'batch_atoms': protein_surface['batch_atoms'].cpu(),
            'atoms_xyz': atom_pos_tensor.cpu()
        }
        #struct_pos = atom_pos_tensor.cpu()
        surf_pos = protein_surface['xyz'].cpu()
        #data.surf2struct = find_surf2struc(surf_pos,pos_ca,5.0,8)
        data.struc2surf = find_struc2surf(surf_pos,pos_ca)
        # data.surf2struct = self.batch_neighbors_kpconv(struct_pos,surf_pos,struct_batch,surf_batch,5.0,64)
        # visualize_protein_surface(protein_surface['xyz'], protein_surface['normals'])
        data.surface = surf
        # P_curvatures = curvatures(
        #     protein_surface["xyz"],
        #     triangles= None,
        #     normals= protein_surface["normals"],
        #     scales=[1.0, 2.0, 3.0, 5.0, 10.0],
        #     batch=protein_surface["batch"],
        # )
        # data.surfembed = P_curvatures
        # acta=assign_curvature_to_alphaC(protein_surface['xyz'], P_curvatures, pos_ca)
        # data.acta = acta.cpu()

        # five side chain torsion angles
        # We only consider the first four torsion angles in side chains since only the amino acid arginine has five side chain torsion angles, and the fifth angle is close to 0.
        #surf_coor,surf_aa_feature=self.extract_vertix(vert_file_path)
        side_chain_embs = self.side_chain_embs(pos_n, pos_ca, pos_c, pos_cb, pos_g, pos_d, pos_e, pos_z, pos_h)
        side_chain_embs[torch.isnan(side_chain_embs)] = 0
        data.side_chain_embs = side_chain_embs

        # three backbone torsion angles
        bb_embs = self.bb_embs(torch.cat((torch.unsqueeze(pos_n,1), torch.unsqueeze(pos_ca,1), torch.unsqueeze(pos_c,1)),1))
        bb_embs[torch.isnan(bb_embs)] = 0
        data.bb_embs = bb_embs

        data.x = torch.unsqueeze(torch.tensor(amino_types),1)
        data.coords_ca = pos_ca
        data.coords_n = pos_n
        data.coords_c = pos_c
        #data.surf_coor=surf_coor
        #data.surf_aa_feature=surf_aa_feature

        assert len(data.x)==len(data.coords_ca)==len(data.coords_n)==len(data.coords_c)==len(data.side_chain_embs)==len(data.bb_embs)

        return data,flag
    
    

    def process(self):
        print('Beginning Processing ...')
        # Get the file list.
        # 获取目录中的所有条目
        file_path = os.path.join(self.root,self.split)
        #vert_path = file_path+'_vert'
        all_entries = os.listdir(file_path)
        # 仅获取文件名（排除子目录）
        file_names_list = [entry for entry in all_entries if os.path.isfile(os.path.join(file_path, entry))]
        wrong_file_path = os.path.join(self.processed_dir, 'find_second.txt')
        # Load the dataset
        print("Reading the CATH_4.2")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wrong_pdb = [] 
            # 读取错误蛋白质文件并生成一个列表
            with open(wrong_file_path, 'r') as file:
                wrong_pdb = file.readlines()
                # 去掉每行末尾的换行符，并生成列表
                wrong_pdb = [line.strip() for line in wrong_pdb]                     
            data_list = []
            for fileIter, curFile in tqdm(enumerate(file_names_list)):
                path_temp = os.path.join(file_path,curFile)
                #vert_temp = os.path.join(vert_path,curFile[:-4]+'.vert') 
                #print('Processing: ', curFile[:4])
                # if curFile == '1a32.A.pdb':
                #     curProtein=self.protein_to_graph(path_temp)
                # else:
                #     continue
                if curFile not in wrong_pdb and curFile[:6] == '1a32.A':
                    print('protein is not second pdb: ', curFile[:-4])
                    continue
                curProtein,flag = self.protein_to_graph(path_temp)
                # if flag==1:
                #     wrong_pdb.append(curFile[:-4])
                #     continue
                curProtein.id = curFile[:-4]
                curProtein.y = curProtein.seq
                # if not curProtein.coords_ca is None and curProtein.coords_ca.shape[0] <=100 :
                #     print('protein is shorter than 100: ', curFile[:-4])
                #     data_list.append(curProtein) 
                data_list.append(curProtein)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print('Done!')
        # print('Wrong PDB files: ', wrong_pdb)
        # with open(osp.join(self.processed_dir, "wrong_pdb.txt"), "w") as f:
        #     for item in wrong_pdb:
        #         f.write(item + "\n")
    
    
if __name__ == "__main__":
    for split in ['train', 'validation', 'test']:
        print('#### Now processing {} CATH 4.2 ####'.format(split))
        dataset = CATHdataset(root='./CATH4.2/', split=split)
        print(dataset)

# import plotly.express as px
# if __name__ == "__main__":
#     dataset = CATHdataset(root='./CATH4.2/', split='draw')
#     data, flag = dataset.protein_to_graph('./CATH4.2/draw/1b5l.A.pdb')

#     # 示例点云
#     points = data.surface['xyz'].numpy()  # 替换为你的数据

#     fig = px.scatter_3d(
#         x=points[:,0], y=points[:,1], z=points[:,2],
#         color=points[:,2],
#         opacity=0.7,
#         title="Protein Surface Point Cloud"
#     )

#     # 隐藏坐标轴和背景
#     fig.update_layout(
#         scene=dict(
#             xaxis=dict(visible=False, showbackground=False, showticklabels=False, showgrid=False),
#             yaxis=dict(visible=False, showbackground=False, showticklabels=False, showgrid=False),
#             zaxis=dict(visible=False, showbackground=False, showticklabels=False, showgrid=False),
#             bgcolor='rgba(0,0,0,0)'  # 设置透明背景
#         ),
#         plot_bgcolor='rgba(0,0,0,0)',  # 整体画布背景透明
#         paper_bgcolor='rgba(0,0,0,0)', # 外边框背景透明
#         margin=dict(l=0, r=0, b=0, t=30)  # 缩小边距（保留顶部空间给标题）
#     )

#     # 调整点的大小和去除非必要的悬停信息
#     fig.update_traces(
#         marker_size=2,
#         hoverinfo='none'  # 隐藏悬停信息
#     )

#     fig.show()
