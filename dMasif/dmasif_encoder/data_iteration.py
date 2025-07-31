import json

import math
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from MFE.dmasif_encoder.geometry_processing import (
    atoms_to_points_normals,
)
from torch_geometric.utils import to_dense_batch

#2
# 定义自定义数据集类
class CATHDataSet(Dataset):
    def __init__(self, datalist):
        self.datalist = datalist

    def __len__(self):
        # 返回数据集的长度
        return len(self.datalist)

    def __getitem__(self, idx):
        # 获取索引 idx 处的样本
        entry = self.datalist[idx]

        # 返回数据项，可以根据需求返回特定字段或者所有字段
        return {
            'title': entry['title'],
            'seq': entry['seq'],
            'coords': entry['coords'],
            'types': entry['types'],
            'center': entry['center'],
            'CA': entry['CA'],
            'C': entry['C'],
            'O': entry['O'],
            'N': entry['N']
        }


def iterate_surface_precompute(dataloader, device):
    processed_dataset = []
    alphabet = 'ACDEFGHIKLMNPQRSTVWY'
    alphabet_set = set([a for a in alphabet])
    with open('./data/chain_set.jsonl') as f:
        lines = f.readlines()
    data_list = []
    for line in tqdm(lines[:10]):
        entry = json.loads(line)
        seq = entry['seq']

        for key, val in entry['coords'].items():
            cleaned_val = [sublist for sublist in val if not all(math.isnan(x) for x in sublist)]
            entry['coords'][key] = np.asarray(cleaned_val)

        ca_type = torch.tile(torch.tensor([0, 1, 0, 0], device=device), (entry['coords']['CA'].shape[0], 1))
        c_type = torch.tile(torch.tensor([1, 0, 0, 0], device=device), (entry['coords']['C'].shape[0], 1))
        o_type = torch.tile(torch.tensor([0, 0, 1, 0], device=device), (entry['coords']['O'].shape[0], 1))
        n_type = torch.tile(torch.tensor([0, 0, 0, 1], device=device), (entry['coords']['N'].shape[0], 1))
        bad_chars = set([s for s in seq]).difference(alphabet_set)
        atoms = np.vstack((entry['coords']['CA'], entry['coords']['C'], entry['coords']['O'],
                           entry['coords']['N']))
        atoms_t = torch.tensor(atoms, device=device)
        # 使用 torch.isnan() 找到非 NaN 值的位置
        valid_mask = ~torch.isnan(atoms_t)
        atoms_T = atoms_t
        # 将 NaN 值替换为 0，以便在求和时不影响结果
        tensor_nan_to_zero = torch.where(valid_mask, atoms_T, torch.zeros_like(atoms_T))
        atoms_center = torch.tensor(tensor_nan_to_zero).mean(dim=-2, keepdim=True)

        if len(bad_chars) == 0:
            if len(entry['seq']) <= 500:
                data_list.append({
                    'title': entry['name'],
                    'seq': entry['seq'],
                    'coords': atoms,
                    'types': torch.cat((ca_type, c_type, o_type, n_type), dim=0),
                    'center': atoms_center,
                    'CA': entry['coords']['CA'],
                    'C': entry['coords']['C'],
                    'O': entry['coords']['O'],
                    'N': entry['coords']['N']
                })

    # for data in data_list:
    #     print(data['title'])
    #     print(data['seq'])
    #     print(data['CA'])
    #     print(data['coords'])

    print("debug")
    dataset = CATHDataSet(data_list)
    cath_dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # for it,data in enumerate(tqdm(cath_dataloader)):
    #     P = process(data)
    # for i in tqdm(range(len(data_list))):
    #     process(data_list[i])

    for it, data in enumerate(tqdm(dataloader)):
        protein = data[4].to(device)
        ligand_center = data[3].to(device)
        P = process(protein,ligand_center,data_list[0])
        data[4].gen_xyz = P["xyz"]
        data[4].gen_normals = P["normals"]
        data[4].gen_batch = P["batch"]
        processed_dataset.append(data)

    return processed_dataset

def Generate_Surface(data_item,device):
    PT = {}
    device=device
    PT["atoms"] = data_item["atom_xyz"]
    PT["batch_atoms"] = torch.zeros(data_item["atom_xyz"].size(0),device=device,dtype=torch.int)
    PT["atomtypes"] = data_item["atomtypes"]
    # Atom information:
    torch.set_printoptions(threshold=float('inf'))
    # print(protein_single.atom_coords_batch)
    # Chemical features: atom coordinates and types.
    if not "xyz" in data_item.keys():
        PT["xyz"], PT["normals"], PT["batch"] = atoms_to_points_normals(
            PT["atoms"],
            PT["batch_atoms"],
            atomtypes=PT["atomtypes"],
            num_atoms=4,
            resolution=1.0,
            sup_sampling=20,
            distance=1.05,
        )
        # print(PT["xyz"].shape)
        # center = data_item['center'].unsqueeze(0)
        # PT["xyz"], PT["normals"], PT["batch"] = select_pocket_random(PT)
    else:
        return data_item


    return PT

def process(protein_single, ligand_center, data_item,device):
# def process(data_item):
    P = {}
    PT = {}
    device = device
    PT["atoms"] = torch.tensor(data_item["coords"],device=device)
    lens = data_item["coords"].shape[0]
    PT["batch_atoms"] = torch.zeros(lens,device=device,dtype=torch.int)
    PT["atom_xyz"] = data_item["coords"]
    PT["atomtypes"] = data_item["types"]
    # Atom information:
    P["atoms"] = protein_single.atom_coords
    P["batch_atoms"] = protein_single.atom_coords_batch
    torch.set_printoptions(threshold=float('inf'))
    # print(protein_single.atom_coords_batch)
    # Chemical features: atom coordinates and types.
    P["atom_xyz"] = protein_single.atom_coords
    P["atomtypes"] = protein_single.atom_types

    if not "gen_xyz" in data_item.keys():
        PT["xyz"], PT["normals"], PT["batch"] = atoms_to_points_normals(
            PT["atoms"],
            PT["batch_atoms"],
            atomtypes=PT["atomtypes"],
            num_atoms=4,
            resolution=1.0,
            sup_sampling=20,
            distance=1.05,
        )
        center = data_item['center'].unsqueeze(0)
        PT["xyz"], PT["normals"], PT["batch"] = select_pocket(PT,center)
    else:
        PT["xyz"] = data_item.gen_xyz
        PT["normals"] = data_item.gen_normals
        PT["batch"] = data_item.gen_batch

    return PT


def extract_single(P_batch, number):
    P = {}
    suface_batch = P_batch["batch"] == number

    P["batch"] = P_batch["batch"][suface_batch]

    # Surface information:
    P["xyz"] = P_batch["xyz"][suface_batch]
    P["normals"] = P_batch["normals"][suface_batch]

    return P

def select_pocket_random(P_batch):
    surface_list = []
    batch_list = []
    normal_list = []
    protein_batch_size = P_batch["batch_atoms"][-1].item() + 1
    
    for i in range(protein_batch_size):
        P = extract_single(P_batch, i)
        
        # 生成随机索引（代替原距离排序逻辑）
        total_points = P["xyz"].shape[0]
        rand_indices = torch.randperm(total_points)[:1024]  # 随机排列取前512
        
        # 确保索引数量一致（当点数不足时自动截断）
        selected_indices = rand_indices[:1024] if total_points >=1024 else torch.arange(total_points)
        
        surface_list.append(P["xyz"][selected_indices])
        normal_list.append(P["normals"][selected_indices])
        batch_list.append(P["batch"][:selected_indices.shape[0]])

    p_xyz = torch.cat(surface_list, dim=0)
    p_normals = torch.cat(normal_list, dim=0)
    p_batch = torch.cat(batch_list, dim=0)
    
    return p_xyz, p_normals, p_batch

def select_pocket(P_batch, ligand_center):
    surface_list = []
    batch_list = []
    normal_list = []
    protein_batch_size = P_batch["batch_atoms"][-1].item() + 1
    for i in range(protein_batch_size):
        P = extract_single(P_batch, i)

        distances = torch.norm(P["xyz"] - ligand_center[i].squeeze(), dim=1)
        sorted_indices = torch.argsort(distances)
        point_nums = 512
        closest_protein_indices = sorted_indices[:point_nums]

        # Append surface embeddings and batches to the lists
        surface_list.append(P["xyz"][closest_protein_indices])
        normal_list.append(P["normals"][closest_protein_indices])
        batch_list.append(P["batch"][:closest_protein_indices.shape[0]])

    p_xyz = torch.cat(surface_list, dim=0)
    p_batch = torch.cat(batch_list, dim=0)
    p_normals = torch.cat(normal_list, dim=0)

    return p_xyz, p_normals, p_batch


def iterate(net, ligand_center, protein, data_item, device='cuda'):
    P_processed = process(protein, ligand_center, data_item)

    outputs = net(P_processed)
    surface_emb, mask = to_dense_batch(outputs["embedding"], outputs["batch"])

    return surface_emb

def iterate_Surf(net,data_item, device='cuda'):
    P_processed = data_item
    outputs = net(P_processed)
    # surface_emb, mask = to_dense_batch(outputs["embedding"], outputs["batch"].long())
    return outputs["embedding"]

class MultiGPU(torch.nn.Module):
    def __init__(self):
        super(MultiGPU, self).__init__()

    def forward(self, protein_single, ligand_center):
        return
