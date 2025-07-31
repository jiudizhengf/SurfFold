import os
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import argparse
import sys
from Bio import SeqIO, pairwise2
import torch
import torch.optim as optim
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from scipy.spatial import KDTree
import sys

from dataset.CATHdataset import CATHdataset

sys.path.insert(0, '..')
sys.path.insert(0, '../..')
from pronet import ProNet
from torch_geometric.data import DataLoader

import warnings
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns
import pandas as pd

#7

warnings.filterwarnings("ignore")

criterion = nn.CrossEntropyLoss()

seed = 6666
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def calc_identity(aln_seq1: str, aln_seq2: str) -> float:
    """
    计算对齐序列的 Percent Identity。
    matches / alignment_length × 100
    只在两侧都不是 '-' 时才计为匹配。
    """
    matches = sum(a == b for a, b in zip(aln_seq1, aln_seq2)
                  if a != '-' and b != '-')
    return matches / len(aln_seq1) * 100

def seq_similarity(seq1: str, seq2: str):
    """
    对两条序列计算并打印全局和局部比对的 % Identity 及对齐示例。
    """
    # 全局比对（Needleman–Wunsch, match=1, mismatch=0）
    aln_g = pairwise2.align.globalxx(seq1, seq2, one_alignment_only=True)[0]
    id_g = calc_identity(aln_g.seqA, aln_g.seqB)

    # 局部比对（Smith–Waterman, match=1, mismatch=0）
    aln_l = pairwise2.align.localxx(seq1, seq2, one_alignment_only=True)[0]
    id_l = calc_identity(aln_l.seqA, aln_l.seqB)
    return id_g, id_l

def train(args, model, loader, optimizer, device):
    model.train()
    loss_accum = 0
    preds = []
    functions = []
    #wrongs = ['2pbz.A', '1j77.A', '2m47.A', '4hg2.A']
    pbar = tqdm(loader, disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        #print(batch.id)
        #if any(wrong in batch.id for wrong in wrongs):
            #print('wrong')
            #continue
        # if any(wrong in batch.id for wrong in wrongs2):
        #     print('wrong2')
        batch = batch.to(device) 
        pred = model(batch)
        # preds.append(torch.argmax(pred, dim=1))
        function = batch.y.long()
        # functions.append(function)
        optimizer.zero_grad()
        loss = criterion(pred, function)
        loss.backward()
        optimizer.step()
        pbar.set_description('train loss: {:.4f}'.format(loss.item()))
        print("train_loss = {:.4f}".format(loss.item()))
        loss_accum += loss.item()

    # functions = torch.cat(functions, dim=0)
    # preds = torch.cat(preds, dim=0)
    # acc = torch.sum(preds == functions) / functions.shape[0]
    train_loss = loss_accum / (step + 1)
    train_perplexity = np.exp(train_loss)
    # print(wrongs)
    return train_loss, train_perplexity


def evaluation(args, model, loader, device):
    model.eval()
    loss_accum = 0
    preds = []
    functions = []
    pbar = tqdm(loader, disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        # pred = model(batch)
        try:    
            pred = model(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                print('\n forward error \n')
                raise (e)
            else:
                print('evaluation OOM')
            torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))

        function = batch.y.long()
        functions.append(function)
        loss = criterion(pred, function)
        pbar.set_description('val loss: {:.4f}'.format(loss.item()))
        print("val_loss = {:.4f}".format(loss.item()))
        loss_accum += loss.item()

    # functions = torch.cat(functions, dim=0)
    # preds = torch.cat(preds, dim=0)
    # acc = torch.sum(preds == functions) / functions.shape[0]
    val_loss = loss_accum / (step + 1)
    val_perplexity = np.exp(val_loss)
    return val_loss, val_perplexity
    #return loss_accum / (step + 1), acc.item()

def Test(args, model, loader, device,task='all'):
    model.eval()
    total_recovery = 0
    loss_accum = 0
    preds = []
    functions = []
    pbar = tqdm(loader, disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
        # pred = model(batch)
        try:    
            pred = model(batch)
        except RuntimeError as e:
            if "CUDA out of memory" not in str(e):
                print('\n forward error \n')
                raise (e)
            else:
                print('evaluation OOM')
            torch.cuda.empty_cache()
            continue
        preds.append(torch.argmax(pred, dim=1))

        function = batch.y.long()
        functions.append(function)
        loss = criterion(pred, function)
        loss_accum += loss.item()
        S_pred = torch.argmax(pred, dim=1)
        cmp = (S_pred == function)
        recovery_ = cmp.float().mean().cpu().numpy()
        pbar.set_description('test loss: {:.4f}'.format(loss.item()))
        # print("test_loss = {:.4f}".format(loss.item()))
        print("test_recovery = {:.4f}".format(recovery_.item()))
        total_recovery += recovery_

    # functions = torch.cat(functions, dim=0)
    # preds = torch.cat(preds, dim=0)
    # acc = torch.sum(preds == functions) / functions.shape[0]
    test_loss = loss_accum / (step + 1)
    test_perplexity = np.exp(test_loss)
    test_recovery = total_recovery / (step + 1)
    return test_loss, test_perplexity, test_recovery
    #return loss_accum / (step + 1), acc.item()

def extract_embeddings(model, loader, device):
    """提取模型中间层嵌入特征"""
    model.eval()
    pbar = tqdm(loader)
    with torch.no_grad():
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            try:
                if batch.seq.shape[0] >300:    
                    # 获取模型中间层嵌入
                    embedding = model.get_embeddings(batch)
                    label = batch.id
                    visualize_tsne(embedding,label ,perplexity=30, n_iter=1000, save_prefix='tsne_visualization.png')
                # 将嵌入转为CPU并添加到列表中
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    print('\n forward error \n')
                    raise (e)
                else:
                    print('OOM during embedding extraction')
                torch.cuda.empty_cache()
                continue
    # 将所有批次的嵌入和标签连接起来
    
def visualize_tsne(embeddings_dict,label ,perplexity=30, n_iter=1000, save_prefix='tsne_visualization'):
    """使用t-SNE可视化多个嵌入特征"""
    
    for feature_name, embedding_matrix in embeddings_dict.items():
        print(f"Performing t-SNE on '{feature_name}' with {embedding_matrix.shape[0]} samples and dimension {embedding_matrix.shape[1]}...")
        
        # 执行t-SNE降维
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
        tsne_results = tsne.fit_transform(embedding_matrix.cpu())
        
        # 创建DataFrame用于可视化
        df = pd.DataFrame({
            'x': tsne_results[:, 0], 
            'y': tsne_results[:, 1],
            'label': [feature_name] * embedding_matrix.shape[0]
        })
        
        # # 绘制t-SNE结果
        # plt.figure(figsize=(12, 10))
        # sns.scatterplot(x='x', y='y', data=df, palette='viridis', alpha=0.7, s=50)
        # plt.title(f't-SNE Visualization of {feature_name}')
        # plt.xlabel('t-SNE dimension 1')
        # plt.ylabel('t-SNE dimension 2')
        # save_path = f"{save_prefix}_{feature_name}.png"
        # plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # plt.show()
        # print(f"t-SNE visualization for '{feature_name}' saved to {save_path}")
         # 使用 JointGrid 绘制散点图与边缘分布曲线
        g = sns.JointGrid(data=df, x='x', y='y', space=0, height=12)
        g.plot_joint(sns.scatterplot, color='b', alpha=0.7, s=50)
        g.plot_marginals(sns.kdeplot, fill=True, alpha=0.5)
        
        # 设置标题与坐标轴标签
        g.ax_joint.set_xlabel('t-SNE dimension 1')
        g.ax_joint.set_ylabel('t-SNE dimension 2')
        g.figure.suptitle(f't-SNE Visualization of {feature_name}', y=1.02)
        
        # 保存并展示
        save_path = f"{save_prefix}_{label}_{feature_name}.png"
        g.figure.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"t-SNE visualization for '{feature_name}' with marginal distributions saved to {save_path}")


def main():
    ### Args
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=int, default=3, help='Device to use')
    parser.add_argument('--num_workers', type=int, default=5, help='Number of workers in Dataloader')

    ### Data
    parser.add_argument('--dataset', type=str, default='CATH4.2', help='CATH4.2')
    parser.add_argument('--dataset_path', type=str, default='./dataset',
                        help='path to load and process the CATH4.2')

    # HomologyTAPE augmentation tricks, see appendix E in the paper (https://openreview.net/pdf?id=9X-hgLDLYkQ)
    parser.add_argument('--mask', action='store_true', help='Random mask some node type')
    parser.add_argument('--noise', action='store_true', help='Add Gaussian noise to node coords')
    parser.add_argument('--deform', action='store_true', help='Deform node coords')
    parser.add_argument('--data_augment_eachlayer', action='store_true', help='Add Gaussian noise to features')
    parser.add_argument('--euler_noise', action='store_true', help='Add Gaussian noise Euler angles')
    parser.add_argument('--mask_aatype', type=float, default=0.2, help='Random mask aatype to 25(unknown:X) ratio')

    ### Model
    parser.add_argument('--level', type=str, default='allatom',
                        help='Choose from \'aminoacid\', \'backbone\', and \'allatom\' levels')
    parser.add_argument('--num_blocks', type=int, default=1, help='Model layers')
    parser.add_argument('--hidden_channels', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--out_channels', type=int, default=128)
    parser.add_argument('--fix_dist', action='store_true')
    parser.add_argument('--cutoff', type=float, default=10, help='Distance constraint for building the protein graph')
    parser.add_argument('--dropout', type=float, default=0.5, help='Dropout') 

    ### Training hyperparameter
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs to train')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_decay_step_size', type=int, default=5, help='Learning rate step size')
    parser.add_argument('--lr_decay_factor', type=float, default=0.5, help='Learning rate factor')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight Decay')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size during training')
    parser.add_argument('--eval_batch_size', type=int, default=16, help='Batch size')

    parser.add_argument('--continue_training', action='store_true')
    parser.add_argument('--save_dir', type=str, default='/trained_models_CATH4.2/allatom', help='Trained model path')
    parser.add_argument('--test_visualization',default=False)
    parser.add_argument('--disable_tqdm', default=False, action='store_true')
    parser.add_argument('--visualize_path', type=str, default='/home/ldr/ProNet_surf/trained_models_CATH4.2/allatom/visualization/best_val.pt', help='Trained model path')
    args = parser.parse_args()
    print(args)

    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")

    ##### load datasets
    print('Loading Train & Val & Test Data...')
    train_set = CATHdataset(root=args.dataset_path + '/CATH4.2', split='train')
    val_set = CATHdataset(root=args.dataset_path + '/CATH4.2', split='validation')
    test_set_all = CATHdataset(root=args.dataset_path + '/CATH4.2', split='test',task='all')
    test_set_single_chain = CATHdataset(root=args.dataset_path + '/CATH4.2', split='test',task='single_chain')
    test_set_short = CATHdataset(root=args.dataset_path + '/CATH4.2', split='test',task='short')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_all = DataLoader(test_set_all, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_single_chain = DataLoader(test_set_single_chain, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader_short = DataLoader(test_set_short, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers)
    print('Done!')
    print('Train, val, test_all, test_single_chain, test_short :', train_set, val_set, test_set_all, test_set_single_chain, test_set_short)
    ##### set up model
    model = ProNet(num_blocks=args.num_blocks, hidden_channels=args.hidden_channels, out_channels=args.out_channels,
                   cutoff=args.cutoff, dropout=args.dropout,
                   data_augment_eachlayer=args.data_augment_eachlayer,
                   euler_noise=args.euler_noise, level=args.level)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_decay_step_size, gamma=args.lr_decay_factor)
    if args.test_visualization:
        # 将氨基酸类型的单字母代码映射为数字
        amino_acid_to_number = {
            'A': 0, 'C': 1, 'D': 2, 'E': 3, 'F': 4, 'G': 5,
            'H': 6, 'I': 7, 'K': 8, 'L': 9, 'M': 10, 'N': 11,
            'P': 12, 'Q': 13, 'R': 14, 'S': 15, 'T': 16, 'V': 17,
            'W': 18, 'Y': 19
        }
        number_to_amino_acid = {num: aa for aa, num in amino_acid_to_number.items()}
        # 3. 定义转换函数
        def tensor_to_sequence(t: torch.Tensor) -> str:
            """
            将一个包含 [0-19] 的 1D Tensor 转成对应的氨基酸长字符串。
            """
            # 如果在 GPU 上，先移到 CPU；再转成 Python list
            nums = t.cpu().tolist()
            # 每个数字映射到一个字符，然后 join
            return ''.join(number_to_amino_acid[n] for n in nums)
        print('Extracting embeddings for visualization...')
        visualize_path = args.visualize_path
        checkpoint = torch.load(visualize_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        #embeddings = extract_embeddings(model, test_loader_all, device)

        model.eval()
        with open('sampled_1000_pairs_gaussian_uniform copy.txt', 'r') as f:
            lines = f.readlines()

        pbar = tqdm(test_loader_all, disable=args.disable_tqdm)
        for step, batch in enumerate(pbar):
            batch = batch.to(device)
            # pred = model(batch)
            try:    
                pred = model(batch)
            except RuntimeError as e:
                if "CUDA out of memory" not in str(e):
                    print('\n forward error \n')
                    raise (e)
                else:
                    print('evaluation OOM')
                torch.cuda.empty_cache()
                continue   
            function = batch.y.long()
            S_pred = torch.argmax(pred, dim=1)
            seq1 = tensor_to_sequence(function)
            seq2 = tensor_to_sequence(S_pred)
            with open('SurfNet_results', 'a') as f_out:
                f_out.write(f"{batch.id}\t"
                            f"{seq1}\t"
                            f"{seq2}\n")
        return    
    if args.continue_training:
        save_dir = args.save_dir
        checkpoint = torch.load(save_dir + '/best_val.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        save_dir = './trained_models_{dataset}/{level}/layer{num_blocks}_cutoff{cutoff}_hidden{hidden_channels}_batch{batch_size}_lr{lr}_{lr_decay_factor}_{lr_decay_step_size}_dropout{dropout}__{time}'.format(
            dataset=args.dataset, level=args.level,
            num_blocks=args.num_blocks, cutoff=args.cutoff, hidden_channels=args.hidden_channels,
            batch_size=args.batch_size,
            lr=args.lr, lr_decay_factor=args.lr_decay_factor, lr_decay_step_size=args.lr_decay_step_size,
            dropout=args.dropout, time=datetime.now())
        print('saving to...', save_dir)
        start_epoch = 1

    num_params = sum(p.numel() for p in model.parameters())

    print('num_parameters:', num_params)
    writer = SummaryWriter(log_dir=save_dir)
    best_val_perplexity = 100
    train_loss_list=[]
    train_perplexity_list=[]
    val_loss_list=[]
    val_perplexity_list=[]
    test_loss_list_all=[]
    test_perplexity_list_all=[]
    test_recovery_list_all=[]
    test_loss_list_single_chain=[]
    test_perplexity_list_single_chain=[]
    test_recovery_list_single_chain=[]  
    test_loss_list_short=[]
    test_perplexity_list_short=[]
    test_recovery_list_short=[]

    for epoch in range(start_epoch, args.epochs + 1):
        print('==== Epoch {} ===='.format(epoch))
        t_start = time.perf_counter()

        train_loss, train_perplexity = train(args, model, train_loader, optimizer, device)
        t_end_train = time.perf_counter()
        val_loss, val_perplexity = evaluation(args, model, val_loader, device)
        t_start_test = time.perf_counter()
        test_loss_all, test_perplexity_all, test_recovery_all = Test(args, model, test_loader_all, device,task='all')
        test_loss_single, test_perplexity_single, test_recovery_single = Test(args, model, test_loader_single_chain, device,task='single_chain')
        test_loss_short, test_perplexity_short, test_recovery_short = Test(args, model, test_loader_short, device,task='short')
        train_loss_list.append(train_loss)
        train_perplexity_list.append(train_perplexity)
        val_loss_list.append(val_loss)
        val_perplexity_list.append(val_perplexity)
        test_loss_list_all.append(test_loss_all)
        test_perplexity_list_all.append(test_perplexity_all)
        test_recovery_list_all.append(test_recovery_all)
        test_loss_list_single_chain.append(test_loss_single)
        test_perplexity_list_single_chain.append(test_perplexity_single)
        test_recovery_list_single_chain.append(test_recovery_single)
        test_loss_list_short.append(test_loss_short)
        test_perplexity_list_short.append(test_perplexity_short)
        test_recovery_list_short.append(test_recovery_short) 
        t_end_test = time.perf_counter()
        
        print('Test: Loss:{:.6f} Perplexity:{:.4f} Recovery:{:.4f}'.format(test_loss_all, test_perplexity_all, test_recovery_all))
        if not save_dir == "" and not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not save_dir == "" and val_perplexity < best_val_perplexity:
            print('Saving best val checkpoint ...')
            checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
            # checkpoint = {'visualization Test'}
            torch.save(checkpoint, save_dir + '/best_val.pt')
            best_val_perplexity = val_perplexity
        t_end = time.perf_counter()
        print('Train: Loss:{:.6f} Perplexity:{:.4f}, Validation: Loss:{:.6f} Perplexity:{:.4f},' \
        'Test all Loss:{:.6f},Test all Perplexity :{:.4f},Test all Recovery:{:.4f},' \
        'Test single chain Perplexity :{:.4f},Test single chain Recovery:{:.4f},' \
        'Test short Perplexity :{:.4f},Test short Recovery:{:.4f}, ' \
        'time:{}, train_time:{}, test_time:{}'.format(
            train_loss, train_perplexity, val_loss, val_perplexity, test_loss_all, test_perplexity_all, test_recovery_all,test_perplexity_single, test_recovery_single,test_perplexity_short, test_recovery_short, t_end - t_start, t_end_train - t_start, t_end_test - t_start_test))
        if optimizer.param_groups[0]['lr'] > 1e-6:
            scheduler.step()
        print('Learning rate:', optimizer.param_groups[0]['lr'])
    plot_training_metrics(
        train_loss_list,
        train_perplexity_list,
        val_loss_list,
        val_perplexity_list,
        test_loss_list_all,
        test_perplexity_list_all,
        test_recovery_list_all
    )
    print("all:{:.6f}".format(max(test_recovery_list_all)))
    print("single_chain:{:.6f}".format(max(test_recovery_list_single_chain)))
    print("short:{:.6f}".format(max(test_recovery_list_short)))
    #writer.close()
    # Save last model
    checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(),
                  'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
    torch.save(checkpoint, save_dir + "/epoch{}.pt".format(epoch))

def plot_training_metrics(
    train_loss_list,
    train_perplexity_list,
    val_loss_list,
    val_perplexity_list,
    test_loss_list,
    test_perplexity_list,
    test_recovery_list,
    figsize=(15, 18)
):
    """
    绘制训练过程多维指标可视化图表
    
    参数：
    - figsize: 图表尺寸（默认 15x18 英寸）
    - 所有*_list参数应为等长列表，表示每个epoch的指标值
    """
    
    plt.figure(figsize=figsize)
    epochs = range(1, len(train_loss_list) + 1)
    
    # 损失对比子图
    plt.subplot(3, 1, 1)
    plt.plot(epochs, train_loss_list, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss_list, 'g--', label='Validation Loss') 
    plt.plot(epochs, test_loss_list, 'r:', label='Test Loss')
    plt.title('Loss Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 困惑度对比子图
    plt.subplot(3, 1, 2)
    plt.plot(epochs, train_perplexity_list, 'b-', label='Train Perplexity')
    plt.plot(epochs, val_perplexity_list, 'g--', label='Validation Perplexity')
    plt.plot(epochs, test_perplexity_list, 'r:', label='Test Perplexity')
    plt.title('Perplexity Comparison')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.yscale('log')  # 困惑度通常用对数尺度更清晰
    plt.grid(True, which='both', alpha=0.3)
    plt.legend()
    
    # 测试恢复率子图
    plt.subplot(3, 1, 3)
    plt.plot(epochs, test_recovery_list, 'm-', label='Test Recovery Rate')
    plt.title('Test Recovery Progress')
    plt.xlabel('Epochs')
    plt.ylabel('Recovery Rate')
    plt.ylim(0, 1.05)  # 假设恢复率是0-1之间的值
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('data-with-cur-large_surf.png')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
