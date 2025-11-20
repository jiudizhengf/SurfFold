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

def train(args, model, loader, optimizer, device):
    model.train()
    loss_accum = 0
    preds = []
    functions = []
    pbar = tqdm(loader, disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device) 
        pred = model(batch)
        function = batch.y.long()
        optimizer.zero_grad()
        loss = criterion(pred, function)
        loss.backward()
        optimizer.step()
        pbar.set_description('train loss: {:.4f}'.format(loss.item()))
        print("train_loss = {:.4f}".format(loss.item()))
        loss_accum += loss.item()

    train_loss = loss_accum / (step + 1)
    train_perplexity = np.exp(train_loss)
    return train_loss, train_perplexity


def evaluation(args, model, loader, device):
    model.eval()
    loss_accum = 0
    preds = []
    functions = []
    pbar = tqdm(loader, disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
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
    val_loss = loss_accum / (step + 1)
    val_perplexity = np.exp(val_loss)
    return val_loss, val_perplexity

def Test(args, model, loader, device,task='all'):
    model.eval()
    total_recovery = 0
    loss_accum = 0
    preds = []
    functions = []
    pbar = tqdm(loader, disable=args.disable_tqdm)
    for step, batch in enumerate(pbar):
        batch = batch.to(device)
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

    test_loss = loss_accum / (step + 1)
    test_perplexity = np.exp(test_loss)
    test_recovery = total_recovery / (step + 1)
    return test_loss, test_perplexity, test_recovery

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
    parser.add_argument('--visualize_path', type=str, help='Trained model path')
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
    
    best_val_perplexity = float('inf') 
    best_test_recovery_all_record = 0.0
    best_test_recovery_single_record = 0.0
    best_test_recovery_short_record = 0.0
    
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
        if val_perplexity < best_val_perplexity:
            print('Validation perplexity improved from {:.4f} to {:.4f}'.format(best_val_perplexity, val_perplexity))
            best_val_perplexity = val_perplexity
            
            # 关键步骤：记录当前 Epoch 的测试集成绩
            best_test_recovery_all_record = test_recovery_all
            best_test_recovery_single_record = test_recovery_single
            best_test_recovery_short_record = test_recovery_short
            
            if not save_dir == "":
                print('Saving best val checkpoint ...')
                checkpoint = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict()}
                torch.save(checkpoint, save_dir + '/best_val.pt')
                
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
    print("-" * 30)
    print("Final Results (Selected by Best Validation Perplexity):")
    print("Best Val Perplexity: {:.4f}".format(best_val_perplexity))
    print("Test all Recovery (at best val): {:.6f}".format(best_test_recovery_all_record))
    print("Test single_chain Recovery (at best val): {:.6f}".format(best_test_recovery_single_record))
    print("Test short Recovery (at best val): {:.6f}".format(best_test_recovery_short_record))
    print("-" * 30)
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
