#!/usr/bin/env python3
"""
Versatile Incremental Learning (VIL) training and evaluation script.
This script supports VIL scenarios where both domain and class distributions change across tasks.
"""
import argparse
import os
import time
import datetime
import random

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from continual_datasets.dataset_utils import set_data_config
from continual_datasets.build_incremental_scenario import build_continual_dataloader
from learners.prompt import CODAPrompt, OSPrompt, L2P
from utils.metric import accuracy, AverageMeter

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Versatile Incremental Learning (VIL) main script"
    )
    parser.add_argument('--config', type=str,
                        help='YAML config file path')
    parser.add_argument('--verbose', action='store_true',
                        help='print detailed logs and save heatmaps')
    parser.add_argument('--log_dir', type=str, default='outputs/vil',
                        help='directory for logs and outputs')
    parser.add_argument('--IL_mode', type=str, default='vil', choices=['vil'],
                        help='incremental learning mode')
    parser.add_argument('--dataset', type=str, default='iDigits',
                        help='dataset name for VIL scenario')
    parser.add_argument('--data_path', type=str, default='data',
                        help='root directory for dataset')
    parser.add_argument('--num_tasks', type=int, required=True,
                        help='total number of tasks')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch size for all tasks')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='number of data loader workers')
    parser.add_argument('--epochs', type=int, default=10,
                        help='number of epochs per task')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='optimizer weight decay')
    parser.add_argument('--optimizer', type=str, default='SGD', choices=['SGD', 'Adam'],
                        help='optimizer type')
    parser.add_argument('--schedule', nargs='+', type=int, default=[],
                        help='learning rate scheduler milestones')
    parser.add_argument('--schedule_type', type=str, default='cosine', choices=['cosine', 'decay'],
                        help='scheduler type')
    parser.add_argument('--model_type', type=str, required=True,
                        help='model type (e.g., zoo)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='backbone model name')
    parser.add_argument('--prompt_param', nargs='+', type=int, default=[1, 1, 1],
                        help='prompt parameters: pool size, prompt length, global prompt length')
    parser.add_argument('--query', type=str, default='vit',
                        help='query type for prompt')
    parser.add_argument(
        '--learners', nargs='+', type=str,
        choices=['CODAPrompt', 'OSPrompt', 'L2P'],
        default=['CODAPrompt', 'OSPrompt'],
        help='which prompt learner classes to run (e.g., CODAPrompt, OSPrompt, L2P)')
    parser.add_argument('--gpuid', type=int, default=0,
                        help='GPU id (negative for CPU)')
    # 데이터 로딩 관련 추가 매개변수들
    parser.add_argument('--shuffle', type=bool, default=False)
    parser.add_argument('--pin_memory', type=bool, default=True)
    parser.add_argument('--seed', type=int, default=0)

    
    args = parser.parse_args()
    args.develop_tasks = False

    if args.config:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
        for k, v in cfg.items():
            setattr(args, k, v)
    return args


def save_accuracy_heatmap(matrix, task_id, args):
    if plt is None:
        return
    os.makedirs(args.log_dir, exist_ok=True)
    fig, ax = plt.subplots()
    cax = ax.matshow(matrix, cmap='viridis', vmin=0, vmax=100)
    fig.colorbar(cax)
    ax.set_title(f'Accuracy Heatmap Task {task_id+1}')
    ax.set_xlabel('Task')
    ax.set_ylabel('Task')
    ax.set_xticks(np.arange(matrix.shape[1]))
    ax.set_yticks(np.arange(matrix.shape[0]))
    plt.tight_layout()
    plt.savefig(os.path.join(args.log_dir, f'heatmap_task_{task_id+1}.png'))
    plt.close(fig)


class VILRunner:
    def __init__(self, args):
        self.args = args

    def train_one_epoch(self, model, criterion, train_loader, optimizer, device, epoch):
        model.train()
        losses = AverageMeter()
        accs = AverageMeter()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits, prompt_loss = model.model(x, train=True)
            loss = criterion(logits, y) + prompt_loss.sum()
            loss.backward()
            optimizer.step()
            acc = accuracy(logits, y)
            losses.update(loss.item(), y.size(0))
            accs.update(acc, y.size(0))
        return losses.avg, accs.avg

    def evaluate_task(self, model, val_loader, device, task_id, class_mask):
        model.eval()
        accs = AverageMeter()
        classes = class_mask[task_id]
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mask = torch.zeros_like(y, dtype=torch.bool)
                for c in classes:
                    mask |= (y == c)
                if not mask.any():
                    continue
                x_sel = x[mask]
                y_sel = y[mask]
                logits, _ = model.model(x_sel, train=False)
                idx = torch.tensor(classes, device=device)
                out_sel = logits[:, idx]
                index_map = {c:i for i,c in enumerate(classes)}
                y_rel = torch.tensor([index_map[int(k)] for k in y_sel.cpu()], device=device)
                accs.update(accuracy(out_sel, y_rel), y_sel.size(0))
        return accs.avg

    def evaluate_till_now(self, model, data_loader, device, task_id, class_mask, acc_matrix):
        args = self.args
        for t in range(task_id + 1):
            acc_matrix[t, task_id] = self.evaluate_task(
                model, data_loader[t]['val'], device, t, class_mask)
        A_i = [np.mean(acc_matrix[: i + 1, i]) for i in range(task_id + 1)]
        A_last = A_i[-1]
        A_avg = np.mean(A_i)
        result_str = (
            f"[Average accuracy till task{task_id+1}] "
            f"A_last: {A_last:.2f} A_avg: {A_avg:.2f}"
        )
        if task_id > 0:
            forgetting = np.mean(
                (np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id]
            )
            result_str += f" Forgetting: {forgetting:.4f}"
        print(result_str)
        if args.verbose:
            sub = acc_matrix[: task_id + 1, : task_id + 1]
            mat = np.where(np.triu(np.ones_like(sub, dtype=bool)), sub, np.nan)
            save_accuracy_heatmap(mat, task_id, args)
        return {"Acc@1": A_last}

    def train_and_evaluate(
        self, model, criterion, data_loader, optimizer, lr_scheduler, device, class_mask
    ):
        args = self.args
        acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
        seen_classes = set()  # 이미 본 클래스 추적
        
        for task_id in range(args.num_tasks):
            print(f"{f'Training on Task {task_id+1}/{args.num_tasks}':=^60}")
            
            # 새로 보는 클래스만 valid_output_dim 증가
            unseen = 0
            for c in class_mask[task_id]:
                if c not in seen_classes:
                    unseen += 1
                    seen_classes.add(c)
            
            if unseen > 0:
                model.add_valid_output_dim(unseen)
                
            train_start = time.time()
            for epoch in range(args.epochs):
                epoch_start = time.time()
                loss_avg, acc_avg = self.train_one_epoch(
                    model,
                    criterion,
                    data_loader[task_id]['train'],
                    optimizer,
                    device,
                    epoch,
                )
                duration = time.time() - epoch_start
                print(
                    f"Epoch [{epoch+1}/{args.epochs}] Completed in "
                    f"{str(datetime.timedelta(seconds=int(duration)))}: "
                    f"Avg Loss = {loss_avg:.4f}, Avg Acc@1 = {acc_avg:.2f}"
                )
                if lr_scheduler is not None:
                    lr_scheduler.step(epoch)
            train_time = time.time() - train_start
            print(
                f"Task {task_id+1} training completed in "
                f"{str(datetime.timedelta(seconds=int(train_time)))}"
            )
            print(f"{f'Testing on Task {task_id+1}/{args.num_tasks}':=^60}")
            eval_start = time.time()
            self.evaluate_till_now(
                model, data_loader, device, task_id, class_mask, acc_matrix
            )
            eval_time = time.time() - eval_start
            print(
                f"Task {task_id+1} evaluation completed in "
                f"{str(datetime.timedelta(seconds=int(eval_time)))}"
            )
        return acc_matrix


def main():
    args = parse_args()
    os.makedirs(args.log_dir, exist_ok=True)
    # set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.gpuid >= 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args = set_data_config(args)
    data_loader, class_mask, domain_list = build_continual_dataloader(args)

    # 사용 가능한 GPU 확인
    if args.gpuid >= 0 and torch.cuda.is_available():
        device = torch.device(f'cuda:{args.gpuid}')
        num_gpus = torch.cuda.device_count()
        print(f"사용 가능한 GPU 개수: {num_gpus}")
    else:
        device = torch.device('cpu')
        num_gpus = 0
        print("GPU를 사용할 수 없어 CPU로 실행합니다.")

    runner = VILRunner(args)

    learner_map = {
        'CODAPrompt': CODAPrompt,
        'OSPrompt': OSPrompt,
        'L2P': L2P,
    }
    model_classes = [learner_map[name] for name in args.learners]
    for ModelClass in model_classes:
        print('=' * 20 + f' Model: {ModelClass.__name__} ' + '=' * 20)
        cfg = {
            'num_classes': args.num_classes,
            'lr': args.lr,
            'debug_mode': False,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'schedule': args.schedule,
            'schedule_type': args.schedule_type,
            'model_type': args.model_type,
            'model_name': args.model_name,
            'optimizer': args.optimizer,
            'gpuid': [args.gpuid],
            'memory': 0,
            'temp': 2.0,
            'out_dim': args.num_classes,
            'overwrite': False,
            'DW': False,
            'batch_size': args.batch_size,
            'upper_bound_flag': False,
            'tasks': None,
            'top_k': 1,
            'prompt_param': [args.num_tasks] + args.prompt_param,
            'query': args.query,
        }
        model = ModelClass(cfg)
        criterion = nn.CrossEntropyLoss()
        optimizer = model.optimizer
        lr_scheduler = model.scheduler
        model = model.to(device)  # 모델을 디바이스로 이동
        
        # 멀티 GPU 지원 설정
        if device.type.startswith('cuda') and num_gpus > 1:
            print(f"{num_gpus}개의 GPU를 병렬로 사용합니다.")
            model = nn.DataParallel(model)

        acc_matrix = runner.train_and_evaluate(
            model,
            criterion,
            data_loader,
            optimizer,
            lr_scheduler,
            device,
            class_mask,
        )
        np.save(
            os.path.join(
                args.log_dir, f'acc_matrix_{ModelClass.__name__}.npy'
            ),
            acc_matrix,
        )


if __name__ == '__main__':
    main()