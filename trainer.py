import os
import sys
import argparse
import torch
import numpy as np
import random
from random import shuffle
from collections import OrderedDict
import dataloaders
from dataloaders.utils import *
from continual_datasets.dataset_utils import set_data_config, RandomSampleWrapper, get_ood_dataset
from continual_datasets.build_incremental_scenario import build_continual_dataloader
import torch.nn.functional as F
import time
import datetime
from sklearn import metrics
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import learners

class Trainer:

    def __init__(self, args, seed, metric_keys, save_keys):

        # process inputs
        self.seed = seed
        self.metric_keys = metric_keys
        self.save_keys = save_keys
        self.log_dir = args.log_dir
        self.batch_size = args.batch_size
        self.workers = args.workers
        self.IL_mode = args.IL_mode
        self.vil_dataloader = False
        self.develop = args.develop
        # model load directory
        self.model_top_dir = args.log_dir

        # VIL/DIL/CIL/JOINT scenarios via continual_datasets
        if args.dataset in ['iDigits', 'DomainNet', 'CORe50', 'CLEAR']:
            self.vil_dataloader = True
            args = set_data_config(args)
            args.data_path = args.dataroot
            args.num_workers = args.workers
            self.dataloader, self.class_mask, self.domain_list = build_continual_dataloader(args)
            self.num_tasks = args.num_tasks
            self.task_names = [str(i+1) for i in range(self.num_tasks)]
            self.grayscale_vis = False
            self.top_k = 1

            self.learner_config = {
                'num_classes': args.num_classes,
                'lr': args.lr,
                'debug_mode': args.debug_mode == 1,
                'momentum': args.momentum,
                'weight_decay': args.weight_decay,
                'schedule': args.schedule,
                'schedule_type': args.schedule_type,
                'model_type': args.model_type,
                'model_name': args.model_name,
                'optimizer': args.optimizer,
                'gpuid': args.gpuid,
                'memory': args.memory,
                'temp': args.temp,
                'out_dim': args.num_classes,
                'overwrite': args.overwrite == 1,
                'DW': args.DW,
                'batch_size': args.batch_size,
                'upper_bound_flag': args.upper_bound_flag,
                'tasks': self.class_mask,
                'top_k': self.top_k,
                'prompt_param': [self.num_tasks, args.prompt_param],
                'query': args.query,
                'IL_mode': args.IL_mode
            }
            print(self.learner_config)
            self.learner_type, self.learner_name = args.learner_type, args.learner_name
            self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

            if self.vil_dataloader:
                self.learner.add_valid_output_dim(args.num_classes)
            return
        # select dataset
        self.grayscale_vis = False
        self.top_k = 1
        if args.dataset == 'CIFAR10':
            Dataset = dataloaders.iCIFAR10
            num_classes = 10
            self.dataset_size = [32,32,3]
        elif args.dataset == 'CIFAR100':
            Dataset = dataloaders.iCIFAR100
            num_classes = 100
            self.dataset_size = [32,32,3]
        elif args.dataset == 'ImageNet_R':
            Dataset = dataloaders.iIMAGENET_R
            num_classes = 200
            self.dataset_size = [224,224,3]
            self.top_k = 1
        else:
            raise ValueError('Dataset not implemented!')

        # upper bound flag
        if args.upper_bound_flag:
            args.other_split_size = num_classes
            args.first_split_size = num_classes

        # load tasks
        class_order = np.arange(num_classes).tolist()
        class_order_logits = np.arange(num_classes).tolist()
        if self.seed > 0 and args.rand_split:
            print('=============================================')
            print('Shuffling....')
            # print('pre-shuffle:' + str(class_order))
            random.seed(self.seed)
            random.shuffle(class_order)
            # print('post-shuffle:' + str(class_order))
            print('=============================================')
        self.tasks = []
        self.tasks_logits = []
        p = 0
        while p < num_classes and (args.max_task == -1 or len(self.tasks) < args.max_task):
            inc = args.other_split_size if p > 0 else args.first_split_size
            self.tasks.append(class_order[p:p+inc])
            self.tasks_logits.append(class_order_logits[p:p+inc])
            p += inc
        self.num_tasks = len(self.tasks)
        self.task_names = [str(i+1) for i in range(self.num_tasks)]

        # number of tasks to perform
        if args.max_task > 0:
            self.max_task = min(args.max_task, len(self.task_names))
        else:
            self.max_task = len(self.task_names)

        # datasets and dataloaders
        k = 1 # number of transforms per image
        if args.model_name.startswith('vit'):
            resize_imnet = True
        else:
            resize_imnet = False



        train_transform = dataloaders.utils.get_transform(dataset=args.dataset, phase='train', aug=args.train_aug, resize_imnet=resize_imnet)
        test_transform  = dataloaders.utils.get_transform(dataset=args.dataset, phase='test', aug=args.train_aug, resize_imnet=resize_imnet)

        self.train_dataset = Dataset(args.dataroot, train=True, lab = True, tasks=self.tasks,
                            download_flag=True, transform=train_transform, 
                            seed=self.seed, rand_split=args.rand_split, validation=args.validation)
        self.test_dataset  = Dataset(args.dataroot, train=False, tasks=self.tasks,
                                download_flag=False, transform=test_transform, 
                                seed=self.seed, rand_split=args.rand_split, validation=args.validation)

        # for oracle
        self.oracle_flag = args.oracle_flag
        self.add_dim = 0

        # Prepare the self.learner (model)
        self.learner_config = {'num_classes': num_classes,
                        'lr': args.lr,
                        'debug_mode': args.debug_mode == 1,
                        'momentum': args.momentum,
                        'weight_decay': args.weight_decay,
                        'schedule': args.schedule,
                        'schedule_type': args.schedule_type,
                        'model_type': args.model_type,
                        'model_name': args.model_name,
                        'optimizer': args.optimizer,
                        'gpuid': args.gpuid,
                        'memory': args.memory,
                        'temp': args.temp,
                        'out_dim': num_classes,
                        'overwrite': args.overwrite == 1,
                        'DW': args.DW,
                        'batch_size': args.batch_size,
                        'upper_bound_flag': args.upper_bound_flag,
                        'tasks': self.tasks_logits,
                        'top_k': self.top_k,
                        'prompt_param':[self.num_tasks,args.prompt_param],
                        'query': args.query
                        }
        self.learner_type, self.learner_name = args.learner_type, args.learner_name
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

    def task_eval(self, t_index, local=False, task='acc'):

        val_name = self.task_names[t_index]
        # print('validation split name:', val_name)
        
        # eval
        if self.vil_dataloader:
            test_loader = self.dataloader[t_index]['val']
            # In VIL setting we are interested in the global classification   
            # accuracy across all classes observed so far. `task_global=True`  
            # keeps the whole classifier active while still allowing us to  
            # select only the relevant targets for the current validation  
            # split. When `local=True` we additionally report the task–local
            # accuracy (restricted output space) to facilitate comparison.
            if local:
                # task-local accuracy (restricted to task_in)
                return self.learner.validation(
                    test_loader,
                    task_in=self.class_mask[t_index],
                    task_global=False,
                    task_metric=task,
                )
            else:
                # task-global accuracy (full output space)
                return self.learner.validation(
                    test_loader,
                    task_in=self.class_mask[t_index],
                    task_global=True,
                    task_metric=task,
                )
        """
        self.test_dataset.load_dataset(t_index, train=True)
        test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
        if local:
            return self.learner.validation(test_loader, task_in = self.tasks_logits[t_index], task_metric=task)
        else:
            return self.learner.validation(test_loader, task_metric=task)
        """

    def train(self, avg_metrics):

        # VIL/DIL/CIL/JOINT training using continual_datasets output
        if self.vil_dataloader:
            print("DEBUG: VIL 모드로 학습을 시작합니다.")
            acc_matrix = np.zeros((self.num_tasks, self.num_tasks))
            for i in range(self.num_tasks):
                self.current_t_index = i
                train_loader = self.dataloader[i]['train']
                test_loader = self.dataloader[i]['val']
                task_name = self.task_names[i]
                print(f"{'='*20} Task {task_name} {'='*20}")
                """
                model_save_dir = os.path.join(self.model_top_dir, f'models/repeat-{self.seed+1}/task-{task_name}/')
                os.makedirs(model_save_dir, exist_ok=True)
                """
                model_save_dir = None
                try:
                    self.learner.model.module.task_id = i
                except Exception:
                    self.learner.model.task_id = i
                if i > 0:
                    try:
                        if self.learner.model.module.prompt is not None:
                            self.learner.model.module.prompt.process_task_count()
                    except Exception:
                        if self.learner.model.prompt is not None:
                            self.learner.model.prompt.process_task_count()
                avg_train_time = self.learner.learn_batch(train_loader, None, model_save_dir, test_loader, develop=self.develop)
                """
                self.learner.save_model(model_save_dir)
                """
                if avg_train_time is not None:
                    avg_metrics['time']['global'][i] = avg_train_time

                # Compute global accuracy for all (seen) tasks so far
                for t in range(i + 1):
                    acc_matrix[t, i] = self.learner.validation(
                        self.dataloader[t]['val'],
                        task_in=self.class_mask[t],
                        task_global=True,
                    )
                self.evaluate_till_now(acc_matrix, i)

                if args.ood_dataset:
                    print(f"{'OOD Evaluation':=^60}")
                    ood_start = time.time()
                    all_id_datasets = torch.utils.data.ConcatDataset([self.dataloader[t]['val'].dataset for t in range(i+1)])
                    ood_dataset = get_ood_dataset(args.ood_dataset, args)
                    device = next(self.learner.model.parameters()).device
                    self.evaluate_ood(self.learner.model, all_id_datasets, ood_dataset, device, args, i)
                    ood_duration = time.time() - ood_start
                    print(f"OOD evaluation after Task {i+1} completed in {str(datetime.timedelta(seconds=int(ood_duration)))}")

            return avg_metrics

        """
        # temporary results saving
        temp_table = {}
        for mkey in self.metric_keys: temp_table[mkey] = []
        temp_dir = self.log_dir + '/temp/'
        if not os.path.exists(temp_dir): os.makedirs(temp_dir)

        # for each task
        for i in range(self.max_task):

            # save current task index
            self.current_t_index = i

            # print name
            train_name = self.task_names[i]
            print('======================', train_name, '=======================')

            # load dataset for task
            task = self.tasks_logits[i]
            if self.oracle_flag:
                self.train_dataset.load_dataset(i, train=False)
                self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)
                self.add_dim += len(task)
            else:
                self.train_dataset.load_dataset(i, train=True)
                self.add_dim = len(task)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # add valid class to classifier
            self.learner.add_valid_output_dim(self.add_dim)

            # load dataset with memory
            self.train_dataset.append_coreset(only=False)

            # load dataloader
            train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True, num_workers=int(self.workers))

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # learn
            self.test_dataset.load_dataset(i, train=False)
            test_loader  = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, drop_last=False, num_workers=self.workers)
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            if not os.path.exists(model_save_dir): os.makedirs(model_save_dir)
            avg_train_time = self.learner.learn_batch(train_loader, self.train_dataset, model_save_dir, test_loader)

            # save model
            self.learner.save_model(model_save_dir)
            
            # evaluate acc
            # acc_table = []
            # acc_table_ssl = []
            # self.reset_cluster_labels = True
            # for j in range(i+1):
            #     acc_table.append(self.task_eval(j))
            # temp_table['acc'].append(np.mean(np.asarray(acc_table)))
            # temp_table['acc'].append(np.mean(np.asarray(acc_table)))

            # save temporary acc results
            # for mkey in ['acc']:
            #     save_file = temp_dir + mkey + '.csv'
            #     np.savetxt(save_file, np.asarray(temp_table[mkey]), delimiter=",", fmt='%.2f')

            if avg_train_time is not None: avg_metrics['time']['global'][i] = avg_train_time

        return avg_metrics
        """
    
    def summarize_acc(self, acc_dict, acc_table, acc_table_pt):

        # unpack dictionary
        avg_acc_all = acc_dict['global']
        avg_acc_pt = acc_dict['pt']
        avg_acc_pt_local = acc_dict['pt-local']

        # Calculate average performance across self.tasks
        # Customize this part for a different performance metric
        if self.vil_dataloader:
            avg_acc_history = [0] * self.num_tasks
            acc_matrix = np.zeros((self.num_tasks, self.num_tasks))
            for i in range(self.num_tasks):
                train_name = self.task_names[i]
                cls_acc_sum = 0
                for j in range(i+1):
                    val_name = self.task_names[j]
                    acc_matrix[i,j] = acc_table[val_name][train_name]
                    cls_acc_sum += acc_table[val_name][train_name]
                    avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                    avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
                avg_acc_history[i] = cls_acc_sum / (i + 1)
            
            # get forget score
            f_score = self.cal_fscore(acc_matrix)
            
            # Gather the final avg accuracy
            avg_acc_all[:,self.seed] = avg_acc_history
            
            # repack dictionary and return
            return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}, f_score
        
        """
        avg_acc_history = [0] * self.max_task
        acc_matrix = np.zeros((self.max_task, self.max_task))
        for i in range(self.max_task):
            train_name = self.task_names[i]
            cls_acc_sum = 0
            for j in range(i+1):
                val_name = self.task_names[j]
                acc_matrix[i,j] = acc_table[val_name][train_name]
                cls_acc_sum += acc_table[val_name][train_name]
                avg_acc_pt[j,i,self.seed] = acc_table[val_name][train_name]
                avg_acc_pt_local[j,i,self.seed] = acc_table_pt[val_name][train_name]
            avg_acc_history[i] = cls_acc_sum / (i + 1)

        # get forget score
        f_score = self.cal_fscore(acc_matrix)

        # Gather the final avg accuracy
        avg_acc_all[:,self.seed] = avg_acc_history

        # repack dictionary and return
        return {'global': avg_acc_all,'pt': avg_acc_pt,'pt-local': avg_acc_pt_local}, f_score
        """

    def evaluate(self, avg_metrics):
        # VIL/DIL/CIL/JOINT evaluation using continual_datasets output
        if self.vil_dataloader:
            print("DEBUG: VIL 모드로 평가를 시작합니다.")
            metric_table = {}
            metric_table_local = {}
            for mkey in self.metric_keys:
                metric_table[mkey] = {}
                metric_table_local[mkey] = {}

            acc_matrix = np.zeros((self.num_tasks, self.num_tasks))
            for i in range(self.num_tasks):
                task_name = self.task_names[i]
                print(f"{'='*20} Evaluating Task {task_name} {'='*20}")
                try:
                    self.learner.model.module.task_id = i
                except Exception:
                    self.learner.model.task_id = i
                if i > 0:
                    try:
                        if self.learner.model.module.prompt is not None:
                            self.learner.model.module.prompt.process_task_count()
                    except Exception:
                        if self.learner.model.prompt is not None:
                            self.learner.model.prompt.process_task_count()
                """
                model_save_dir = os.path.join(self.model_top_dir, f'models/repeat-{self.seed+1}/task-{task_name}/')
                self.learner.task_count = i
                self.learner.pre_steps()
                self.learner.load_model(model_save_dir)
                """

                metric_table['acc'][task_name] = OrderedDict()
                metric_table_local['acc'][task_name] = OrderedDict()
                # Global accuracy over the full output space
                for j in range(i+1):
                    val_name = self.task_names[j]
                    acc = self.learner.validation(
                        self.dataloader[j]['val'],
                        task_in=self.class_mask[j],
                        task_global=True,
                    )
                    metric_table['acc'][task_name][val_name] = acc
                    acc_matrix[j, i] = acc
                for j in range(i+1):
                    val_name = self.task_names[j]
                    acc_loc = self.learner.validation(self.dataloader[j]['val'], task_in=self.class_mask[j], task_global=False)
                    metric_table_local['acc'][task_name][val_name] = acc_loc

                self.evaluate_till_now(acc_matrix, i)

            avg_metrics['acc'], f_score = self.summarize_acc(avg_metrics['acc'], metric_table['acc'], metric_table_local['acc'])
            return avg_metrics, f_score

        """
        self.learner = learners.__dict__[self.learner_type].__dict__[self.learner_name](self.learner_config)

        # store results
        metric_table = {}
        metric_table_local = {}
        for mkey in self.metric_keys:
            metric_table[mkey] = {}
            metric_table_local[mkey] = {}
            
        for i in range(self.max_task):

            # increment task id in prompting modules
            if i > 0:
                try:
                    if self.learner.model.module.prompt is not None:
                        self.learner.model.module.prompt.process_task_count()
                except:
                    if self.learner.model.prompt is not None:
                        self.learner.model.prompt.process_task_count()

            # load model
            model_save_dir = self.model_top_dir + '/models/repeat-'+str(self.seed+1)+'/task-'+self.task_names[i]+'/'
            self.learner.task_count = i 
            self.learner.add_valid_output_dim(len(self.tasks_logits[i]))
            self.learner.pre_steps()
            self.learner.load_model(model_save_dir)

            # set task id for model (needed for prompting)
            try:
                self.learner.model.module.task_id = i
            except:
                self.learner.model.task_id = i

            # evaluate acc
            metric_table['acc'][self.task_names[i]] = OrderedDict()
            metric_table_local['acc'][self.task_names[i]] = OrderedDict()
            self.reset_cluster_labels = True
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table['acc'][val_name][self.task_names[i]] = self.task_eval(j)
            for j in range(i+1):
                val_name = self.task_names[j]
                metric_table_local['acc'][val_name][self.task_names[i]] = self.task_eval(j, local=True)

        # summarize metrics
        avg_metrics['acc'], f_score = self.summarize_acc(avg_metrics['acc'], metric_table['acc'],  metric_table_local['acc'])

        return avg_metrics, f_score
        """

    def cal_fscore(self, y):
        index = y.shape [1]
        fgt = 0
        for t in range(1, index):
            for i in range(t):
                fgt += (y[t-1,i]-y[t,i])*(1/t)

        fgt = fgt/(index-1)
        return fgt

    def evaluate_till_now(self, acc_matrix, task_id):
        print("DEBUG: evaluate_till_now 함수가 호출되었습니다.")
        print(f"DEBUG: acc_matrix 형태: {acc_matrix.shape}, task_id: {task_id}")
        A_list = [np.mean(acc_matrix[:k+1, k]) for k in range(task_id+1)]
        A_last = A_list[-1]
        A_avg = np.mean(A_list)
        result_str = f"[Average accuracy till task{task_id+1}] A_last: {A_last:.2f} A_avg: {A_avg:.2f}"
        if task_id > 0:
            forgetting = np.mean((np.max(acc_matrix, axis=1) - acc_matrix[:, task_id])[:task_id])
            result_str += f" Forgetting: {forgetting:.4f}"
        print(result_str)

def save_logits_statistics(id_logits, ood_logits, args, task_id):
    os.makedirs(os.path.join(args.log_dir, 'ood_statistics'), exist_ok=True)
    np.savez(os.path.join(args.log_dir, 'ood_statistics', f'logits_task{task_id}.npz'),
             id_logits=id_logits.cpu().numpy(),
             ood_logits=ood_logits.cpu().numpy())

def save_anomaly_histogram(id_scores, ood_scores, args, suffix, task_id):
    os.makedirs(os.path.join(args.log_dir, 'ood_histograms'), exist_ok=True)
    plt.figure()
    plt.hist(id_scores, bins=50, alpha=0.5, label='ID')
    plt.hist(ood_scores, bins=50, alpha=0.5, label='OOD')
    plt.legend()
    plt.title(f'{suffix.upper()} Histogram (Task {task_id})')
    plt.savefig(os.path.join(args.log_dir, 'ood_histograms', f'task{task_id}_{suffix}_hist.png'))
    plt.close()

    def evaluate_ood(self, model, id_datasets, ood_dataset, device, args, task_id=None):
        model.eval()

        ood_method = args.ood_method.upper()

        def MSP(logits):
            return F.softmax(logits, dim=1).max(dim=1)[0]

        def ENERGY(logits):
            return torch.logsumexp(logits, dim=1)

        def KL(logits):
            uniform = torch.ones_like(logits) / logits.shape[-1]
            return F.cross_entropy(logits, uniform, reduction='none')

        id_size = len(id_datasets)
        ood_size = len(ood_dataset)
        min_size = min(id_size, ood_size)
        if args.develop:
            min_size = 1000
        if args.verbose:
            print(f"ID dataset size: {id_size}, OOD dataset size: {ood_size}. Using {min_size} samples each for evaluation.")

        id_dataset_aligned = RandomSampleWrapper(id_datasets, min_size, args.seed) if id_size > min_size else id_datasets
        ood_dataset_aligned = RandomSampleWrapper(ood_dataset, min_size, args.seed) if ood_size > min_size else ood_dataset

        aligned_id_loader = DataLoader(id_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        aligned_ood_loader = DataLoader(ood_dataset_aligned, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        id_logits_list, ood_logits_list = [], []

        with torch.no_grad():
            for inputs, _ in aligned_id_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                id_logits_list.append(logits)

            for inputs, _ in aligned_ood_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                ood_logits_list.append(logits)

        id_logits = torch.cat(id_logits_list, dim=0)
        ood_logits = torch.cat(ood_logits_list, dim=0)

        if args.save:
            save_logits_statistics(id_logits, ood_logits, args, task_id if task_id is not None else 0)

        binary_labels = np.concatenate([np.ones(id_logits.shape[0]), np.zeros(ood_logits.shape[0])])

        methods = ["MSP", "ENERGY", "KL"] if ood_method == "ALL" else [ood_method]

        results = {}
        for method in methods:
            if method == "MSP":
                id_scores, ood_scores = MSP(id_logits), MSP(ood_logits)
            elif method == "ENERGY":
                id_scores, ood_scores = ENERGY(id_logits), ENERGY(ood_logits)
            else:
                id_scores, ood_scores = KL(id_logits), KL(ood_logits)

            if args.verbose:
                save_anomaly_histogram(id_scores.cpu().numpy(), ood_scores.cpu().numpy(), args, suffix=method.lower(), task_id=task_id)

            all_scores = torch.cat([id_scores, ood_scores], dim=0).cpu().numpy()

            fpr, tpr, _ = metrics.roc_curve(binary_labels, all_scores, drop_intermediate=False)
            auroc = metrics.auc(fpr, tpr)
            idx_tpr95 = np.abs(tpr - 0.95).argmin()
            fpr_at_tpr95 = fpr[idx_tpr95]

            print(f"[{method}]: evaluating metrics...")
            print(f"AUROC: {auroc * 100:.2f}%, FPR@TPR95: {fpr_at_tpr95 * 100:.2f}%")

            results[method] = {"auroc": auroc, "fpr_at_tpr95": fpr_at_tpr95, "scores": all_scores}

        return results