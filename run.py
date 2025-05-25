from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import sys
import argparse
import torch
import numpy as np
import yaml
import json
import random
from trainer import Trainer

def create_args():
    
    # This function prepares the variables shared across demo.py
    parser = argparse.ArgumentParser()

    # Standard Args
    parser.add_argument('--gpuid', nargs="+", type=int, default=None,
                         help="The list o f gpuid, ex:--gpuid 3 1. Negative value means cpu-only. 기본값 None은 모든 사용 가능한 GPU를 사용함을 의미합니다.")
    parser.add_argument('--log_dir', type=str, default="outputs/out",
                         help="Save experiments results in dir for future plotting!")
    parser.add_argument('--learner_type', type=str, default='prompt', help="The type (filename) of learner")
    parser.add_argument('--learner_name', type=str, default='CODAPrompt', help="The class name of learner")
    parser.add_argument('--query', type=str, default='vit', help="choose one of [poolformer]")
    parser.add_argument('--debug_mode', type=int, default=0, metavar='N',
                        help="activate learner specific settings for debug_mode")
    parser.add_argument('--seed', type=int, default=0, help="Random seed for reproducibility")
    parser.add_argument('--overwrite', type=int, default=0, metavar='N', help='Train regardless of whether saved model exists')

    # CL Args          
    parser.add_argument('--oracle_flag', default=False, action='store_true', help='Upper bound for oracle')
    parser.add_argument('--upper_bound_flag', default=False, action='store_true', help='Upper bound')
    parser.add_argument('--memory', type=int, default=0, help="size of memory for replay")
    parser.add_argument('--temp', type=float, default=2., dest='temp', help="temperature for distillation")
    parser.add_argument('--DW', default=False, action='store_true', help='dataset balancing')
    parser.add_argument('--prompt_param', nargs="+", type=float, default=[1, 1, 1],
                         help="e prompt pool size, e prompt length, g prompt length")

    # IL mode for continual learning scenario
    parser.add_argument('--IL_mode', type=str, default='cil',
                        help="Incremental learning mode: one of ['cil','dil','vil','joint']")
    # Config Arg
    parser.add_argument('--config', type=str, default="configs/config.yaml",
                         help="yaml experiment config input")
    parser.add_argument('--develop', default=False, action='store_true', help='develop mode')
    parser.add_argument('--ood_dataset', type=str, default=None, help='Name of OOD dataset to evaluate')
    parser.add_argument('--ood_method', type=str, default='ALL', help='OOD detection method: MSP, ENERGY, KL, ALL')
    parser.add_argument('--save', action='store_true', help='Save OOD evaluation statistics')
    parser.add_argument('--verbose', action='store_true', help='Verbose OOD evaluation logs')
    return parser

def get_args(argv):
    parser=create_args()
    args = parser.parse_args(argv)
    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    config.update(vars(args))
    return argparse.Namespace(**config)

# want to save everything printed to outfile
class Logger(object):
    def __init__(self, name):
        self.terminal = sys.stdout
        self.log = open(name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.log.flush()

if __name__ == '__main__':
    args = get_args(sys.argv[1:])

    # 사용 가능한 모든 GPU 감지 및 설정
    if args.gpuid is None:
        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 0:
                args.gpuid = list(range(num_gpus))
                print(f"자동으로 감지된 GPU {num_gpus}개를 모두 사용합니다: {args.gpuid}")
            else:
                args.gpuid = [-1]  # CPU 모드
                print("사용 가능한 GPU가 없어 CPU 모드로 실행합니다.")
        else:
            args.gpuid = [-1]  # CPU 모드
            print("CUDA를 사용할 수 없어 CPU 모드로 실행합니다.")

    # determinstic backend
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic=True

    # duplicate output stream to output file
    if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
    log_out = args.log_dir + '/output.log'
    sys.stdout = Logger(log_out)

    # save args
    with open(args.log_dir + '/args.yaml', 'w') as yaml_file:
        yaml.dump(vars(args), yaml_file, default_flow_style=False)
    
    metric_keys = ['acc','time',]
    save_keys = ['global', 'pt', 'pt-local']
    global_only = ['time']
    avg_metrics = {}
    for mkey in metric_keys: 
        avg_metrics[mkey] = {}
        for skey in save_keys: avg_metrics[mkey][skey] = []

    # 한 번만 실행되도록 수정
    print('************************************')
    print('* STARTING EXPERIMENT')
    print('************************************')

    # 사용자가 설정한 seed 사용
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # set up a trainer
    trainer = Trainer(args, seed, metric_keys, save_keys)

    # init total run metrics storage
    max_task = args.num_tasks if args.max_task <= 0 else args.max_task
    for mkey in metric_keys: 
        avg_metrics[mkey]['global'] = np.zeros((max_task, 1))
        if (not (mkey in global_only)):
            avg_metrics[mkey]['pt'] = np.zeros((max_task, max_task, 1))
            avg_metrics[mkey]['pt-local'] = np.zeros((max_task, max_task, 1))

    # train model
    avg_metrics = trainer.train(avg_metrics)  

    # evaluate model
    avg_metrics, f_score = trainer.evaluate(avg_metrics)

    # 결과 출력
    print('===실험 결과 요약===')
    for mkey in metric_keys: 
        print(mkey, ' | value:', avg_metrics[mkey]['global'][-1, 0])
    print ('F-score:', f_score)
    


