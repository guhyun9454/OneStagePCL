# bash experiments/imagenet-r.sh
# experiment settings
DATASET=ImageNet_R
N_CLASS=200

# save directory
#OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/origin/${DATASET}/10-task
#OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/cls_share/${DATASET}/10-task
#OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/clsReluatt_share/${DATASET}/10-task
#OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/meanwocls_share/${DATASET}/10-task
#OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/outputs_swinquery_vit_l2_1e-3/${DATASET}/10-task
#OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/ori_clsdetach_l2_1_T10/${DATASET}/10-task
#OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/outputs_swinquery_vit_l2dist_5e-4/${DATASET}/10-task

#OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/ori_cls_naivecls_clipvit_distill_l2_0.001/${DATASET}/10-task



#OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/ori_cls_swinquery_vit_qrloss_l2sftmax_1e-4/${DATASET}/5-task
#CONFIG=configs/imnet-r_prompt_short.yaml
#REPEAT=5
#OVERWRITE=0
## process inputs
#mkdir -p $OUTDIR
## OS-P
#python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
#    --learner_type prompt --learner_name OSPrompt \
#    --prompt_param 100 8 1e-4  --query swin\
#    --log_dir ${OUTDIR}/os-p



OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/ori_cls_swinquery_vit_qrloss_l2sftmax_1e-4/${DATASET}/20-task
CONFIG=configs/imnet-r_prompt_long.yaml
REPEAT=5
OVERWRITE=0
# process inputs
mkdir -p $OUTDIR
# OS-P
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OSPrompt \
    --prompt_param 100 8 1e-4  --query swin\
    --log_dir ${OUTDIR}/os-p




OUTDIR=/vast/palmer/scratch/panda/yk569/OneStagePCL/ori_cls_swinquery_vit_qrloss_l2sftmax_1e-4/${DATASET}/10-task
CONFIG=configs/imnet-r_prompt.yaml
REPEAT=5
OVERWRITE=0
# process inputs
mkdir -p $OUTDIR
# OS-P
python -u run.py --config $CONFIG --gpuid $1 --repeat $REPEAT --overwrite $OVERWRITE \
    --learner_type prompt --learner_name OSPrompt \
    --prompt_param 100 8 1e-4  --query swin\
    --log_dir ${OUTDIR}/os-p