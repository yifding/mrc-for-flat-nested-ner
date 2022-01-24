#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-xp-002
#$-l gpu_card=4
#$-N CRC_train_mrc_ner

export PATH=/nfs/yding4/conda_envs/mrc_for_ner/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/mrc_for_ner/lib:$LD_LIBRARY_PATH


TIME=0118
FILE=att33_cased_base
REPO_PATH=/nfs/yding4/AVE_project/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

DATA_DIR=/nfs/yding4/AVE_project/consumable/title_bullet/33att/sample_data_1/mrc_for_ner/33att/
#DATA_DIR=/nfs/yding4/AVE_project/consumable/title_bullet/33att/sample_data_1/mrc_for_ner/33_test/

BERT_DIR=/nfs/yding4/AVE_project/mrc-for-flat-nested-ner/yd_data/bert-base-cased
OUTPUT_BASE=/nfs/yding4/AVE_project/mrc-for-flat-nested-ner/yd_script/01_18_2021/train_mrc_for_ner
# OUTPUT_BASE=/scratch365/yding4/AVE_project/mrc-for-flat-nested-ner/yd_data/experiments

BATCH=10
GRAD_ACC=4
BERT_DROPOUT=0.1
MRC_DROPOUT=0.3
LR=3e-5
LR_MINI=3e-7
LR_SCHEDULER=polydecay
SPAN_WEIGHT=0.1
WARMUP=0
MAX_LEN=200
MAX_NORM=1.0
MAX_EPOCH=20
INTER_HIDDEN=2048
WEIGHT_DECAY=0.01
OPTIM=torch.adam
VAL_CHECK=0.2
PREC=16
SPAN_CAND=pred_and_gold


# OUTPUT_DIR=${OUTPUT_BASE}/mrc_ner/${TIME}/${FILE}_cased_large_lr${LR}_drop${MRC_DROPOUT}_norm${MAXNORM}_weight${SPAN_WEIGHT}_warmup${WARMUP}_maxlen${MAXLEN}
OUTPUT_DIR=${OUTPUT_BASE}/mrc_ner
mkdir -p ${OUTPUT_DIR}


CUDA_VISIBLE_DEVICES=0,1
WORLD_SIZE=2
/nfs/yding4/conda_envs/mrc_for_ner/bin/python ${REPO_PATH}/train/mrc_ner_trainer.py \
--data_dir ${DATA_DIR} \
--bert_config_dir ${BERT_DIR} \
--max_length ${MAX_LEN} \
--batch_size ${BATCH} \
--gpus="${WORLD_SIZE}" \
--precision=${PREC} \
--progress_bar_refresh_rate 1 \
--lr ${LR} \
--val_check_interval ${VAL_CHECK} \
--accumulate_grad_batches ${GRAD_ACC} \
--default_root_dir ${OUTPUT_DIR} \
--mrc_dropout ${MRC_DROPOUT} \
--bert_dropout ${BERT_DROPOUT} \
--max_epochs ${MAX_EPOCH} \
--span_loss_candidates ${SPAN_CAND} \
--weight_span ${SPAN_WEIGHT} \
--warmup_steps ${WARMUP} \
--max_length ${MAX_LEN} \
--gradient_clip_val ${MAX_NORM} \
--weight_decay ${WEIGHT_DECAY} \
--optimizer ${OPTIM} \
--lr_scheduler ${LR_SCHEDULER} \
--classifier_intermediate_hidden_size ${INTER_HIDDEN} \
--flat \
--lr_mini ${LR_MINI}    \
--distributed_backend=ddp