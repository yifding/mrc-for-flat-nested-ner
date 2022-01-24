#!/usr/bin/env bash
# -*- coding: utf-8 -*-

#$-m abe
#$-M dyfdyf0125@gmail.com
#$-q gpu@qa-xp-002
#$-l gpu_card=4
#$-N CRC_train_mrc_ner


export PATH=/nfs/yding4/conda_envs/mrc_for_ner/bin:$PATH
export LD_LIBRARY_PATH=/nfs/yding4/conda_envs/mrc_for_ner/lib:$LD_LIBRARY_PATH

REPO_PATH=/nfs/yding4/AVE_project/mrc-for-flat-nested-ner
export PYTHONPATH="$PYTHONPATH:$REPO_PATH"

/nfs/yding4/conda_envs/mrc_for_ner/bin/python ${REPO_PATH}/datasets/mrc_ner_dataset.py