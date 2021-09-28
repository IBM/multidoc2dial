#!/bin/sh

export PYTHONPATH="../":"${PYTHONPATH}"
domain=$1 # all dmv va ssa studentaid
seg=$2  # token structure
score=$3 # original reranking reranking_original
task=$4 # grounding generation
split=$5 # val test

dpr=dpr-$domain-$seg
DATA_DIR=../data/mdd_$domain/dd-$task-$seg
KB_FOLDER=../data/mdd_kb/knowledge_dataset-$dpr
MODEL_PATH=$CHECKPOINTS/mdd-$task-$dpr-$score/


python rag/eval_rag.py \
--model_type rag_token_dialdoc \
--scoring_func $score \
--gold_pid_path $DATA_DIR/$split.pids \
--passages_path $KB_FOLDER/my_knowledge_dataset \
--index_path $KB_FOLDER/my_knowledge_dataset_index.faiss \
--index_name dialdoc \
--n_docs 10 \
--model_name_or_path $MODEL_PATH \
--eval_mode e2e \
--evaluation_set $DATA_DIR/$split.source \
--gold_data_path $DATA_DIR/$split.target \
--gold_data_mode ans \
--recalculate \
--eval_all_checkpoints \
--predictions_path results.txt