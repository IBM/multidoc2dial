#!/bin/sh

export PYTHONPATH="../":"${PYTHONPATH}"
export TOKENIZERS_PARALLELISM=false
domain=$1 # all dmv ssa va studentaid
seg=$2 # token structure
score=$3 # original reranking reranking_original
task=$4 # grounding generation
seed=$RANDOM 

dpr=dpr-$domain-$seg
MODEL_NAME_OR_PATH=$CHECKPOINTS/rag-$dpr
KB_FOLDER=../data/mdd_kb/knowledge_dataset-$dpr
DATA_DIR=../data/mdd_$domain/dd-$task-$seg

python rag/finetune_rag_dialdoc.py \
    --seed $seed \
    --segmentation $seg \
    --do_marginalize 1 \
    --data_dir $DATA_DIR \
    --scoring_func $score \
    --output_dir $CHECKPOINTS/mdd-$task-$dpr-$score \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --model_type rag_token_dialdoc \
    --index_name dialdoc \
    --passages_path $KB_FOLDER/my_knowledge_dataset \
    --index_path $KB_FOLDER/my_knowledge_dataset_index.faiss \
    --fp16 \
    --profile \
    --do_train \
    --gpus 1 \
    --n_train -1 \
    --n_val -1 \
    --n_test -1 \
    --n_docs 5 \
    --train_batch_size 8 \
    --eval_batch_size 2 \
    --max_combined_length 300 \
    --max_source_length 128 \
    --max_target_length 50 \
    --val_max_target_length 50 \
    --test_max_target_length 50 \
    --label_smoothing 0.1 \
    --dropout 0.1 \
    --attention_dropout 0.1 \
    --weight_decay 0.001 \
    --adam_epsilon 1e-08 \
    --max_grad_norm 0.1 \
    --lr_scheduler polynomial \
    --learning_rate 3e-05 \
    --num_train_epochs 2 \
    --warmup_steps 500 \
    --gradient_accumulation_steps 1
