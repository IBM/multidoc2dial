#!/bin/sh

domain=all # all dmv ssa studentaid va
seg=$1 # token structure

dpr=dpr-$domain-$seg
rag_model_name=$CHECKPOINTS/rag-$dpr
ctx_model_name=$CHECKPOINTS/$dpr/ctx_encoder
KB_FOLDER=../data/mdd_kb/

python rag/use_own_knowledge_dataset.py \
--rag_model_name $rag_model_name \
--dpr_ctx_encoder_model_name  $ctx_model_name \
--csv_path $KB_FOLDER/mdd-$seg-$domain.csv \
--output_dir $KB_FOLDER/knowledge_dataset-$dpr

