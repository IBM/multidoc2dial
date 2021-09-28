#!/bin/sh

domain=$1 # dmv ssa studentaid va
seg=$2 # token structure

dpr=dpr-$domain-$seg
rag_model_name=$CHECKPOINTS/rag-$dpr
ctx_model_name=$CHECKPOINTS/$dpr/ctx_encoder
KB_FOLDER=../data/mdd_kb/

# for train and validation split (with $domain as target domain)
python rag/use_own_knowledge_dataset.py \
--rag_model_name $rag_model_name \
--dpr_ctx_encoder_model_name  $ctx_model_name \
--csv_path $KB_FOLDER/mdd-$seg-wo-$domain.csv \
--output_dir $KB_FOLDER/knowledge_dataset-$dpr-wo

# for test split (with $domain as target domain)
python rag/use_own_knowledge_dataset.py \
--rag_model_name $rag_model_name \
--dpr_ctx_encoder_model_name  $ctx_model_name \
--csv_path $KB_FOLDER/mdd-$seg-$domain.csv \
--output_dir $KB_FOLDER/knowledge_dataset-$dpr

