#!/bin/sh

domain=$1 # "all" or "dmv", "ssa", "studentaid", "va" for domain adaptation setup
seg=$2 # token or structure

dpr=dpr-$domain-$seg
rag_model_name=$CHECKPOINTS/rag-$dpr
# config "ctx_model_name" for ctx encoder to your local path to DPR encoder;
# ctx_model_name=$CHECKPOINTS/$dpr/ctx_encoder
# or use our fine-tuned DPR encoders, such as "sivasankalpp/dpr-multidoc2dial-token-ctx-encoder" or "sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder"
ctx_model_name=sivasankalpp/dpr-multidoc2dial-$seg-ctx-encoder
KB_FOLDER=../data/mdd_kb/

python rag/use_own_knowledge_dataset.py \
--rag_model_name $rag_model_name \
--dpr_ctx_encoder_model_name  $ctx_model_name \
--csv_path $KB_FOLDER/mdd-$seg-$domain.csv \
--output_dir $KB_FOLDER/knowledge_dataset-$dpr

