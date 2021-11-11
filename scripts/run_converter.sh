#!/bin/sh

domain=$1
seg=$2

config=dpr-$domain-$seg

dpr=dpr_mdd-$domain-$seg
src=YOUR_DPR_CHECKPOINT

mkdir $CHECKPOINTS/$config

python convert_dpr_original_checkpoint_to_pytorch.py \
--type question_encoder \
--src $src \
--dest $CHECKPOINTS/dpr-$domain-$seg/question_encoder

python convert_dpr_original_checkpoint_to_pytorch.py \
--type ctx_encoder \
--src $src \
--dest $CHECKPOINTS/dpr-$domain-$seg/ctx_encoder


# generate rag model 
cp ../data/tokenizer_config.json $CHECKPOINTS/$config/question_encoder/
cp ../data/vocab.txt $CHECKPOINTS/$config/question_encoder/
cp ../data/tokenizer_config.json $CHECKPOINTS/$config/ctx_encoder/
cp ../data/vocab.txt $CHECKPOINTS/$config/ctx_encoder/

# config "model_path" for question encoder to your local path to DPR encoder;
# or use our uploaded model, such as "sivasankalpp/dpr-multidoc2dial-token-question-encoder" or "sivasankalpp/dpr-multidoc2dial-structure-question-encoder"
python model_converter.py \
--model_path $CHECKPOINTS/$config/question_encoder \
--out_path $CHECKPOINTS/rag-$config