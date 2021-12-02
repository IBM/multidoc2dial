#!/bin/sh

domain=$1
seg=$2
config=dpr-$domain-$seg

# config "model_path" for question encoder to your local path to DPR encoder;
# or use our uploaded model, such as "sivasankalpp/dpr-multidoc2dial-token-question-encoder" or "sivasankalpp/dpr-multidoc2dial-structure-question-encoder"
python model_converter.py \
--model_path sivasankalpp/dpr-multidoc2dial-$seg-question-encoder \
--out_path $CHECKPOINTS/rag-$config