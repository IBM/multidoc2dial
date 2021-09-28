#!/bin/sh

mkdir ../data && \
cd ../data && \
wget http://doc2dial.github.io/multidoc2dial/file/multidoc2dial.zip && \
wget http://doc2dial.github.io/multidoc2dial/file/multidoc2dial_domain.zip && \
unzip multidoc2dial.zip && \
unzip multidoc2dial_domain.zip && \
rm *.zip && \
wget https://huggingface.co/facebook/rag-token-nq/raw/main/question_encoder_tokenizer/tokenizer_config.json && \
wget https://huggingface.co/facebook/rag-token-nq/raw/main/question_encoder_tokenizer/vocab.txt