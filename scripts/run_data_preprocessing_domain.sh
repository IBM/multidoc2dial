#!/bin/sh

domain=$1 # dmv va ssa or studentaid
seg=$2 # token or structure
task=$3 # grounding or generation
YOUR_DIR=../data # change it to your own local dir

python data_preprocessor.py \
--dataset_config_name multidoc2dial_$domain \
--output_dir $YOUR_DIR/mdd_$domain \
--target_domain $domain \
--kb_dir $YOUR_DIR/mdd_kb \
--segmentation $seg \
--task $task