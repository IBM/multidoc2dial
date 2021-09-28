#!/bin/sh

seg=$1 # token or structure
task=$2 # grounding or generation
YOUR_DIR=../data # change it to your own local dir


python data_preprocessor.py \
--dataset_config_name multidoc2dial \
--output_dir $YOUR_DIR/mdd_all \
--kb_dir $YOUR_DIR/mdd_kb \
--segmentation $seg \
--task $task 