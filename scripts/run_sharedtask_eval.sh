python sharedtask_eval.py \
--task grounding \
--prediction_json ../sharedtask/sample_grounding_dev_predictions.json \
--reference_json ../sharedtask/sample_grounding_dev_references.json
    

python sharedtask_eval.py \
  --task utterance \
  --prediction_json ../sharedtask/sample_utterance_dev_predictions.json \
  --reference_json ../sharedtask/sample_utterance_dev_references.json