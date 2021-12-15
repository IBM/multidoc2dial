# MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents
This repository provides data and code for the corresponding [paper](https://arxiv.org/abs/2109.12595) "MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents" (EMNLP 2021) by Song Feng *, Siva Sankalp Patel*, Wan Hui and Sachindra Joshi.
Please cite the paper and star the repository if you find the paper, data and code useful for your work.

```bibtex
@inproceedings{feng2021multidoc2dial,
    title={MultiDoc2Dial: Modeling Dialogues Grounded in Multiple Documents},
    author={Feng, Song and Patel, Siva Sankalp and Wan, Hui and Joshi, Sachindra},
    booktitle={EMNLP},
    year={2021}
}
```

## Installation

Please refer to `conda_env.yml` for creating a virtual environment.

```bash
conda env create -f conda_env.yml
```

## Data

Please run the commands to download data. It will download the document and dialogue data into folder  `data/multidoc2dial`.

```bash
cd scripts
./run_download.sh
```

### Document preprocessing

To segment the document into passages, please refer to
> [`run_data_preprocessing.sh`](scripts/run_data_preprocessing.sh)

### Data preprocessing for fine-tuning DPR

If you are finetuning DPR on MultiDoc2Dial, please refer to [`run_data_preprocessing_dpr.sh`](scripts/run_data_preprocessing_dpr.sh) create positive and negative examples in the format of [DPR](https://github.com/facebookresearch/DPR).

## Run Baselines

### Finetuning DPR

To finetune DPR, we use Facebook [DPR](https://github.com/facebookresearch/DPR) (March 2021 release)  with an effective batch size 128. You can finetune DPR on MultiDoc2Dial data yourself ; or use our finetuned version.

*If you would like to finetune DPR yourself*, please refer to Facebook [DPR](https://github.com/facebookresearch/DPR) for detailed instructions.

Or

*If you would like to use our finetuned DPR encoders*, please use the the following paths as the model path to ctx or question encoder (for instance, [`run_converter_modelcard.sh`](scripts/run_converter_modelcard.sh)),
  - `sivasankalpp/dpr-multidoc2dial-token-question-encoder` for fine-tuned DPR question encoder based on token-segmented document passages ([link](https://huggingface.co/sivasankalpp/dpr-multidoc2dial-token-question-encoder))
  - `sivasankalpp/dpr-multidoc2dial-token-ctx-encoder` for fine-tuned DPR ctx encoder based on token-segmented document passages ([link](https://huggingface.co/sivasankalpp/dpr-multidoc2dial-token-ctx-encoder))
  - `sivasankalpp/dpr-multidoc2dial-structure-question-encoder` fine-tuned DPR question encoder based on structure-segmented document passages ([link](https://huggingface.co/sivasankalpp/dpr-multidoc2dial-structure-question-encoder))
  - `sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder` for fine-tuned DPR ctx encoder based on structure-segmented document passages ([link](https://huggingface.co/sivasankalpp/dpr-multidoc2dial-structure-ctx-encoder))

### Using finetuned DPR encoders in RAG

*If you obtain your own finetuned DPR checkpoints,*
1. Download the following files from RAG model cards to "../data" folder
  - <https://huggingface.co/facebook/rag-token-nq/resolve/main/question_encoder_tokenizer/tokenizer_config.json>
  - <https://huggingface.co/facebook/rag-token-nq/blob/main/question_encoder_tokenizer/vocab.txt>

2. Convert your fine-tuned DPR checkpoint and add it to RAG model. Please refer to [`run_converter.sh`](scripts/run_converter.sh).

OR

*If you use our finetuned DPR encoders*, please refer to [`run_converter_modelcard.sh`](scripts/run_converter_modelcard.sh).


### Finetuning RAG

Our implementation is based on [Huggingface RAG](https://huggingface.co/docs/transformers/master/model_doc/rag). Please refer to their [README](https://github.com/huggingface/transformers/tree/master/examples/research_projects/rag#readme) for more detailed explanations on document retrieval and finetuning RAG.

To create FAISS index, please refer to
> [`run_kb_index.sh`](scripts/run_kb_index.sh)

To finetune RAG on MultiDoc2Dial data, please refer to
> [`run_finetune_rag.sh`](scripts/run_finetune_rag.sh)

## Evaluations

To evaluate the retrieval results (recall@n for passage and document level), please refer to
> [`run_eval_rag_re.sh`](scripts/run_eval_rag_re.sh)

To evaluate the generation results, please refer to
> [`run_eval_rag_e2e.sh`](scripts/run_eval_rag_e2e.sh)

## Results

The evaluation results on the validation set of agent response generation task Please refer to the `scripts` for corresponding hyperparameters.

| Model       |F1    |    EM|  BLEU|  r@1 | r@5 |  r@10 |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- |
| D-token-nq  | 30.9 | 2.8 | 15.7 | 25.8 | 48.2 | 57.7  |
| D-struct-nq | 31.5 | 3.2 | 16.6 | 27.4 | 51.1 | 60.2  |
| D-token-ft  | 33.2 | 3.4 | 18.8 | 35.2 | 63.4 | 72.9  |
| D-struct-ft | 33.7 | 3.5 | 19.5 | 37.5 | 67.0 | 75.8  |

## Leaderboard

Please check out our [**leaderboard**](https://eval.ai/web/challenges/challenge-page/1437/overview) and [**Shared Task**](http://doc2dial.github.io/workshop2022/#shared).

## Acknowledgement

Our code is based on [Huggingface Transformers](https://github.com/huggingface/transformers). Our dataset is based on [Doc2Dial](https://arxiv.org/abs/2011.06623). We thank the authors for sharing their great work.