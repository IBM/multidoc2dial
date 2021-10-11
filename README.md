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

Please run the commands to download data. It will download the document and dialogue data in folder  `data/multidoc2dial` and `data/multidoc2dial_domain` for domain adaptation setup.

```bash
cd scripts
./run_download.sh
```

### Document preprocessing

To split the document to passages and create FAISS index, please refer to

[`run_data_preprocessing.sh`](scripts/run_data_preprocessing.sh)

For domain adaptation set up with a target domain, e.g.,  `domain=ssa`, please refer to

[`run_data_preprocessing_domain.sh`](scripts/run_data_preprocessing_domain.sh)

### Data preprocessing for fine-tuning DPR

To create positive and negative examples in the format of [DPR](https://github.com/facebookresearch/DPR) , please refer to

[`run_data_preprocessing_dpr.sh`](scripts/run_data_preprocessing_dpr.sh)

For domain adaptation setting, please refer to

[`run_data_preprocessing_dpr_domain.sh`](scripts/run_data_preprocessing_dpr_domain.sh)

## Baselines

### Finetuning DPR

To finetune DPR, we use Facebook [DPR](https://github.com/facebookresearch/DPR) with an effective batch size 128.

Convert DPR checkpoint with the [converter](https://github.com/huggingface/transformers/blob/master/src/transformers/models/dpr/convert_dpr_original_checkpoint_to_pytorch.py) , please refer to

Download the following files from RAG model cards to "../data" folder

- <https://huggingface.co/facebook/rag-token-nq/resolve/main/question_encoder_tokenizer/tokenizer_config.json>
- <https://huggingface.co/facebook/rag-token-nq/blob/main/question_encoder_tokenizer/vocab.txt>

[`run_converter.sh`](scripts/run_converter.sh)

### Finetuning RAG

To finetune RAG on MultiDoc2Dial data, please refer to

[`run_finetune_rag.sh`](scripts/run_finetune_rag.sh)

## Evaluations

To evaluate the retrieval results, please refer to

[`run_eval_rag_re.sh`](scripts/run_eval_rag_re.sh)

To evaluate the generation results, please refer to

[`run_eval_rag_e2e.sh`](scripts/run_eval_rag_e2e.sh)

## Results

The evaluation results on the validation set of agent response generation task Please refer to the `scripts` for corresponding hyperparameters.


| Model       |F1    |    EM|  BLEU|  r@1 | r@5 |  r@10 |
| ----------- | ---- | ---- | ---- | ---- | ---- | ---- |
| D-token-nq  | 30.9 | 2.8 | 15.7 | 25.8 | 48.2 | 57.7  |
| D-struct-nq | 31.5 | 3.2 | 16.6 | 27.4 | 51.1 | 60.2  |
| D-token-ft  | 33.2 | 3.4 | 18.8 | 35.2 | 63.4 | 72.9  |
| D-struct-ft | 33.7 | 3.5 | 19.5 | 37.5 | 67.0 | 75.8  |
