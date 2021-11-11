""" Evaluation script for RAG models."""

import argparse
import ast
import logging
import os
import sys

import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from datasets import load_metric

from transformers import BartForConditionalGeneration, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration
from transformers import logging as transformers_logging


sys.path.append(os.path.join(os.getcwd()))  # noqa: E402 # isort:skip
from utils_rag import exact_match_score, f1_score, load_bm25  # noqa: E402 # isort:skip
from dialdoc.models.rag.modeling_rag_dialdoc import DialDocRagTokenForGeneration
from dialdoc.models.rag.retrieval_rag_dialdoc import DialDocRagRetriever


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

transformers_logging.set_verbosity_info()

# os.environ['KMP_DUPLICATE_LIB_OK']='True'


def get_top_n_indices(bm25, query, n=5):
    query = query.lower().split()
    scores = bm25.get_scores(query)
    scores_i = [(i, score) for i, score in enumerate(scores)]
    sorted_indices = sorted(scores_i, key=lambda score: score[1], reverse=True)
    return sorted_indices[:n]


def infer_model_type(model_name_or_path):
    if "token_dialdoc" in model_name_or_path:
        return "rag_token_dialdoc"
    if "token" in model_name_or_path:
        return "rag_token"
    if "sequence" in model_name_or_path:
        return "rag_sequence"
    if "bart" in model_name_or_path:
        return "bart"
    return None


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    return max(metric_fn(prediction, gt) for gt in ground_truths)


def get_scores(args, preds_path, gold_data_path):
    hypos = [line.strip() for line in open(preds_path, "r").readlines()]
    answers = []

    if args.gold_data_mode == "qa":
        data = pd.read_csv(gold_data_path, sep="\t", header=None)
        for answer_list in data[1]:
            ground_truths = ast.literal_eval(answer_list)
            answers.append(ground_truths)
    else:
        references = [line.strip() for line in open(gold_data_path, "r").readlines()]
        answers = [[reference] for reference in references]

    f1 = em = total = 0
    for prediction, ground_truths in zip(hypos, answers):
        total += 1
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)

    em = 100.0 * em / total
    f1 = 100.0 * f1 / total

    metric = load_metric("sacrebleu")
    metric.add_batch(predictions=hypos, references=answers)
    sacrebleu = metric.compute()["score"]

    logger.info(f"F1: {f1: .2f}")
    logger.info(f"EM: {em: .2f}")
    logger.info(f"sacrebleu: {sacrebleu: .2f}")
    logger.info(f"all: {f1: .2f} & {em: .2f} & {sacrebleu: .2f} ")


def get_precision_at_k(args, preds_path, gold_data_path):
    k = args.k
    hypos = [line.strip().split("####")[0] for line in open(preds_path, "r").readlines()]
    hypos_pid = [line.strip().split("####")[-1] for line in open(preds_path, "r").readlines()]
    references = [line.strip() for line in open(gold_data_path, "r").readlines()]
    pids = [line.strip().split("\t") for line in open(args.gold_pid_path, "r").readlines()]

    r_1 = r_5 = r_10 = em = total = 0
    for hypo, reference in zip(hypos, references):
        hypo_provenance = set(hypo.split("\t")[:k])
        ref_provenance = set(reference.split("\t"))
        total += 1
        em += len(hypo_provenance & ref_provenance) / k
        r_1 += int(bool(set(hypo.split("\t")[:1]) & ref_provenance))
        r_5 += int(bool(set(hypo.split("\t")[:5]) & ref_provenance))
        r_10 += int(bool(set(hypo.split("\t")[:10]) & ref_provenance))
    r_1 = 100.0 * r_1 / total
    r_5 = 100.0 * r_5 / total
    r_10 = 100.0 * r_10 / total
    # logger.info(f"Doc_Prec@{k}: {em: .2f}")
    # logger.info(f"Doc_Prec@{1}: {r_1: .2f}")
    logger.info(f"Doc_Prec@1: {r_1: .2f}")
    logger.info(f"Doc_Prec@5: {r_5: .2f}")
    logger.info(f"Doc_Prec@10: {r_10: .2f}")

    r_1_p = r_5_p = r_10_p = total = 0
    for hypo, reference in zip(hypos_pid, pids):
        hypo = hypo.split("\t")
        # hypo_provenance = set(hypo)
        ref_provenance = set(reference)
        total += 1
        # em += len([r for r in reference if r in hypo_provenance]) == len(reference)
        r_1_p += int(bool(set(hypo[:1]) & ref_provenance))
        r_5_p += int(bool(set(hypo[:5]) & ref_provenance))
        r_10_p += int(bool(set(hypo[:10]) & ref_provenance))
    r_1_p = 100.0 * r_1_p / total
    r_5_p = 100.0 * r_5_p / total
    r_10_p = 100.0 * r_10_p / total
    logger.info(f"Pid_Prec@1: {r_1_p: .2f}")
    logger.info(f"Pid_Prec@5: {r_5_p: .2f}")
    logger.info(f"Pid_Prec@10: {r_10_p: .2f}")
    logger.info(f"all: {r_1: .2f} & {r_5: .2f} & {r_10: .2f}  & {r_1_p: .2f} & {r_5_p: .2f} & {r_10_p: .2f} & ")


def mean_pool(vector: torch.LongTensor):
    return vector.sum(axis=0) / vector.shape[0]


def get_attn_mask(tokens_tensor: torch.LongTensor) -> torch.tensor:
    return tokens_tensor != 0


def evaluate_batch_retrieval(args, rag_model, questions, domains=None):  # old_q
    def strip_title(title):
        if title.startswith('"'):
            title = title[1:]
        if title.endswith('"'):
            title = title[:-1]
        return title

    # retriever_input_ids_0 = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
    #     old_q,
    #     return_tensors="pt",
    #     padding=True,
    #     truncation=True,
    # )["input_ids"].to(args.device)
    # question_enc_outputs = rag_model.rag.question_encoder(retriever_input_ids_0)
    # question_enc_pool_output = question_enc_outputs[0]

    inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
        questions,
        return_tensors="pt",
        padding=True,
        truncation=True,
        add_special_tokens=True,
        return_token_type_ids=True,
    )

    retriever_input_ids = inputs_dict.input_ids.to(args.device)
    token_type_ids = inputs_dict.token_type_ids.to(args.device)
    attention_mask = inputs_dict.attention_mask.to(args.device)

    dpr_out = rag_model.rag.question_encoder(
        retriever_input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
    )
    combined_out = dpr_out.pooler_output

    ## Get mask for current turn input ids
    curr_turn_mask = torch.logical_xor(attention_mask, token_type_ids)
    current_turn_input_ids = retriever_input_ids * curr_turn_mask
    current_turn_only_out = rag_model.rag.question_encoder(
        current_turn_input_ids, attention_mask=curr_turn_mask.long(), return_dict=True
    )
    current_turn_output = current_turn_only_out.pooler_output

    ## Split the dpr sequence output
    sequence_output = dpr_out.hidden_states[-1]
    attn_mask = get_attn_mask(retriever_input_ids)
    ## Split sequence output, and pool each sequence
    seq_out_0 = []  # last turn, if query; doc structure if passage
    seq_out_1 = []  # dial history, if query; passage text if passage
    dialog_lengths = []
    for i in range(sequence_output.shape[0]):
        seq_out_masked = sequence_output[i, attn_mask[i], :]
        segment_masked = token_type_ids[i, attn_mask[i]]
        seq_out_masked_0 = seq_out_masked[segment_masked == 0, :]
        seq_out_masked_1 = seq_out_masked[segment_masked == 1, :]
        dialog_lengths.append((len(seq_out_masked_0), len(seq_out_masked_1)))
        ### perform pooling
        seq_out_0.append(mean_pool(seq_out_masked_0))
        seq_out_1.append(mean_pool(seq_out_masked_1))

    pooled_output_0 = torch.cat([seq.view(1, -1) for seq in seq_out_0], dim=0)
    pooled_output_1 = torch.cat([seq.view(1, -1) for seq in seq_out_1], dim=0)

    if args.scoring_func in ["reranking_original", "current_original"]:
        current_out = current_turn_output
    else:
        current_out = pooled_output_0

    if args.bm25:
        logger.info("Using BM25 for retrieval")
        doc_ids = []
        doc_scores = []
        for input_string in questions:
            input_string = " [SEP] ".join(input_string)
            sorted_indices = get_top_n_indices(rag_model.bm25, input_string, rag_model.config.n_docs)
            doc_ids.append([x[0] for x in sorted_indices])
            doc_scores.append([x[-1] for x in sorted_indices])
        all_docs = rag_model.retriever.index.get_doc_dicts(np.array(doc_ids))
    else:
        if args.scoring_func != "original":
            current_input = current_out.cpu().detach().to(torch.float32).numpy()
            history_input = pooled_output_1.cpu().detach().to(torch.float32).numpy()
        else:
            current_input = combined_out.cpu().detach().to(torch.float32).numpy()
            history_input = combined_out.cpu().detach().to(torch.float32).numpy()
        result = rag_model.retriever(
            retriever_input_ids,
            combined_out.cpu().detach().to(torch.float32).numpy(),
            current_input,
            history_input,
            dialog_lengths=dialog_lengths,
            domain=domains,
            prefix=rag_model.rag.generator.config.prefix,
            n_docs=rag_model.config.n_docs,
            return_tensors="pt",
        )
        all_docs = rag_model.retriever.index.get_doc_dicts(result.doc_ids)
        doc_ids = result.doc_ids
    provenance_strings = []

    for i, docs in enumerate(all_docs):
        provenance = [strip_title(title) for title in docs["title"]]
        # provenance_strings.append("\t".join(provenance))
        pids = "\t".join([str(int(e)) for e in doc_ids[i]])
        provenance_strings.append("\t".join(provenance) + "####" + pids)
    return provenance_strings


def evaluate_batch_e2e(args, rag_model, questions, domains=None):
    with torch.no_grad():
        inputs_dict = rag_model.retriever.question_encoder_tokenizer.batch_encode_plus(
            questions,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True,
            return_token_type_ids=True,
        )

        input_ids = inputs_dict.input_ids.to(args.device)
        token_type_ids = inputs_dict.token_type_ids.to(args.device)
        attention_mask = inputs_dict.attention_mask.to(args.device)
        outputs = rag_model.generate(  # rag_model overwrites generate
            input_ids,
            domain=domains,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            num_beams=args.num_beams,
            min_length=args.min_length,
            max_length=args.max_length,
            early_stopping=False,
            num_return_sequences=1,
            bad_words_ids=[[0, 0]],  # BART likes to repeat BOS tokens, dont allow it to generate more than one
        )
        answers = rag_model.retriever.generator_tokenizer.batch_decode(outputs, skip_special_tokens=True)

        if args.print_predictions:
            for q, a in zip(questions, answers):
                logger.info("Q: {} - A: {}".format(q, a))

        return answers


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scoring_func",
        default="original",
        type=str,
        help="different scoring function, `original`, `linear`, `nonlinear`, `reranking`, `reranking_original`",
    )
    parser.add_argument(
        "--bm25",
        type=str,
        default=None,
        help="file folder",
    )
    parser.add_argument(
        "--mapping_file",
        type=str,
        default=None,
        help="file folder",
    )
    parser.add_argument(
        "--model_type",
        choices=["rag_sequence", "rag_token", "rag_token_dialdoc", "bart"],
        type=str,
        help="RAG model type: rag_sequence, rag_token or bart, if none specified, the type is inferred from the model_name_or_path",
    )
    parser.add_argument(
        "--index_name",
        default=None,
        choices=["dialdoc", "custom", "exact", "compressed", "legacy"],
        type=str,
        help="RAG model retriever type",
    )
    parser.add_argument(
        "--index_path",
        default=None,
        type=str,
        help="Path to the retrieval index",
    )
    parser.add_argument(
        "--passages_path",
        default=None,
        type=str,
        help="Path to the knowledge data",
    )
    parser.add_argument("--n_docs", default=5, type=int, help="Number of retrieved docs")
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        required=True,
        help="Path to pretrained checkpoints or model identifier from huggingface.co/models",
    )
    parser.add_argument(
        "--eval_mode",
        choices=["e2e", "retrieval"],
        default="retrieval",
        type=str,
        help="Evaluation mode, e2e calculates exact match and F1 of the downstream task, retrieval calculates precision@k.",
    )
    parser.add_argument("--k", default=1, type=int, help="k for the precision@k calculation")
    parser.add_argument(
        "--evaluation_set",
        default=None,
        type=str,
        required=True,
        help="Path to a file containing evaluation samples",
    )
    parser.add_argument(
        "--gold_data_path",
        default=None,
        type=str,
        required=True,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument(
        "--gold_domain_path",
        default=None,
        type=str,
        required=False,
        help="Path to a tab-separated file with gold domains",
    )
    parser.add_argument(
        "--gold_pid_path",
        default=None,
        type=str,
        required=True,
        help="Path to a tab-separated file with gold samples",
    )
    parser.add_argument(
        "--gold_data_mode",
        default="qa",
        type=str,
        choices=["qa", "ans"],
        help="Format of the gold data file"
        "qa - a single line in the following format: question [tab] answer_list"
        "ans - a single line of the gold file contains the expected answer string",
    )
    parser.add_argument(
        "--predictions_path",
        type=str,
        default="predictions.txt",
        help="Name of the predictions file, to be stored in the checkpoints directory",
    )
    parser.add_argument(
        "--eval_all_checkpoints",
        action="store_true",
        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number",
    )
    parser.add_argument(
        "--eval_batch_size",
        default=8,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--recalculate",
        help="Recalculate predictions even if the prediction file exists",
        action="store_true",
    )
    parser.add_argument(
        "--num_beams",
        default=4,
        type=int,
        help="Number of beams to be used when generating answers",
    )
    parser.add_argument("--min_length", default=1, type=int, help="Min length of the generated answers")
    parser.add_argument("--max_length", default=50, type=int, help="Max length of the generated answers")

    parser.add_argument(
        "--print_predictions",
        action="store_true",
        help="If True, prints predictions while evaluating.",
    )
    parser.add_argument(
        "--print_docs",
        action="store_true",
        help="If True, prints docs retried while generating.",
    )
    args = parser.parse_args()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return args


def main(args):
    model_kwargs = {}
    if args.model_type is None:
        args.model_type = infer_model_type(args.model_name_or_path)
        assert args.model_type is not None
    if args.model_type.startswith("rag"):
        if args.model_type == "rag_token":
            model_class = RagTokenForGeneration
        elif args.model_type == "rag_token_dialdoc":
            model_class = DialDocRagTokenForGeneration
        else:
            model_class = RagSequenceForGeneration
        model_kwargs["n_docs"] = args.n_docs
        if args.index_name is not None:
            model_kwargs["index_name"] = args.index_name
        if args.index_path is not None:
            model_kwargs["index_path"] = args.index_path
        if args.passages_path is not None:
            model_kwargs["passages_path"] = args.passages_path
        if args.mapping_file is not None:
            model_kwargs["mapping_file"] = args.mapping_file
    else:
        model_class = BartForConditionalGeneration

    bm25 = None
    if args.bm25:
        bm25 = load_bm25(args.bm25)

    checkpoints = (
        [f.path for f in os.scandir(args.model_name_or_path) if f.is_dir()]
        if args.eval_all_checkpoints
        else [args.model_name_or_path]
    )

    logger.info("Evaluate the following checkpoints: %s", checkpoints)

    score_fn = get_scores if args.eval_mode == "e2e" else get_precision_at_k
    evaluate_batch_fn = evaluate_batch_e2e if args.eval_mode == "e2e" else evaluate_batch_retrieval

    for checkpoint in checkpoints:
        if os.path.exists(args.predictions_path) and (not args.recalculate):
            logger.info("Calculating metrics based on an existing predictions file: {}".format(args.predictions_path))
            score_fn(args, args.predictions_path, args.gold_data_path)
            continue

        logger.info("***** Running evaluation for {} *****".format(checkpoint))
        logger.info("  Batch size = %d", args.eval_batch_size)
        logger.info("  Predictions will be stored under {}".format(args.predictions_path))
        logger.info("  Using scoring function {}".format(args.scoring_func))

        if args.model_type.startswith("rag"):
            if "dialdoc" in args.model_type:
                retriever = DialDocRagRetriever.from_pretrained(checkpoint, **model_kwargs)
                retriever.config.scoring_func = args.scoring_func
                retriever.config.n_docs = args.n_docs
                retriever.config.bm25 = args.bm25
                retriever.config.mapping_file = args.mapping_file
                model = model_class.from_pretrained(checkpoint, retriever=retriever, **model_kwargs)
                if bm25:
                    model.bm25 = bm25
                model.config.scoring_func = args.scoring_func
                model.config.n_docs = args.n_docs
                model.config.bm25 = args.bm25
                model.config.mapping_file = args.mapping_file

            else:
                retriever = RagRetriever.from_pretrained(checkpoint, **model_kwargs)
                model = model_class.from_pretrained(checkpoint, retriever=retriever, **model_kwargs)
            model.retriever.init_retrieval()
        else:
            model = model_class.from_pretrained(checkpoint, **model_kwargs)
        model.to(args.device)

        with open(args.evaluation_set, "r") as eval_file, open(args.predictions_path, "w") as preds_file:
            questions = []
            if args.gold_domain_path:
                dom_file = open(args.gold_domain_path, "r")
                domains = []
                for line1, line2 in tqdm(zip(eval_file, dom_file)):
                    question = line1.strip()
                    questions.append(question)
                    domain = line2.strip()
                    domains.append(domain)
                    if len(questions) == args.eval_batch_size:
                        new_questions = list(tuple(question.split("[SEP]")) for question in questions)
                        answers = evaluate_batch_fn(args, model, new_questions, domains)
                        preds_file.write("\n".join(answers) + "\n")
                        preds_file.flush()
                        questions = []
                if len(questions) > 0:
                    new_questions = list(tuple(question.split("[SEP]")) for question in questions)
                    answers = evaluate_batch_fn(args, model, new_questions, domains)
                    preds_file.write("\n".join(answers))
                    preds_file.flush()
            else:
                for line in tqdm(eval_file):
                    question = line.strip()
                    questions.append(question)
                    if len(questions) == args.eval_batch_size:
                        new_questions = list(tuple(question.split("[SEP]")) for question in questions)
                        answers = evaluate_batch_fn(args, model, new_questions)
                        preds_file.write("\n".join(answers) + "\n")
                        preds_file.flush()
                        questions = []
                if len(questions) > 0:
                    new_questions = list(tuple(question.split("[SEP]")) for question in questions)
                    answers = evaluate_batch_fn(args, model, new_questions)
                    preds_file.write("\n".join(answers))
                    preds_file.flush()

        score_fn(args, args.predictions_path, args.gold_data_path)


if __name__ == "__main__":
    args = get_args()
    main(args)