""" F1_score is from Official evaluation script for v1.1 of the SQuAD dataset. """
import json
import string
import re
import argparse
from collections import Counter
from datasets import load_metric
from rag.utils_rag import f1_score, exact_match_score


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def matching_evaluate(references, predictions):
    f1 = em = total = 0
    for id_, ref_text in references.items():
        total += 1
        ground_truths = [ref_text]
        prediction = predictions[id_]
        f1 += metric_max_over_ground_truths(f1_score, prediction, ground_truths)
        em += metric_max_over_ground_truths(exact_match_score, prediction, ground_truths)
    f1 = 100.0 * f1 / total
    em = 100.0 * em / total

    return f1, em


def matching_metrics(task, reference_json, prediction_json):
    d_id_reference = {}
    references_text = []
    references_list = []
    with open(reference_json) as fp_ref:
        data = json.load(fp_ref)
        for d_ref in data:
            d_id_reference[d_ref["id"]] = d_ref["text"]
            references_list.append([d_ref["text"]])
            references_text.append(d_ref["text"])
    predictions = []
    d_id_prediction = {}
    with open(prediction_json) as fp_pred:
        data = json.load(fp_pred)
        for d_pred in data:
            d_id_prediction[d_pred["id"]] = d_pred["text"]
            predictions.append(d_pred["text"])
    assert (
        len(predictions) == len(references_list) == len(references_text)
    ), "Ensure the matching count of instances of references and predictioins"

    f1_score, em_score = matching_evaluate(references=d_id_reference, predictions=d_id_prediction)
    metric_sacrebleu = load_metric("sacrebleu")
    results = metric_sacrebleu.compute(predictions=predictions, references=references_list)
    sacrebleu_score = results["score"]

    metric_meteor = load_metric("meteor")
    results = metric_meteor.compute(predictions=predictions, references=references_text)
    meteor_score = round(results["meteor"] * 100, 4)

    metric_rouge = load_metric("rouge")
    results = metric_rouge.compute(predictions=predictions, references=references_text)
    rouge_score = round(results["rougeL"].mid.fmeasure * 100, 4)

    if task == "grounding":
        output = {"EM": em_score, "F1": f1_score}
    else:
        output = {"F1": f1_score, "SACREBLEU": sacrebleu_score, "METEOR": meteor_score, "ROUGE-L": rouge_score}
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Select metrics for task that is either 'grounding' or 'utterance' generation",
    )
    parser.add_argument(
        "--prediction_json",
        type=str,
        required=True,
        help="Path to predictions",
    )
    parser.add_argument(
        "--reference_json",
        type=str,
        required=True,
        help="Path to references",
    )

    args = parser.parse_args()
    output = matching_metrics(args.task, args.reference_json, args.prediction_json)
    print("task:", args.task)
    print("output:", output)


if __name__ == "__main__":
    """
    task: grounding
    output: {'EM': 20.0, 'F1': 28.047519076264688}

    task: utterance
    output: {'F1': 13.25862068965517, 'SACREBLEU': 1.5941520509774114, 'METEOR': 8.3403, 'ROUGE-L': 9.0637}
    """
    main()
