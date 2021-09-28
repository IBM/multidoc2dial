import json
import os
import argparse
import csv
import sys
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from rank_bm25 import BM25Okapi
from datasets import load_dataset


DOMAINS = ["va", "ssa", "dmv", "studentaid"]
SEP = "####"  # separator for passages

sys.path.insert(2, str(Path(__file__).resolve().parents[1]))


def rm_blank(text, is_shortern=False):
    text = text.replace(" ", "").replace("\n", "").replace("\t", "").replace("\r", "")
    if is_shortern:
        text = text[3:-3]
    return text


def text2line(text):
    return text.replace("\n", " ").replace("\r", " ").replace("\t", " ").strip()


def split_text_section(spans, title, args):
    def get_text(buff, title, span):
        text = " ".join(buff).replace("\n", " ")
        parent_titles = [title.replace("/", "-").rsplit("#")[0]]
        if len(span["parent_titles"]["text"]) > 1:
            parent_titles = [ele.replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]["text"]]
        text = " / ".join(parent_titles) + " // " + text
        return text2line(text)

    buff = []
    pre_sec, pre_title, pre_span = None, None, None
    passages = []
    subtitles = []
    for span in spans:
        parent_titles = title
        if len(span["parent_titles"]["text"]) > 1:
            parent_titles = [ele.replace("/", "-").rsplit("#")[0] for ele in span["parent_titles"]["text"]]
            parent_titles = " / ".join(parent_titles)
        if pre_sec == span["id_sec"] or pre_title == span["title"].strip():
            buff.append(span["text_sp"])
        elif buff:
            text = get_text(buff, title, pre_span)
            passages.append(text)
            subtitles.append(parent_titles)
            buff = [span["text_sp"]]
        else:
            buff.append(span["text_sp"])
        pre_sec = span["id_sec"]
        pre_span = span
        pre_title = span["title"].strip()
    if buff:
        text = get_text(buff, title, span)
        passages.append(text)
        subtitles.append(parent_titles)
    return passages, subtitles


def split_text(text: str, n=100, character=" "):
    """Split the text every ``n``-th occurrence of ``character``"""
    text = text.split(character)
    passages = [character.join(text[i : i + n]).strip() for i in range(0, len(text), n)]
    return [passage for passage in passages if len(passage) > 0]


def get_bm25(passages):
    passages_tokenized = [passage.strip().lower().split() for passage in passages]
    bm25 = BM25Okapi(passages_tokenized)
    return bm25


def get_top_n_indices(bm25, query, n=5):
    query = query.lower().split()
    scores = bm25.get_scores(query)
    scores_i = [(i, score) for i, score in enumerate(scores)]
    sorted_indices = sorted(scores_i, key=lambda score: score[1], reverse=True)
    return [x[0] for x in sorted_indices[:n]]


def get_positive_passages(positive_pids, doc_scores, passage_map):
    """
    Get positive passages for a given grounding using BM25 scores from the positive passage pool
    Parameters:
        positive_pids: list
            Positive passage indices
        doc_scores: list
            BM25 scores against the query's grounding for all passages
        passage_map: dict
            All passages mapped with their ids
    Returns:
        positive_passage_pool
    """
    scores = [(i, score) for (i, score) in doc_scores if i in positive_pids]
    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)

    top_n_passages = [
        {"psg_id": ix, "score": score, "title": passage_map[ix]["title"], "text": passage_map[ix]["text"]}
        for ix, score in top_scores
    ]

    return top_n_passages


def get_negative_passages(positive_pids, doc_scores, passage_map, begin=5, n=10):
    """
    Get hard negative passages for a given grounding using BM25 scores across all passages.
    Filter out all passages from the query's positive passage pool
    """
    scores = [(i, score) for (i, score) in doc_scores if i not in positive_pids]
    top_scores = sorted(scores, key=lambda x: x[1], reverse=True)
    negative_passages = [
        {"psg_id": ix, "score": score, "title": passage_map[ix]["title"], "text": passage_map[ix]["text"]}
        for ix, score in top_scores[begin : begin + n]
    ]
    assert len(negative_passages) == n
    return negative_passages


def create_dpr_data(args):
    dd = DD_Loader(args)
    args.split = "train" if not args.split else args.split
    dd.get_doc_passages(args)
    doc_passages = dd.d_doc_psg
    all_passages = dd.doc_psg_all
    all_domains = dd.doc_domain_all

    d_in = dd.get_dial(args)
    source = d_in["source"]
    target = d_in["target"]
    qids = d_in["qid"]
    titles = d_in["title"]
    pids = d_in["pid"]
    domains = d_in["domain"]

    passage_map = {}
    for title in doc_passages:
        psg_start_ix = doc_passages[title][0]
        n_psgs = doc_passages[title][1]
        for i in range(n_psgs):
            passage_map[psg_start_ix + i] = {"text": all_passages[psg_start_ix + i], "title": title}

    # Create passage index using BM25
    print("Creating passage index ...")
    bm25 = get_bm25(all_passages)

    dataset = []
    for qid, query, grounding, title, pid_pos, domain in tqdm(
        zip(qids, source, target, titles, pids, domains), total=len(source), desc="Creating dataset ..."
    ):
        if args.last_turn_only:
            query = query.split("[SEP]")[0].strip()
        scores_g = bm25.get_scores(grounding.strip().lower().split())
        if args.in_domain_only:
            doc_scores_g = []
            for idx, score in enumerate(scores_g):
                if dd.doc_domain_all[idx] == domain:
                    doc_scores_g.append((idx, score))
        else:
            doc_scores_g = [(i, score) for i, score in enumerate(scores_g)]
        positive_passages = get_positive_passages(
            positive_pids=pid_pos, doc_scores=doc_scores_g, passage_map=passage_map
        )
        hard_negative_passages = get_negative_passages(
            positive_pids=pid_pos, doc_scores=doc_scores_g, passage_map=passage_map
        )
        scores_q = bm25.get_scores(query.strip().lower().split())
        if args.in_domain_only:
            doc_scores_q = []
            for idx, score in enumerate(scores_q):
                if all_domains[idx] == domain:
                    doc_scores_q.append((idx, score))
        else:
            doc_scores_q = [(i, score) for i, score in enumerate(scores_q)]
        negative_passages = get_negative_passages(
            positive_pids=pid_pos, doc_scores=doc_scores_q, passage_map=passage_map
        )
        sample = {
            "dataset": args.dataset_config_name,
            "qid": qid,
            "question": query,
            "answers": [grounding],
            "positive_ctxs": positive_passages,
            "negative_ctxs": negative_passages,
            "hard_negative_ctxs": hard_negative_passages,
        }
        dataset.append(sample)
    os.makedirs(args.output_dir, exist_ok=True)
    if args.target_domain:
        config = f"{args.dataset_config_name}.{args.segmentation}"
    else:
        config = f"{args.dataset_config_name}_all.{args.segmentation}"
    outfile = os.path.join(args.output_dir, f"dpr.{config}.{args.split}.json")
    print("Writing dataset to {}".format(outfile))
    with open(outfile, "w") as f:
        json.dump(dataset, f, indent=4)
    passage_file = os.path.join(args.output_dir, f"dpr.psg.{config}.json")
    passages = []
    for k, v in sorted(passage_map.items()):
        v.update({"id": k})
        passages.append(v)
    with open(passage_file, "w") as f:
        json.dump(passages, f, indent=4)


def map_passages(grounding, all_psgs, start_idx, num_psg):
    mapping = []
    for start in range(start_idx, start_idx + num_psg):
        current_mapping = []
        for end in range(start + 1, start_idx + num_psg + 1):
            content = "".join(all_psgs[start:end])
            if grounding in content or rm_blank(grounding.lower(), True) in rm_blank(content.lower()):
                current_mapping = list(range(start, end))
            if len(current_mapping) == 1:
                return current_mapping
            elif len(current_mapping) > 1:
                break
        if current_mapping:
            mapping = current_mapping
    return mapping


def load_doc_dataset(args):
    doc_data = load_dataset(args.dataset_name, "document_domain", split="train", ignore_verifications=True)
    return doc_data


class DD_Loader:
    def __init__(self, args) -> None:
        self.doc_dataset = load_doc_dataset(args)
        self.dial_dataset = load_dataset(
            args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, ignore_verifications=True
        )
        self.d_doc_data = defaultdict(dict)  # doc -> "doc_text", "spans"
        self.d_doc_psg = {}
        self.doc_psg_all = []
        self.doc_domain_all = []
        self.d_pid_domain = {}

    def reset(self):
        self.d_doc_data = defaultdict(dict)
        self.d_doc_psg = {}
        self.doc_psg_all = []
        self.doc_domain_all = []
        self.d_pid_domain = {}

    def get_doc_passages(self, args):
        # self.doc_dataset = load_doc_dataset(args)
        start_idx = 0
        for ex in self.doc_dataset:
            if args.target_domain and ex["domain"] not in args.included_domains:
                continue
            if args.segmentation == "token":
                passages = split_text(ex["doc_text"])
            else:
                passages, subtitles = split_text_section(ex["spans"], ex["title"], args)
            self.doc_psg_all.extend(passages)
            self.doc_domain_all.extend([ex["domain"]] * len(passages))
            self.d_doc_psg[ex["doc_id"]] = (start_idx, len(passages))
            for i in range(start_idx, start_idx + len(passages)):
                self.d_pid_domain[i] = ex["domain"]
            start_idx += len(passages)
            self.d_doc_data[ex["doc_id"]]["doc_text"] = ex["doc_text"]
            self.d_doc_data[ex["doc_id"]]["spans"] = {}
            self.d_doc_data[ex["doc_id"]]["domain"] = ex["domain"]
            for d_span in ex["spans"]:
                self.d_doc_data[ex["doc_id"]]["spans"][d_span["id_sp"]] = d_span

    def get_dial(self, args):
        source, target, qids, titles, pids, domains, das = [], [], [], [], [], [], []
        # self.dial_dataset = load_dataset(
        #     args.dataset_name, args.dataset_config_name, cache_dir=args.cache_dir, ignore_verifications=True
        # )

        for ex in self.dial_dataset[args.split]:
            qid = ex["id"]
            doc_id = ex["title"]
            query = ex["question"]
            domain = ex["domain"]
            da = ex["da"]
            if args.num_token > 0:
                query = " ".join(query.split()[: args.num_token])
            grounding = ex["answers"]["text"][0]
            utterance = ex.get("utterance", "")
            source_txt = text2line(query)
            target_txt = text2line(utterance) if args.task == "generation" else text2line(grounding)
            if not source_txt or not target_txt:
                continue
            start_idx, num_psg = self.d_doc_psg[doc_id]
            pids_pos = map_passages(grounding, self.doc_psg_all, start_idx, num_psg)
            source.append(source_txt)
            target.append(target_txt)
            qids.append(qid)
            pids.append(pids_pos)
            titles.append(doc_id)
            domains.append(domain)
            das.append(da)
        d_out = {
            "source": source,
            "target": target,
            "qid": qids,
            "title": titles,
            "pid": pids,
            "domain": domains,
            "da": das,
        }
        return d_out

    def save_kb_files(self, args):
        os.makedirs(args.kb_dir, exist_ok=True)
        if args.target_domain and len(args.included_domains) > 1:
            config = f"{args.segmentation}-wo-{args.target_domain}"
        elif args.target_domain and len(args.included_domains) == 1:
            config = f"{args.segmentation}-{args.target_domain}"
        else:
            config = f"{args.segmentation}-all"
        with open(
            os.path.join(args.kb_dir, f"mdd-{config}.csv"),
            "w",
            encoding="utf8",
        ) as fp:
            csv_writer = csv.writer(fp, delimiter="\t")
            for k, (start_id, num_psg) in self.d_doc_psg.items():
                psgs = [text2line(e) for e in self.doc_psg_all[start_id : start_id + num_psg]]
                csv_writer.writerow([k, SEP.join(psgs)])
        with open(os.path.join(args.kb_dir, f"pid_domain-{config}.json"), "w", encoding="utf8") as fp:
            json.dump(self.d_pid_domain, fp, indent=4)

    def save_dial_files(self, args, d_in):
        sp = "val" if args.split == "validation" else args.split
        if not args.output_dir:
            args.output_dir = f"data_mdd_wo_{args.target_domain}"
        od = f"{args.output_dir}/dd-{args.task}-{args.segmentation}"
        os.makedirs(od, exist_ok=True)
        source = d_in["source"]
        target = d_in["target"]
        qids = d_in["qid"]
        titles = d_in["title"]
        pids = d_in["pid"]
        domains = d_in["domain"]
        das = d_in["da"]

        with open(os.path.join(od, f"{sp}.domain"), "w", encoding="utf8") as fp:
            fp.write("\n".join(domains))
        with open(os.path.join(od, f"{sp}.da"), "w", encoding="utf8") as fp:
            fp.write("\n".join(das))
        with open(os.path.join(od, f"{sp}.source"), "w", encoding="utf8") as fp:
            fp.write("\n".join(source))
        with open(os.path.join(od, f"{sp}.target"), "w", encoding="utf8") as fp:
            fp.write("\n".join(target))
        with open(os.path.join(od, f"{sp}.qids"), "w", encoding="utf8") as fp:
            fp.write("\n".join(qids))
        with open(os.path.join(od, f"{sp}.titles"), "w", encoding="utf8") as fp:
            fp.write("\n".join(titles))
        with open(os.path.join(od, f"{sp}.pids"), "w", encoding="utf8") as fp:
            lines_pid = []
            for ids in pids:
                lines_pid.append("\t".join([str(e) for e in ids]))
            fp.write("\n".join(lines_pid))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="hf_datasets/doc2dial/doc2dial_pub.py",
        help="dataset name or path for data loader",
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default="multidoc2dial",
        help="hugging dataset config name",
    )
    parser.add_argument(
        "--target_domain",
        type=str,
        default="",  # default is empty, which indicates that all domains are included.
        help="target or test domain in domain adaptation setup, one domain from ssa, va, dmv, studentaid",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="path to output the data files",
    )
    parser.add_argument(
        "--kb_dir",
        type=str,
        default="YOUR_DIR/data_mdd_kb",
        help="path to output kb data files",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=os.environ["HF_HOME"],
        help="Path for caching the downloaded data by HuggingFace Datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="",
        help="Data split is 'train', 'validation' or 'test'",
    )
    parser.add_argument(
        "--last_turn_only",
        type=bool,
        help="Only include the latest turn in dialogue",
    )
    parser.add_argument(
        "--segmentation",
        type=str,
        default="structure",
        help="`token` or `structure`",
    )
    parser.add_argument(
        "--num_token",
        type=int,
        default=-1,
        help="number of tokens of a query; -1 indicates all tokens",
    )
    parser.add_argument(
        "--task",
        default="grounding",
        help="task: grounding, generation",
    )
    parser.add_argument(
        "--dpr",
        action="store_true",
        help="generate DPR data",
    )
    parser.add_argument(
        "--in_domain_only",
        action="store_true",
        help="bm25 retrievals within domain",
    )

    args = parser.parse_args()
    if not args.dataset_config_name:
        args.dataset_config_name = "multidoc2dial"
    if args.target_domain:
        args.dataset_config_name = f"multidoc2dial_{args.target_domain}"
    if not args.dpr:
        dd = DD_Loader(args)
        splits = [args.split] if args.split else ["train", "validation", "test"]  # test split at last
        if not args.target_domain:
            dd.get_doc_passages(args)
            dd.save_kb_files(args)
            for split in splits:
                args.split = split
                d_out = dd.get_dial(args)
                dd.save_dial_files(args, d_out)
        else:
            for split in splits:
                args.split = split
                if split == "test":
                    args.included_domains = [args.target_domain]
                    dd.reset()
                else:
                    args.included_domains = [ele for ele in DOMAINS if ele != args.target_domain]
                if not dd.doc_psg_all:
                    dd.get_doc_passages(args)
                d_out = dd.get_dial(args)
                dd.save_kb_files(args)
                dd.save_dial_files(args, d_out)
    else:
        if args.target_domain:
            args.included_domains = [ele for ele in DOMAINS if ele != args.target_domain]
        create_dpr_data(args)


if __name__ == "__main__":
    main()