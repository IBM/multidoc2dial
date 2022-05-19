from typing import List, Optional, Tuple

import os
import time
import torch
import numpy as np
import json

from transformers.models.rag.retrieval_rag import (
    HFIndexBase,
    RagRetriever,
    LegacyIndex,
    CustomHFIndex,
    CanonicalHFIndex,
    LEGACY_INDEX_PATH,
)
from transformers.models.rag.tokenization_rag import RagTokenizer
from transformers.file_utils import requires_backends
from transformers.tokenization_utils_base import BatchEncoding

from transformers.utils import logging

from dialdoc.models.rag.configuration_rag_dialdoc import DialDocRagConfig

logger = logging.get_logger(__name__)


class DialDocIndex(CustomHFIndex):
    def load_pid_domain_mapping(self, mapping_file):
        with open(mapping_file, "r") as f_in:
            map = json.load(f_in)

        new_map = {}
        for k, v in map.items():
            new_map[int(k)] = v
        del map
        self.mapping = new_map

    def get_top_docs(self, question_hidden_states: np.ndarray, n_docs=5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, ids = self.dataset.search_batch("embeddings", question_hidden_states, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return (
            np.array(ids),
            np.array(vectors),
            np.array(scores),
        )  # shapes (batch_size, n_docs), (batch_size, n_docs, d) and (batch_size, n_docs)

    def search_batch_domain(self, embeddings, domain, n_docs=5):
        scores, ids = self.dataset.search_batch("embeddings", embeddings, 1200)
        filtered_scores, filtered_ids = [], []
        for i in range(len(ids)):
            dom = domain[i]
            f_s, f_id = [], []
            for score, id in zip(scores[i], ids[i]):
                if id != -1 and self.mapping[id] == dom:
                    f_s.append(score)
                    f_id.append(id)
                if len(f_id) == n_docs:
                    filtered_scores.append(f_s)
                    filtered_ids.append(f_id)
                    break
            if 0 < len(f_id) < n_docs:  ## bandage for cases where the retriever finds less than n_docs
                while len(f_id) < n_docs:
                    f_id.append(f_id[0])
                    f_s.append(f_s[0])
                filtered_scores.append(f_s)
                filtered_ids.append(f_id)
            ## TODO: what happens if none of the retrieved docs are not in GT domain

        return filtered_scores, filtered_ids

    def get_top_docs_domain(
        self, question_hidden_states: np.ndarray, domain, n_docs=5
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores, ids = self.search_batch_domain(question_hidden_states, domain, n_docs)
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])

        return (
            np.array(ids),
            np.array(vectors),
            np.array(scores),
        )  # shapes (batch_size, n_docs), (batch_size, n_docs, d) and (batch_size, n_docs)

    def get_top_docs_rerank_domain(
        self,
        combined_hidden_states: np.ndarray,
        current_hidden_states: np.ndarray,
        n_docs=5,
        dialog_lengths=None,
        domain=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores1, ids1 = self.search_batch_domain(combined_hidden_states, domain, n_docs)
        scores2, ids2 = self.search_batch_domain(current_hidden_states, domain, n_docs)
        ids3 = [[None] * (n_docs * 2)] * len(ids1)
        scores3 = [[0] * (n_docs * 2)] * len(ids1)
        scores = []
        ids = []
        for r in range(len(ids1)):
            if dialog_lengths:
                if dialog_lengths[r][0] < 10:
                    ids.append(ids1[r])
                    scores.append(scores1[r])
                    continue
            n1, n2 = len(ids1[r]), len(ids2[r])
            i = j = k = 0
            while i < n1 and j < n2:
                if scores1[r][i] >= scores2[r][j]:
                    ids3[r][k] = ids1[r][i]
                    scores3[r][k] = scores1[r][i]
                    k, i = k + 1, i + 1
                else:
                    ids3[r][k] = ids2[r][j]
                    scores3[r][k] = scores2[r][i]
                    k, j = k + 1, j + 1
            while i < n1:
                ids3[r][k] = ids1[r][i]
                scores3[r][k] = scores1[r][i]
                k, i = k + 1, i + 1
            while j < n2:
                ids3[r][k] = ids2[r][j]
                scores3[r][k] = scores2[r][j]
                k, j = k + 1, j + 1
            ids_new = []
            scores_new = []
            for ii, ele in enumerate(ids3[r]):
                if ele not in ids_new:
                    ids_new.append(ele)
                    scores_new.append(scores3[r][ii])
            ids.append(ids_new[:n_docs])
            scores.append(scores_new[:n_docs])
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors), np.array(scores)

    def get_top_docs_multihandle(
        self,
        current_hidden_states: np.ndarray,
        history_hidden_states: np.ndarray,
        scoring_func,
        n_docs=5,
        dialog_lengths=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        total_docs = len(self.dataset)
        scores_current, ids_current = self.dataset.search_batch("embeddings", current_hidden_states, 500)
        scores_history, ids_history = self.dataset.search_batch("embeddings", history_hidden_states, 500)

        final_scores = []
        final_ids = []
        for i in range(len(ids_current)):
            ids_current_i, scores_current_i = ids_current[i], scores_current[i]
            ids_history_i, scores_history_i = ids_history[i], scores_history[i]

            scaling_factor = None
            if dialog_lengths:
                curr_length, history_length = dialog_lengths[i]
                scaling_factor = 1.2 if curr_length > 10 else 1.0

            ## common ids between question and history
            common_ids = set(ids_current_i).intersection(set(ids_history_i))
            common_ids = {i for i in common_ids if i >= 0}
            if len(common_ids) < n_docs:
                logger.info("Only {} common ids found".format(len(common_ids)))
                logger.info(
                    "Picking the best ids from top matches with current turn and adding them to common_ids until we reach n_docs={}".format(
                        n_docs
                    )
                )
                new_ids = []
                for id in ids_current_i:
                    if len(common_ids) == n_docs:
                        break
                    if id not in common_ids:
                        new_ids.append(id)
                        common_ids.add(id)

                ids_current_i_common, scores_current_i_common = self.filter_ids(
                    common_ids, ids_current_i, scores_current_i
                )
                ids_history_i_common, scores_history_i_common = self.filter_ids(
                    common_ids, ids_history_i, scores_history_i
                )

                doc_dicts = self.get_doc_dicts(np.array(new_ids))
                for j, id in enumerate(new_ids):
                    ids_history_i_common.append(id)
                    score = np.inner(history_hidden_states[i], doc_dicts[j]["embeddings"])
                    scores_history_i_common.append(score)

                assert len(ids_current_i_common) == len(ids_history_i_common)

            else:
                ## only keep ids and scores that are common between question and history
                ids_current_i_common, scores_current_i_common = self.filter_ids(
                    common_ids, ids_current_i, scores_current_i
                )
                ids_history_i_common, scores_history_i_common = self.filter_ids(
                    common_ids, ids_history_i, scores_history_i
                )

                assert len(ids_current_i_common) == len(ids_history_i_common)

            ## sort by ids
            q_doc_ids, q_doc_scores = zip(*sorted(zip(ids_current_i_common, scores_current_i_common)))
            h_doc_ids, h_doc_scores = zip(*sorted(zip(ids_history_i_common, scores_history_i_common)))

            q_doc_ids, q_doc_scores = list(q_doc_ids), list(q_doc_scores)
            h_doc_ids, h_doc_scores = list(h_doc_ids), list(h_doc_scores)

            assert q_doc_ids == h_doc_ids

            ## Combine scores using scoring function
            rescored_ids = []
            rescored_scores = []
            for id, q_score, h_score in zip(q_doc_ids, q_doc_scores, h_doc_scores):
                rescored_ids.append(id)
                inp = torch.Tensor([q_score, h_score])
                if scaling_factor:
                    rescored_scores.append(scoring_func(inp, scaling_factor).tolist())
                else:
                    rescored_scores.append(scoring_func(inp).tolist())

            rescored_scores, rescored_ids = zip(*sorted(zip(rescored_scores, rescored_ids), reverse=True))
            rescored_scores, rescored_ids = list(rescored_scores), list(rescored_ids)

            final_ids.append(rescored_ids[:n_docs])
            final_scores.append(rescored_scores[:n_docs])

        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in final_ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return (
            np.array(final_ids),
            np.array(vectors),
            np.array(final_scores),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def get_top_docs_rerank(
        self,
        combined_hidden_states: np.ndarray,
        current_hidden_states: np.ndarray,
        n_docs=5,
        dialog_lengths=None,
        domain=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scores1, ids1 = self.dataset.search_batch("embeddings", combined_hidden_states, n_docs)
        scores2, ids2 = self.dataset.search_batch("embeddings", current_hidden_states, n_docs)
        ids3 = [[None] * (n_docs * 2)] * len(ids1)
        scores3 = [[0] * (n_docs * 2)] * len(ids1)
        scores = []
        ids = []
        for r in range(len(ids1)):
            if dialog_lengths:
                if dialog_lengths[r][0] < 10:
                    ids.append(ids1[r])
                    scores.append(scores1[r])
                    continue
            n1, n2 = len(ids1[r]), len(ids2[r])
            i = j = k = 0
            while i < n1 and j < n2:
                if scores1[r][i] >= scores2[r][j]:
                    ids3[r][k] = ids1[r][i]
                    scores3[r][k] = scores1[r][i]
                    k, i = k + 1, i + 1
                else:
                    ids3[r][k] = ids2[r][j]
                    scores3[r][k] = scores2[r][i]
                    k, j = k + 1, j + 1
            while i < n1:
                ids3[r][k] = ids1[r][i]
                scores3[r][k] = scores1[r][i]
                k, i = k + 1, i + 1
            while j < n2:
                ids3[r][k] = ids2[r][j]
                scores3[r][k] = scores2[r][j]
                k, j = k + 1, j + 1
            ids_new = []
            scores_new = []
            for ii, ele in enumerate(ids3[r]):
                if ele not in ids_new:
                    ids_new.append(ele)
                    scores_new.append(scores3[r][ii])
            ids.append(ids_new[:n_docs])
            scores.append(scores_new[:n_docs])
        docs = [self.dataset[[i for i in indices if i >= 0]] for indices in ids]
        vectors = [doc["embeddings"] for doc in docs]
        for i in range(len(vectors)):
            if len(vectors[i]) < n_docs:
                vectors[i] = np.vstack([vectors[i], np.zeros((n_docs - len(vectors[i]), self.vector_size))])
        return np.array(ids), np.array(vectors), np.array(scores)


def get_top_n_indices(bm25, query, n=5):
    query = query.lower().split()
    scores = bm25.get_scores(query)
    scores_i = [(i, score) for i, score in enumerate(scores)]
    sorted_indices = sorted(scores_i, key=lambda score: score[1], reverse=True)
    return sorted_indices[:n]


class DialDocRagRetriever(RagRetriever):
    def __init__(self, config, question_encoder_tokenizer, generator_tokenizer, index=None, init_retrieval=True):
        super().__init__(
            config, question_encoder_tokenizer, generator_tokenizer, index=index, init_retrieval=init_retrieval
        )
        if config.scoring_func in ["domain", "reranking_domain"]:
            self.index.load_pid_domain_mapping(config.mapping_file)

        if config.scoring_func == "nonlinear":
            logger.info("Using nonlinear scorer in RagRetriever")
            self.nn_scorer = torch.nn.Sequential(
                torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1), torch.nn.ReLU()
            )

    @staticmethod
    def _build_index(config):
        if config.index_name == "legacy":
            return LegacyIndex(
                config.retrieval_vector_size,
                config.index_path or LEGACY_INDEX_PATH,
            )
        elif config.index_name == "custom":
            return CustomHFIndex.load_from_disk(
                vector_size=config.retrieval_vector_size,
                dataset_path=config.passages_path,
                index_path=config.index_path,
            )
        elif config.index_name == "dialdoc":
            return DialDocIndex.load_from_disk(
                vector_size=config.retrieval_vector_size,
                dataset_path=config.passages_path,
                index_path=config.index_path,
            )
        else:
            return CanonicalHFIndex(
                vector_size=config.retrieval_vector_size,
                dataset_name=config.dataset,
                dataset_split=config.dataset_split,
                index_name=config.index_name,
                index_path=config.index_path,
                use_dummy_dataset=config.use_dummy_dataset,
            )

    @classmethod
    def from_pretrained(cls, retriever_name_or_path, indexed_dataset=None, **kwargs):
        requires_backends(cls, ["datasets", "faiss"])
        config = kwargs.pop("config", None) or DialDocRagConfig.from_pretrained(retriever_name_or_path, **kwargs)
        rag_tokenizer = RagTokenizer.from_pretrained(retriever_name_or_path, config=config)
        question_encoder_tokenizer = rag_tokenizer.question_encoder
        generator_tokenizer = rag_tokenizer.generator
        if indexed_dataset is not None:
            config.index_name = "custom"
            index = CustomHFIndex(config.retrieval_vector_size, indexed_dataset)
        else:
            index = cls._build_index(config)
        return cls(
            config,
            question_encoder_tokenizer=question_encoder_tokenizer,
            generator_tokenizer=generator_tokenizer,
            index=index,
        )

    def postprocess_docs(self, docs, input_strings, prefix, n_docs, return_tensors=None):
        r"""
        Postprocessing retrieved ``docs`` and combining them with ``input_strings``.

        Args:
            docs  (:obj:`dict`):
                Retrieved documents.
            input_strings (:obj:`str`):
                Input strings decoded by ``preprocess_query``.
            prefix (:obj:`str`):
                Prefix added at the beginning of each input, typically used with T5-based models.

        Return:
            :obj:`tuple(tensors)`: a tuple consisting of two elements: contextualized ``input_ids`` and a compatible
            ``attention_mask``.
        """

        def cat_input_and_doc(doc_title, doc_text, input_string, prefix):
            if doc_title.startswith('"'):
                doc_title = doc_title[1:]
            if doc_title.endswith('"'):
                doc_title = doc_title[:-1]
            if prefix is None:
                prefix = ""
            out = (prefix + input_string + self.config.doc_sep + doc_text).replace("  ", " ")

            return out

        rag_input_strings = [
            cat_input_and_doc(
                docs[i]["title"][j],
                docs[i]["text"][j],
                input_strings[i],
                prefix,
            )
            for i in range(len(docs))
            for j in range(n_docs)
        ]

        contextualized_inputs = self.generator_tokenizer.batch_encode_plus(
            rag_input_strings,
            max_length=self.config.max_combined_length,
            return_tensors=return_tensors,
            padding="max_length",
            truncation=True,
        )

        return contextualized_inputs["input_ids"], contextualized_inputs["attention_mask"]

    def _main_retrieve(
        self,
        combined_hidden_states: np.ndarray,
        current_hidden_states: np.ndarray,
        history_hidden_states: np.ndarray,
        n_docs: int,
        dialog_lengths: List[Tuple] = None,
        domain: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        def linear(a: List[int]):
            return sum(a)

        def linear2(a: List[int]):
            return a[0] + 0.5 * a[1]

        def linear3(a: List[int], scaling_factor=1):
            return scaling_factor * a[0] + 0.5 * a[1]

        def nonlinear(a: List[int]):
            with torch.no_grad():
                return self.nn_scorer(a)

        combined_hidden_states_batched = self._chunk_tensor(combined_hidden_states, self.batch_size)
        current_hidden_states_batched = self._chunk_tensor(current_hidden_states, self.batch_size)
        history_hidden_states_batched = self._chunk_tensor(history_hidden_states, self.batch_size)
        if (domain is None or len(domain) == 0) and self.config.scoring_func != "domain":
            domain_batched = [[""]] * len(combined_hidden_states_batched)
        else:
            domain_batched = self._chunk_tensor(domain, self.batch_size)
        ids_batched = []
        vectors_batched = []
        scores_batched = []
        for comb_h_s, curr_h_s, hist_h_s in zip(
            combined_hidden_states_batched,
            current_hidden_states_batched,
            history_hidden_states_batched,
            # domain_batched,
        ):
            start_time = time.time()
            if self.config.scoring_func in ["linear", "linear2", "linear3", "nonlinear"]:
                if self.config.scoring_func == "linear":
                    dialog_lengths = None
                    scoring_func = linear
                elif self.config.scoring_func == "linear2":
                    dialog_lengths = None
                    scoring_func = linear2
                elif self.config.scoring_func == "linear3":
                    scoring_func = linear3
                else:
                    dialog_lengths = None
                    scoring_func = nonlinear
                ids, vectors, scores = self.index.get_top_docs_multihandle(
                    curr_h_s, hist_h_s, scoring_func, n_docs, dialog_lengths=dialog_lengths
                )
            elif self.config.scoring_func in ["reranking_original", "reranking"]:
                ids, vectors, scores = self.index.get_top_docs_rerank(comb_h_s, curr_h_s, n_docs, None, dom_batch)
            elif self.config.scoring_func == "reranking2":
                ids, vectors, scores = self.index.get_top_docs_rerank(
                    comb_h_s, curr_h_s, n_docs, dialog_lengths=dialog_lengths
                )
            elif self.config.scoring_func in ["current_original", "current_pooled"]:
                ids, vectors, scores = self.index.get_top_docs(curr_h_s, n_docs)
            elif self.config.scoring_func in ["domain"]:
                ids, vectors, scores = self.index.get_top_docs_domain(comb_h_s, dom_batch, n_docs)
            elif self.config.scoring_func in ["reranking_domain"]:
                ids, vectors, scores = self.index.get_top_docs_rerank_domain(
                    comb_h_s, curr_h_s, n_docs, None, dom_batch
                )
            else:
                ids, vectors, scores = self.index.get_top_docs(comb_h_s, n_docs)
            logger.debug(f"index search time: {time.time() - start_time} sec, batch size {comb_h_s.shape}")
            ids_batched.extend(ids)
            vectors_batched.extend(vectors)
            scores_batched.extend(scores)
        return (
            np.array(ids_batched),
            np.array(vectors_batched),
            np.array(scores_batched),
        )  # shapes (batch_size, n_docs) and (batch_size, n_docs, d)

    def retrieve(
        self,
        combined_hidden_states: np.ndarray,
        current_hidden_states: np.ndarray,
        history_hidden_states: np.ndarray,
        n_docs: int,
        dialog_lengths: List[Tuple] = None,
        domain: List[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
        """
        Retrieves documents for specified ``question_hidden_states``.

        Args:
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`):
                A batch of query vectors to retrieve with.
            n_docs (:obj:`int`):
                The number of docs retrieved per query.

        Return:
            :obj:`Tuple[np.ndarray, np.ndarray, List[dict]]`: A tuple with the following objects:

            - **retrieved_doc_embeds** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs, dim)`) -- The retrieval
              embeddings of the retrieved docs per query.
            - **doc_ids** (:obj:`np.ndarray` of shape :obj:`(batch_size, n_docs)`) -- The ids of the documents in the
              index
            - **doc_dicts** (:obj:`List[dict]`): The :obj:`retrieved_doc_embeds` examples per query.
        """

        doc_ids, retrieved_doc_embeds, doc_scores = self._main_retrieve(
            combined_hidden_states, current_hidden_states, history_hidden_states, n_docs, dialog_lengths, domain
        )
        return retrieved_doc_embeds, doc_ids, doc_scores, self.index.get_doc_dicts(doc_ids)

    def __call__(
        self,
        question_input_ids: List[List[int]],
        combined_hidden_states: np.ndarray,
        current_hidden_states: np.ndarray,
        history_hidden_states: np.ndarray,
        dialog_lengths: List[Tuple],
        domain: List[str],
        prefix=None,
        n_docs=None,
        return_tensors=None,
        bm25=None,
    ) -> BatchEncoding:
        """
        Retrieves documents for specified :obj:`question_hidden_states`.

        Args:
            question_input_ids: (:obj:`List[List[int]]`) batch of input ids
            question_hidden_states (:obj:`np.ndarray` of shape :obj:`(batch_size, vector_size)`:
                A batch of query vectors to retrieve with.
            prefix: (:obj:`str`, `optional`):
                The prefix used by the generator's tokenizer.
            n_docs (:obj:`int`, `optional`):
                The number of docs retrieved per query.
            return_tensors (:obj:`str` or :class:`~transformers.file_utils.TensorType`, `optional`, defaults to "pt"):
                If set, will return tensors instead of list of python integers. Acceptable values are:

                * :obj:`'tf'`: Return TensorFlow :obj:`tf.constant` objects.
                * :obj:`'pt'`: Return PyTorch :obj:`torch.Tensor` objects.
                * :obj:`'np'`: Return Numpy :obj:`np.ndarray` objects.

        Returns: :class:`~transformers.BatchEncoding`: A :class:`~transformers.BatchEncoding` with the following
        fields:

            - **context_input_ids** -- List of token ids to be fed to a model.

              `What are input IDs? <../glossary.html#input-ids>`__

            - **context_attention_mask** -- List of indices specifying which tokens should be attended to by the model
            (when :obj:`return_attention_mask=True` or if `"attention_mask"` is in :obj:`self.model_input_names`).

              `What are attention masks? <../glossary.html#attention-mask>`__

            - **retrieved_doc_embeds** -- List of embeddings of the retrieved documents
            - **doc_ids** -- List of ids of the retrieved documents
        """

        n_docs = n_docs if n_docs is not None else self.n_docs
        prefix = prefix if prefix is not None else self.config.generator.prefix

        input_strings = self.question_encoder_tokenizer.batch_decode(question_input_ids, skip_special_tokens=True)
        if self.config.bm25:
            doc_ids = []
            doc_scores = []
            for input_string in input_strings:
                # doc_ids.append(self.config.bm25.get(input_string, [])[:self.config.n_docs])
                # doc_scores = ???
                sorted_indices = get_top_n_indices(bm25, input_string, self.config.n_docs)
                doc_ids.append([x[0] for x in sorted_indices])
                doc_scores.append([x[-1] for x in sorted_indices])
            docs = self.index.get_doc_dicts(np.array(doc_ids))

            retrieved_doc_embeds = [docs[i]["embeddings"] for i in range(len(doc_ids))]
        else:
            retrieved_doc_embeds, doc_ids, doc_scores, docs = self.retrieve(
                combined_hidden_states=combined_hidden_states,
                current_hidden_states=current_hidden_states,
                history_hidden_states=history_hidden_states,
                n_docs=n_docs,
                dialog_lengths=dialog_lengths,
                domain=domain,
            )
        context_input_ids, context_attention_mask = self.postprocess_docs(
            docs, input_strings, prefix, n_docs, return_tensors=return_tensors
        )

        return BatchEncoding(
            {
                "context_input_ids": context_input_ids,
                "context_attention_mask": context_attention_mask,
                "retrieved_doc_embeds": retrieved_doc_embeds,
                "doc_ids": doc_ids,
                "doc_scores": doc_scores,
            },
            tensor_type=return_tensors,
        )
