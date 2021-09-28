# coding=utf-8
# Copyright 2020, The RAG Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" RAG model configuration """

import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.models.rag.configuration_rag import RagConfig


RAG_CONFIG_DOC = r"""
    :class:`~transformers.RagConfig` stores the configuration of a `RagModel`. Configuration objects inherit from
    :class:`~transformers.PretrainedConfig` and can be used to control the model outputs. Read the documentation from
    :class:`~transformers.PretrainedConfig` for more information.

    Args:
        title_sep (:obj:`str`, `optional`, defaults to  ``" / "``):
            Separator inserted between the title and the text of the retrieved document when calling
            :class:`~transformers.RagRetriever`.
        doc_sep (:obj:`str`, `optional`, defaults to  ``" // "``):
            Separator inserted between the the text of the retrieved document and the original input when calling
            :class:`~transformers.RagRetriever`.
        n_docs (:obj:`int`, `optional`, defaults to 5):
            Number of documents to retrieve.
        max_combined_length (:obj:`int`, `optional`, defaults to 300):
            Max length of contextualized input returned by :meth:`~transformers.RagRetriever.__call__`.
        retrieval_vector_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the document embeddings indexed by :class:`~transformers.RagRetriever`.
        retrieval_batch_size (:obj:`int`, `optional`, defaults to 8):
            Retrieval batch size, defined as the number of queries issues concurrently to the faiss index encapsulated
            :class:`~transformers.RagRetriever`.
        dataset (:obj:`str`, `optional`, defaults to :obj:`"wiki_dpr"`):
            A dataset identifier of the indexed dataset in HuggingFace Datasets (list all available datasets and ids
            using :obj:`datasets.list_datasets()`).
        dataset_split (:obj:`str`, `optional`, defaults to :obj:`"train"`)
            Which split of the :obj:`dataset` to load.
        index_name (:obj:`str`, `optional`, defaults to :obj:`"compressed"`)
            The index name of the index associated with the :obj:`dataset`. One can choose between :obj:`"legacy"`,
            :obj:`"exact"` and :obj:`"compressed"`.
        index_path (:obj:`str`, `optional`)
            The path to the serialized faiss index on disk.
        passages_path: (:obj:`str`, `optional`):
            A path to text passages compatible with the faiss index. Required if using
            :class:`~transformers.models.rag.retrieval_rag.LegacyIndex`
        use_dummy_dataset (:obj:`bool`, `optional`, defaults to ``False``)
            Whether to load a "dummy" variant of the dataset specified by :obj:`dataset`.
        label_smoothing (:obj:`float`, `optional`, defaults to 0.0):
            Only relevant if ``return_loss`` is set to :obj:`True`. Controls the ``epsilon`` parameter value for label
            smoothing in the loss calculation. If set to 0, no label smoothing is performed.
        do_marginalize (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If :obj:`True`, the logits are marginalized over all documents by making use of
            ``torch.nn.functional.log_softmax``.
        reduce_loss (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to reduce the NLL loss using the ``torch.Tensor.sum`` operation.
        do_deduplication (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not to deduplicate the generations from different context documents for a given input. Has to be
            set to :obj:`False` if used while training with distributed backend.
        exclude_bos_score (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not to disregard the BOS token when computing the loss.
        output_retrieved(:obj:`bool`, `optional`, defaults to :obj:`False`):
            If set to ``True``, :obj:`retrieved_doc_embeds`, :obj:`retrieved_doc_ids`, :obj:`context_input_ids` and
            :obj:`context_attention_mask` are returned. See returned tensors for more detail.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models).
        forced_eos_token_id (:obj:`int`, `optional`):
            The id of the token to force as the last generated token when :obj:`max_length` is reached. Usually set to
            :obj:`eos_token_id`.
"""


@add_start_docstrings(RAG_CONFIG_DOC)
class DialDocRagConfig(RagConfig):
    model_type = "rag"
    is_composition = True

    def __init__(self, mapping_file=None, segmentation=None, scoring_func="reranking", dataset="multidoc2dial", *args, **kwargs):
        self.mapping_file = mapping_file
        self.segmentation = segmentation
        self.dataset = dataset
        self.scoring_func = scoring_func
        super(DialDocRagConfig, self).__init__(*args, **kwargs)
