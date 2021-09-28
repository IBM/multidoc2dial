from typing import List, Optional, Callable

import torch

from transformers.models.rag.modeling_rag import (
    RagModel,
    RagTokenForGeneration,
    RetrievAugLMOutput,
    RetrievAugLMMarginOutput,
)
from transformers.models.rag.retrieval_rag import RagRetriever
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel
from transformers.generation_beam_search import BeamSearchScorer

from dialdoc.models.rag.configuration_rag_dialdoc import DialDocRagConfig

from transformers.utils import logging

logger = logging.get_logger(__name__)


class DialDocRagModel(RagModel):
    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,
        **kwargs,
    ):
        self.config_class = DialDocRagConfig
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an question_encoder and a generator has to be provided."
        if config is None:
            config = DialDocRagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )
        else:
            assert isinstance(config, self.config_class), f"config: {config} has to be of type {self.config_class}"
        super(RagModel, self).__init__(config)
        if question_encoder is None:
            from transformers.models.auto.modeling_auto import AutoModel

            question_encoder = AutoModel.from_config(config.question_encoder)

        if generator is None:
            from transformers.models.auto.modeling_auto import AutoModelForSeq2SeqLM

            generator = AutoModelForSeq2SeqLM.from_config(config.generator)

        self.retriever = retriever
        if self.retriever is not None:
            assert isinstance(
                retriever, RagRetriever
            ), f"`self.retriever` is of type {type(self.retriever)}, but should be of type `RagRetriever`"
            self.retriever = retriever

        self.question_encoder = question_encoder
        self.generator = generator

        self.bm25 = kwargs.pop("bm25", None)
        if self.bm25:
            logger.info("Using BM25 inside RAG Model")

    @staticmethod
    def mean_pool(vector: torch.LongTensor):
        return vector.sum(axis=0) / vector.shape[0]

    @staticmethod
    def get_attn_mask(tokens_tensor: torch.LongTensor) -> torch.tensor:
        return tokens_tensor != 0

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        doc_scores=None,
        context_input_ids=None,
        context_attention_mask=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_retrieved=None,
        n_docs=None,
        domain=None,
    ):
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_retrieved = output_retrieved if output_retrieved is not None else self.config.output_retrieved

        # whether retriever has to be used
        has_to_retrieve = (
            self.retriever is not None
            and (context_input_ids is None or context_attention_mask is None or doc_scores is None)
            and encoder_outputs is None
        )
        # encoder_outputs are pre-computed during RAG-token generation
        if encoder_outputs is None:
            dialog_lengths = None
            if has_to_retrieve:
                question_enc_outputs = self.question_encoder(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
                )
                # if self.config.scoring_func in ['linear', 'linear2', 'linear3', 'nonlinear', 'reranking', 'reranking2']:
                if self.config.scoring_func != "original":
                    combined_out = question_enc_outputs.pooler_output

                    ## Get mask for current turn input ids
                    curr_turn_mask = torch.logical_xor(attention_mask, token_type_ids)
                    current_turn_input_ids = input_ids * curr_turn_mask
                    current_turn_only_out = self.question_encoder(
                        current_turn_input_ids, attention_mask=curr_turn_mask.long(), return_dict=True
                    )
                    current_turn_output = current_turn_only_out.pooler_output

                    ## Split the dpr sequence output
                    sequence_output = question_enc_outputs.hidden_states[-1]
                    attn_mask = self.get_attn_mask(input_ids)
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
                        seq_out_0.append(self.mean_pool(seq_out_masked_0))
                        seq_out_1.append(self.mean_pool(seq_out_masked_1))

                    pooled_output_q = torch.cat([seq.view(1, -1) for seq in seq_out_0], dim=0)
                    pooled_output_h = torch.cat([seq.view(1, -1) for seq in seq_out_1], dim=0)

                    if self.config.scoring_func in ["reranking_original", "current_original"]:
                        current_out = current_turn_output
                    else:
                        current_out = pooled_output_q

                    retriever_outputs = self.retriever(
                        input_ids,
                        combined_out.cpu().detach().to(torch.float32).numpy(),
                        current_out.cpu().detach().to(torch.float32).numpy(),
                        pooled_output_h.cpu().detach().to(torch.float32).numpy(),
                        prefix=self.generator.config.prefix,
                        n_docs=n_docs,
                        dialog_lengths=dialog_lengths,
                        domain=domain,
                        return_tensors="pt",
                    )
                else:
                    combined_out = question_enc_outputs[0]  # hidden states of question encoder

                    retriever_outputs = self.retriever(
                        input_ids,
                        combined_out.cpu().detach().to(torch.float32).numpy(),
                        combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                        combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                        prefix=self.generator.config.prefix,
                        n_docs=n_docs,
                        dialog_lengths=dialog_lengths,
                        domain=domain,
                        return_tensors="pt",
                        bm25=self.bm25,
                    )

                (
                    context_input_ids,
                    context_attention_mask,
                    retrieved_doc_embeds,
                    retrieved_doc_ids,
                    retrieved_doc_scores,
                ) = (
                    retriever_outputs["context_input_ids"],
                    retriever_outputs["context_attention_mask"],
                    retriever_outputs["retrieved_doc_embeds"],
                    retriever_outputs["doc_ids"],
                    retriever_outputs["doc_scores"],
                )

                # set to correct device
                retrieved_doc_embeds = retrieved_doc_embeds.to(combined_out)
                context_input_ids = context_input_ids.to(input_ids)
                context_attention_mask = context_attention_mask.to(input_ids)
                doc_scores = retrieved_doc_scores.to(combined_out)

                # compute doc_scores
                if self.config.scoring_func in [
                    "reranking",
                    "reranking2",
                    "original",
                    "reranking_original",
                    "current_original",
                    "current_pooled",
                ]:
                    doc_scores = torch.bmm(combined_out.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
                elif self.config.scoring_func in ["linear", "linear2", "linear3", "nonlinear"]:
                    doc_scores_curr = torch.bmm(
                        pooled_output_q.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)

                    doc_scores_hist = torch.bmm(
                        pooled_output_h.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                    ).squeeze(1)

                    if self.config.scoring_func == "linear":
                        doc_scores = doc_scores_curr + doc_scores_hist
                    elif self.config.scoring_func == "linear2":
                        doc_scores = doc_scores_curr + 0.5 * doc_scores_hist
                    elif self.config.scoring_func == "linear3":
                        # TODO: linear 3 scoring
                        doc_scores = doc_scores_curr + 0.5 * doc_scores_hist
                    else:  # nonlinear
                        bsz = doc_scores_curr.shape[0]
                        doc_scores_curr_flattened = doc_scores_curr.flatten().unsqueeze(
                            1
                        )  # from (B, n_docs) to (Bxn_docs, 1)
                        doc_scores_hist_flattened = doc_scores_hist.flatten().unsqueeze(
                            1
                        )  # from (B, n_docs) to (Bxn_docs, 1)
                        scorer_inp = torch.cat(
                            [doc_scores_curr_flattened, doc_scores_hist_flattened], dim=1
                        )  # (Bxn_docs, 2)
                        scores = self.retriever.nn_scorer(scorer_inp)
                        doc_scores = scores.reshape((bsz, -1))

            else:
                assert (
                    context_input_ids is not None
                ), "Make sure that `context_input_ids` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
                assert (
                    context_attention_mask is not None
                ), "Make sure that `context_attention_mask` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."
                assert (
                    doc_scores is not None
                ), "Make sure that `doc_scores` are passed, if no `retriever` is set. Alternatively, you can set a retriever using the `set_retriever(...)` function."

        assert (
            doc_scores is not None
        ), "Make sure that `doc_scores` are passed when passing `encoder_outputs` to the forward function."

        assert (
            doc_scores.shape[1] % n_docs
        ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # Decoder input without context documents
        if decoder_input_ids is not None:
            decoder_input_ids = decoder_input_ids.repeat_interleave(n_docs, dim=0)

        if decoder_attention_mask is not None:
            decoder_attention_mask = decoder_attention_mask.repeat_interleave(n_docs, dim=0)

        gen_outputs = self.generator(
            input_ids=context_input_ids,
            attention_mask=context_attention_mask,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            return_dict=True,
        )

        if not has_to_retrieve:
            combined_out = None
            question_enc_hidden_states = None
            question_enc_attentions = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None
        else:
            question_enc_hidden_states = question_enc_outputs.hidden_states
            question_enc_attentions = question_enc_outputs.attentions

        if not has_to_retrieve or not output_retrieved:
            # don't output retrieved docs
            context_input_ids = (None,)
            context_attention_mask = None
            retrieved_doc_embeds = None
            retrieved_doc_ids = None

        return RetrievAugLMOutput(
            logits=gen_outputs.logits,
            doc_scores=doc_scores,
            past_key_values=gen_outputs.past_key_values,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            retrieved_doc_embeds=retrieved_doc_embeds,
            retrieved_doc_ids=retrieved_doc_ids,
            question_encoder_last_hidden_state=combined_out,
            question_enc_hidden_states=question_enc_hidden_states,
            question_enc_attentions=question_enc_attentions,
            generator_enc_last_hidden_state=gen_outputs.encoder_last_hidden_state,
            generator_enc_hidden_states=gen_outputs.encoder_hidden_states,
            generator_enc_attentions=gen_outputs.encoder_attentions,
            generator_dec_hidden_states=gen_outputs.decoder_hidden_states,
            generator_dec_attentions=gen_outputs.decoder_attentions,
            generator_cross_attentions=gen_outputs.cross_attentions,
        )


class DialDocRagTokenForGeneration(RagTokenForGeneration):
    config_class = DialDocRagConfig

    def __init__(
        self,
        config: Optional[PretrainedConfig] = None,
        question_encoder: Optional[PreTrainedModel] = None,
        generator: Optional[PreTrainedModel] = None,
        retriever: Optional = None,
        bm25: Optional = None,
        **kwargs,
    ):
        assert config is not None or (
            question_encoder is not None and generator is not None
        ), "Either a configuration or an encoder and a generator has to be provided."

        if config is None:
            config = DialDocRagConfig.from_question_encoder_generator_configs(
                question_encoder.config, generator.config, **kwargs
            )

        super(RagTokenForGeneration, self).__init__(config)
        # instantiate model
        if bm25:
            logger.info("Using bm25")
            self.rag = DialDocRagModel(
                config=config, question_encoder=question_encoder, generator=generator, retriever=retriever, bm25=bm25
            )
            self.bm25 = bm25
        else:
            self.rag = DialDocRagModel(
                config=config, question_encoder=question_encoder, generator=generator, retriever=retriever
            )
            self.bm25 = None

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        past_key_values=None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        output_retrieved=None,
        do_marginalize=None,
        reduce_loss=None,
        labels=None,
        n_docs=None,
        domain=None,
        **kwargs,  # needs kwargs for generation
    ):
        r"""
        do_marginalize (:obj:`bool`, `optional`):
            If :obj:`True`, the logits are marginalized over all documents by making use of
            ``torch.nn.functional.log_softmax``.
        reduce_loss (:obj:`bool`, `optional`):
            Only relevant if ``labels`` is passed. If :obj:`True`, the NLL loss is reduced using the
            ``torch.Tensor.sum`` operation.
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Legacy dictionary, which is required so that model can use `generate()` function.

        Returns:

        Example::

            >>> from transformers import RagTokenizer, RagRetriever, RagTokenForGeneration
            >>> import torch

            >>> tokenizer = RagTokenizer.from_pretrained("facebook/rag-token-nq")
            >>> retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", index_name="exact", use_dummy_dataset=True)
            >>> # initialize with RagRetriever to do everything in one forward call
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", retriever=retriever)

            >>> inputs = tokenizer("How many people live in Paris?", return_tensors="pt")
            >>> with tokenizer.as_target_tokenizer():
            ...    targets = tokenizer("In Paris, there are 10 million people.", return_tensors="pt")
            >>> input_ids = inputs["input_ids"]
            >>> labels = targets["input_ids"]
            >>> outputs = model(input_ids=input_ids, labels=labels)

            >>> # or use retriever separately
            >>> model = RagTokenForGeneration.from_pretrained("facebook/rag-token-nq", use_dummy_dataset=True)
            >>> # 1. Encode
            >>> question_hidden_states = model.question_encoder(input_ids)[0]
            >>> # 2. Retrieve
            >>> docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors="pt")
            >>> doc_scores = torch.bmm(question_hidden_states.unsqueeze(1), docs_dict["retrieved_doc_embeds"].float().transpose(1, 2)).squeeze(1)
            >>> # 3. Forward to generator
            >>> outputs = model(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores, decoder_input_ids=labels)

            >>> # or directly generate
            >>> generated = model.generate(context_input_ids=docs_dict["context_input_ids"], context_attention_mask=docs_dict["context_attention_mask"], doc_scores=doc_scores)
            >>> generated_string = tokenizer.batch_decode(generated, skip_special_tokens=True)
        """
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        do_marginalize = do_marginalize if do_marginalize is not None else self.config.do_marginalize
        reduce_loss = reduce_loss if reduce_loss is not None else self.config.reduce_loss

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = labels
            use_cache = False

        outputs = self.rag(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            context_input_ids=context_input_ids,
            context_attention_mask=context_attention_mask,
            doc_scores=doc_scores,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_retrieved=output_retrieved,
            n_docs=n_docs,
            domain=domain,
        )

        loss = None
        logits = outputs.logits
        if labels is not None:
            assert decoder_input_ids is not None
            loss = self.get_nll(
                outputs.logits,
                outputs.doc_scores,
                labels,
                reduce_loss=reduce_loss,
                epsilon=self.config.label_smoothing,
                n_docs=n_docs,
            )

        if do_marginalize:
            logits = self.marginalize(logits, outputs.doc_scores, n_docs)

        return RetrievAugLMMarginOutput(
            loss=loss,
            logits=logits,
            doc_scores=outputs.doc_scores,
            past_key_values=outputs.past_key_values,
            context_input_ids=outputs.context_input_ids,
            context_attention_mask=outputs.context_attention_mask,
            retrieved_doc_embeds=outputs.retrieved_doc_embeds,
            retrieved_doc_ids=outputs.retrieved_doc_ids,
            question_encoder_last_hidden_state=outputs.question_encoder_last_hidden_state,
            question_enc_hidden_states=outputs.question_enc_hidden_states,
            question_enc_attentions=outputs.question_enc_attentions,
            generator_enc_last_hidden_state=outputs.generator_enc_last_hidden_state,
            generator_enc_hidden_states=outputs.generator_enc_hidden_states,
            generator_enc_attentions=outputs.generator_enc_attentions,
            generator_dec_hidden_states=outputs.generator_dec_hidden_states,
            generator_dec_attentions=outputs.generator_dec_attentions,
            generator_cross_attentions=outputs.generator_cross_attentions,
        )

    @staticmethod
    def mean_pool(vector: torch.LongTensor):
        return vector.sum(axis=0) / vector.shape[0]

    @staticmethod
    def get_attn_mask(tokens_tensor: torch.LongTensor) -> torch.tensor:
        return tokens_tensor != 0

    def generate(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        context_input_ids=None,
        context_attention_mask=None,
        doc_scores=None,
        domain=None,
        max_length=None,
        min_length=None,
        early_stopping=None,
        use_cache=None,
        num_beams=None,
        num_beam_groups=None,
        diversity_penalty=None,
        bos_token_id=None,
        pad_token_id=None,
        eos_token_id=None,
        length_penalty=None,
        no_repeat_ngram_size=None,
        encoder_no_repeat_ngram_size=None,
        repetition_penalty=None,
        bad_words_ids=None,
        num_return_sequences=None,
        decoder_start_token_id=None,
        n_docs=None,
        prefix_allowed_tokens_fn: Callable[[int, torch.Tensor], List[int]] = None,
        forced_bos_token_id: Optional[int] = None,
        forced_eos_token_id: Optional[int] = None,
        remove_invalid_values: Optional[bool] = None,
        **model_kwargs,
    ):
        # set default parameters
        n_docs = n_docs if n_docs is not None else self.config.n_docs
        num_beams = num_beams if num_beams is not None else self.config.num_beams
        num_beam_groups = num_beam_groups if num_beam_groups is not None else self.config.num_beam_groups
        max_length = max_length if max_length is not None else self.config.max_length
        num_return_sequences = (
            num_return_sequences if num_return_sequences is not None else self.config.num_return_sequences
        )
        bos_token_id = bos_token_id if bos_token_id is not None else self.config.generator.bos_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.config.generator.eos_token_id
        pad_token_id = pad_token_id if pad_token_id is not None else self.config.generator.pad_token_id
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        decoder_start_token_id = (
            decoder_start_token_id
            if decoder_start_token_id is not None
            else self.config.generator.decoder_start_token_id
        )
        remove_invalid_values = (
            remove_invalid_values if remove_invalid_values is not None else self.config.remove_invalid_values
        )

        # retrieve docs
        dialog_lengths = None
        if self.retriever is not None and context_input_ids is None:
            if self.config.scoring_func != "original":
                dpr_out = self.question_encoder(
                    input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict=True
                )
                combined_out = dpr_out.pooler_output

                ## Get mask for current turn input ids
                curr_turn_mask = torch.logical_xor(attention_mask, token_type_ids)
                current_turn_input_ids = input_ids * curr_turn_mask
                current_turn_only_out = self.question_encoder(
                    current_turn_input_ids, attention_mask=curr_turn_mask.long(), return_dict=True
                )
                current_turn_output = current_turn_only_out.pooler_output

                ## Split the dpr sequence output
                sequence_output = dpr_out.hidden_states[-1]
                attn_mask = self.get_attn_mask(input_ids)
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
                    seq_out_0.append(self.mean_pool(seq_out_masked_0))
                    seq_out_1.append(self.mean_pool(seq_out_masked_1))

                pooled_output_0 = torch.cat([seq.view(1, -1) for seq in seq_out_0], dim=0)
                pooled_output_1 = torch.cat([seq.view(1, -1) for seq in seq_out_1], dim=0)

                if self.config.scoring_func in ["reranking_original", "current_original"]:
                    current_out = current_turn_output
                else:
                    current_out = pooled_output_0

                out = self.retriever(
                    input_ids,
                    combined_out.cpu().detach().to(torch.float32).numpy(),
                    current_out.cpu().detach().to(torch.float32).numpy(),
                    pooled_output_1.cpu().detach().to(torch.float32).numpy(),
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    dialog_lengths=dialog_lengths,
                    domain=domain,
                    return_tensors="pt",
                )
            else:
                combined_out = self.question_encoder(input_ids, attention_mask=attention_mask)[0]
                out = self.retriever(
                    input_ids,
                    combined_out.cpu().detach().to(torch.float32).numpy(),
                    combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                    combined_out.cpu().detach().to(torch.float32).numpy(),  ## sending dummy
                    prefix=self.generator.config.prefix,
                    n_docs=n_docs,
                    dialog_lengths=dialog_lengths,
                    domain=domain,
                    return_tensors="pt",
                    bm25=self.bm25,
                )

            context_input_ids, context_attention_mask, retrieved_doc_embeds, retrieved_doc_scores = (
                out["context_input_ids"],
                out["context_attention_mask"],
                out["retrieved_doc_embeds"],
                out["doc_scores"],
            )

            # set to correct device
            retrieved_doc_embeds = retrieved_doc_embeds.to(combined_out)
            context_input_ids = context_input_ids.to(input_ids)
            context_attention_mask = context_attention_mask.to(input_ids)
            doc_scores = retrieved_doc_scores.to(combined_out)

            # compute doc_scores
            if self.config.scoring_func in [
                "reranking",
                "reranking2",
                "original",
                "reranking_original",
                "current_original",
                "current_pooled",
            ]:
                doc_scores = torch.bmm(combined_out.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)).squeeze(1)
            elif self.config.scoring_func in ["linear", "linear2", "linear3", "nonlinear"]:
                doc_scores_curr = torch.bmm(
                    pooled_output_0.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                ).squeeze(1)

                doc_scores_hist = torch.bmm(
                    pooled_output_1.unsqueeze(1), retrieved_doc_embeds.transpose(1, 2)
                ).squeeze(1)

                if self.config.scoring_func == "linear":
                    doc_scores = doc_scores_curr + doc_scores_hist
                elif self.config.scoring_func == "linear2":
                    doc_scores = doc_scores_curr + 0.5 * doc_scores_hist
                elif self.config.scoring_func == "linear3":
                    # TODO
                    doc_scores = doc_scores_curr + 0.5 * doc_scores_hist
                else:  # nonlinear
                    bsz = doc_scores_curr.shape[0]
                    doc_scores_curr_flattened = doc_scores_curr.flatten().unsqueeze(
                        1
                    )  # from (B, n_docs) to (Bxn_docs, 1)
                    doc_scores_hist_flattened = doc_scores_hist.flatten().unsqueeze(
                        1
                    )  # from (B, n_docs) to (Bxn_docs, 1)
                    scorer_inp = torch.cat(
                        [doc_scores_curr_flattened, doc_scores_hist_flattened], dim=1
                    )  # (Bxn_docs, 2)
                    scores = self.retriever.nn_scorer(scorer_inp)
                    doc_scores = scores.reshape((bsz, -1))

        assert (
            context_input_ids.shape[0] % n_docs
        ) == 0, f" The first dimension of `context_input_ids` should be a multiple of `n_docs`={n_docs}, but is {context_input_ids.shape[0]}."

        # batch_size
        batch_size = context_input_ids.shape[0] // n_docs

        encoder = self.rag.generator.get_encoder()
        encoder_outputs = encoder(input_ids=context_input_ids, attention_mask=context_attention_mask, return_dict=True)

        input_ids = torch.full(
            (batch_size * num_beams, 1),
            decoder_start_token_id,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )
        last_hidden_state = encoder_outputs["last_hidden_state"]

        def extend_enc_output(tensor, num_beams=None):
            # split into `batch_size`, `num_beams`, `num_docs`
            tensor = tensor[None, None, :].reshape((batch_size, 1, n_docs) + tensor.shape[1:])
            # repeat same last hidden states over `num_beams` dimension
            tensor = tensor.expand((batch_size, num_beams, n_docs) + tensor.shape[3:])
            # merge `batch_size`, `num_beams`, `num_docs` dims again
            return tensor.reshape((batch_size * num_beams * n_docs,) + tensor.shape[3:])

        # correctly extend last_hidden_state and attention mask
        context_attention_mask = extend_enc_output(context_attention_mask, num_beams=num_beams)
        encoder_outputs["last_hidden_state"] = extend_enc_output(last_hidden_state, num_beams=num_beams)

        doc_scores = doc_scores.repeat_interleave(num_beams, dim=0)

        # define start_len & additional parameters
        model_kwargs["doc_scores"] = doc_scores
        model_kwargs["encoder_outputs"] = encoder_outputs
        model_kwargs["attention_mask"] = context_attention_mask
        model_kwargs["n_docs"] = n_docs

        pre_processor = self._get_logits_processor(
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            encoder_no_repeat_ngram_size=encoder_no_repeat_ngram_size,
            encoder_input_ids=context_input_ids,
            bad_words_ids=bad_words_ids,
            min_length=min_length,
            max_length=max_length,
            eos_token_id=eos_token_id,
            forced_bos_token_id=forced_bos_token_id,
            forced_eos_token_id=forced_eos_token_id,
            prefix_allowed_tokens_fn=prefix_allowed_tokens_fn,
            num_beams=num_beams,
            num_beam_groups=num_beam_groups,
            diversity_penalty=diversity_penalty,
            remove_invalid_values=remove_invalid_values,
        )

        if num_beams == 1:
            if num_return_sequences > 1:
                raise ValueError(
                    f"num_return_sequences has to be 1, but is {num_return_sequences} when doing greedy search."
                )
            return self.greedy_search(
                input_ids,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        elif num_beams > 1:
            length_penalty = length_penalty if length_penalty is not None else self.config.length_penalty
            early_stopping = early_stopping if early_stopping is not None else self.config.early_stopping
            if num_return_sequences > num_beams:
                raise ValueError("`num_return_sequences` has to be smaller or equal to `num_beams`.")
            beam_scorer = BeamSearchScorer(
                batch_size=batch_size,
                num_beams=num_beams,
                device=self.device,
                length_penalty=length_penalty,
                do_early_stopping=early_stopping,
                num_beam_hyps_to_keep=num_return_sequences,
            )
            return self.beam_search(
                input_ids,
                beam_scorer,
                logits_processor=pre_processor,
                max_length=max_length,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id,
                **model_kwargs,
            )
        else:
            raise ValueError(f"`num_beams` has to be an integer strictly superior to 0 (≥ 1), but is {num_beams}")
