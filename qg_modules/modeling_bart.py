# coding=utf-8
# Copyright 2020 The Facebook AI Research Team Authors and The HuggingFace Inc. team.
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
"""PyTorch BART model, ported from the fairseq repo."""
import logging
import random
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .activations import ACT2FN
from .configuration_bart import BartConfig
from .file_utils import add_start_docstrings, add_start_docstrings_to_callable
from .modeling_utils import PreTrainedModel, create_position_ids_from_input_ids


logger = logging.getLogger(__name__)


BART_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bart-large": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large/pytorch_model.bin",
    "bart-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-mnli/pytorch_model.bin",
    "bart-large-cnn": "https://s3.amazonaws.com/models.huggingface.co/bert/facebook/bart-large-cnn/pytorch_model.bin",
}

BART_START_DOCSTRING = r"""

    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class. Use it as a regular PyTorch Module and
    refer to the PyTorch documentation for all matters related to general usage and behavior.

    Parameters:
        config (:class:`~transformers.BartConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.

"""
BART_GENERATION_EXAMPLE = r"""
    Examples::

        from transformers import BartTokenizer, BartForConditionalGeneration, BartConfig
        # see ``examples/summarization/bart/evaluate_cnn.py`` for a longer example
        model = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
        tokenizer = BartTokenizer.from_pretrained('bart-large-cnn')
        ARTICLE_TO_SUMMARIZE = "My friends are cool but they eat too many carbs."
        inputs = tokenizer.batch_encode_plus([ARTICLE_TO_SUMMARIZE], max_length=1024, return_tensors='pt')
        # Generate Summary
        summary_ids = model.generate(inputs['input_ids'], num_beams=4, max_length=5, early_stopping=True)
        print([tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in summary_ids])

"""

BART_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
               Indices of input sequence tokens in the vocabulary. Use BartTokenizer.encode to produce them.
            Padding will be ignored by default should you provide it.
            Indices can be obtained using :class:`transformers.BartTokenizer.encode(text)`.
        attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices in input_ids.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`, defaults to :obj:`None`):
            Provide for translation and summarization training. By default, the model will create this tensor by shifting the input_ids right, following the paper.
        decoder_attention_mask (:obj:`torch.Tensor` of shape :obj:`(batch_size, 1, tgt_seq_len, tgt_seq_len)`, `optional`, defaults to :obj:`None`):
            Default behavior: generate a tensor that ignores pad tokens and future tokens, as in the paper.
            If you want to change padding behavior, you should read :func:`~transformers.modeling_bart._prepare_decoder_inputs` and modify.
            See diagram 1 in the paper for more info on the default strategy
"""
LARGE_NEGATIVE = -1e8


def _prepare_bart_decoder_inputs(
    config, input_ids, decoder_input_ids=None, decoder_attn_mask=None, mask_dtype=None,
):
    """Prepare masks that ignore padding tokens in the decoder and a causal lm mask for the decoder if
    none are provided. This mimics the default behavior in fairseq. To override it pass in masks.
    Note: this is not called during generation
    """
    pad_token_id = config.pad_token_id
    need_causal_mask = not config.output_past
    if decoder_input_ids is None:
        decoder_input_ids = shift_tokens_right(input_ids, pad_token_id)
    bsz, tgt_len = decoder_input_ids.size()[:2]
    if decoder_attn_mask is None:
        decoder_padding_mask = make_padding_mask(decoder_input_ids, pad_token_id)
        if need_causal_mask:
            causal_lm_mask = torch.triu(fill_with_neg_inf(torch.zeros(tgt_len, tgt_len)), 1)
        else:
            causal_lm_mask = None
        new_shape = (bsz, tgt_len, tgt_len)
        # make it broadcastable so can just be added to the attention coefficients
        decoder_attn_mask = _combine_masks(decoder_padding_mask, causal_lm_mask, new_shape).to(device=input_ids.device)
        if mask_dtype is not None:
            decoder_attn_mask = decoder_attn_mask.to(mask_dtype)
    assert decoder_attn_mask is None or decoder_attn_mask.shape == (bsz, 1, tgt_len, tgt_len)
    return decoder_input_ids, decoder_attn_mask


class PretrainedBartModel(PreTrainedModel):
    config_class = BartConfig
    base_model_prefix = "model"
    pretrained_model_archive_map = BART_PRETRAINED_MODEL_ARCHIVE_MAP

    def _init_weights(self, module):
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        decoder_input_ids, decoder_attn_mask = _prepare_bart_decoder_inputs(self.config, input_ids,)
        dummy_inputs = {
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
            "decoder_attention_mask": decoder_attn_mask,
        }
        return dummy_inputs


def _make_linear_from_emb(emb):
    vocab_size, emb_size = emb.weight.shape
    lin_layer = nn.Linear(vocab_size, emb_size, bias=False)
    lin_layer.weight.data = emb.weight.data
    return lin_layer


# Helper Functions, mostly for making masks
def _check_shapes(shape_1, shape2):
    if shape_1 != shape2:
        raise AssertionError("shape mismatch: {} != {}".format(shape_1, shape2))


def _combine_masks(key_padding_mask, causal_lm_mask, targ_size):
    """Make one mask of shape (bsz, 1, tgt_len, src_len) """
    a = torch.zeros(targ_size)  # targ_size is(bsz, tgt_len, src_len)
    b = torch.zeros(targ_size)
    if key_padding_mask is not None:  # (bsz, tgt_len) -> targ_size
        _check_shapes(key_padding_mask.shape, targ_size[:2])
        reshaped = key_padding_mask.unsqueeze(2).expand(*targ_size)
        a[reshaped] = LARGE_NEGATIVE

    if causal_lm_mask is not None:  # (tgt_len, src_len) -> targ_size
        _check_shapes(causal_lm_mask.shape, targ_size[-2:])
        b = causal_lm_mask.unsqueeze(0).expand(*targ_size)
    return (a + b).unsqueeze(1).clamp(LARGE_NEGATIVE,)


def shift_tokens_right(input_ids, pad_token_id):
    """Shift input ids one token to the right, and wrap the last non pad token (usually <eos>)."""
    prev_output_tokens = input_ids.clone()
    index_of_eos = (input_ids.ne(pad_token_id).sum(dim=1) - 1).unsqueeze(-1)
    prev_output_tokens[:, 0] = input_ids.gather(1, index_of_eos).squeeze()
    prev_output_tokens[:, 1:] = input_ids[:, :-1]
    return prev_output_tokens


def make_padding_mask(input_ids, padding_idx=1):
    """True for pad tokens"""
    padding_mask = input_ids.eq(padding_idx)
    if not padding_mask.any():
        padding_mask = None
    return padding_mask


# Helper Modules


class EncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.output_attentions = config.output_attentions
        self.self_attn = SelfAttention(
            self.embed_dim, config.encoder_attention_heads, dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(self, x, encoder_padding_mask):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, src_len)` where padding elements are indicated by ``1``.
            for t_tgt, t_src is excluded (or masked out), =0 means it is
            included in attention

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        residual = x
        x, attn_weights = self.self_attn(query=x, key=x, key_padding_mask=encoder_padding_mask,)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn_weights


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer
    is a :class:`EncoderLayer`.

    Args:
        config: BartConfig
    """

    def __init__(self, config: BartConfig, embed_tokens):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = config.max_position_embeddings

        self.embed_tokens = embed_tokens

        self.embed_positions = LearnedPositionalEmbedding(config.max_position_embeddings, embed_dim, self.padding_idx,)
        self.layers = nn.ModuleList([EncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = LayerNorm(embed_dim)

    def forward(
        self, input_ids, attention_mask=None,
    ):
        """
        Args:
            input_ids (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            attention_mask (torch.LongTensor): indicating which indices are padding tokens.
        Returns:
            Tuple comprised of:
                - **x** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *self.output_hidden_states:* is True.
                - **all_attentions** (List[Tensor]): Attention weights for each layer.
                During training might not be of length n_layers because of layer dropout.
        """
        # check attention mask and invert
        if attention_mask is not None:
            assert attention_mask.dim() == 2
            attention_mask = attention_mask.eq(0)

        inputs_embeds = self.embed_tokens(input_ids)
        embed_pos = self.embed_positions(input_ids)
        x = inputs_embeds + embed_pos
        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        encoder_states, all_attentions = [], []
        for encoder_layer in self.layers:
            if self.output_hidden_states:
                encoder_states.append(x)
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                attn = None
            else:
                x, attn = encoder_layer(x, attention_mask)

            if self.output_attentions:
                all_attentions.append(attn)

        if self.output_hidden_states:
            encoder_states.append(x)

        encoder_states = [hidden_state.transpose(0, 1) for hidden_state in encoder_states]
        return x, encoder_states, all_attentions


class DecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = SelfAttention(
            embed_dim=self.embed_dim, num_heads=config.decoder_attention_heads, dropout=config.attention_dropout,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.encoder_attn = SelfAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            encoder_decoder_attention=True,
        )
        self.encoder_attn_layer_norm = LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = LayerNorm(self.embed_dim)

    def forward(
        self, x, encoder_hidden_states, encoder_attn_mask=None, layer_state=None, attention_mask=None,
    ):
        residual = x

        if layer_state is None:
            layer_state = {}
        # next line mutates layer state
        x, self_attn_weights = self.self_attn(query=x, key=x, layer_state=layer_state, attn_mask=attention_mask,)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)
        residual = x
        assert self.encoder_attn.cache_key != self.self_attn.cache_key

        x, encoder_attn_weights = self.encoder_attn(
            query=x,
            key=encoder_hidden_states,
            key_padding_mask=encoder_attn_mask,
            layer_state=layer_state,  # mutates layer state
        )
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x

        x = self.encoder_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return (
            x,
            self_attn_weights,
            layer_state,
        )  # just self_attn weights for now, following t5, layer_state = cache for decoding


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer
    is a :class:`DecoderLayer`.
    Args:
        config: BartConfig
        embed_tokens (torch.nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, embed_tokens: nn.Embedding):
        super().__init__()
        self.output_past = config.output_past
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = embed_tokens.padding_idx
        self.max_target_positions = config.max_position_embeddings
        self.embed_tokens = embed_tokens
        self.embed_positions = LearnedPositionalEmbedding(
            config.max_position_embeddings, config.d_model, self.padding_idx,
        )
        self.layers = nn.ModuleList(
            [DecoderLayer(config) for _ in range(config.decoder_layers)]
        )  # type: List[DecoderLayer]
        self.layernorm_embedding = LayerNorm(config.d_model)

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        encoder_padding_mask,
        combined_mask,
        decoder_cached_states=None,
        generation_mode=False,
        **unused
    ):
        """
        Includes several features from "Jointly Learning to Align and
        Translate with Transformer Models" (Garg et al., EMNLP 2019).

        Args:
            input_ids (LongTensor): previous decoder outputs of shape
                `(batch, tgt_len)`, for teacher forcing
            encoder_hidden_states: output from the encoder, used for
                encoder-side attention
            encoder_padding_mask: for ignoring pad tokens
            decoder_cached_states (dict or None): dictionary used for storing state during generation

        Returns:
            tuple:
                - the decoder's features of shape `(batch, tgt_len, embed_dim)`
                - hidden states
                - attentions
        """
        # check attention mask and invert
        if encoder_padding_mask is not None:
            assert encoder_padding_mask.dim() == 2
            encoder_padding_mask = encoder_padding_mask.eq(0)

        # embed positions
        positions = self.embed_positions(input_ids, generation_mode=generation_mode)

        if generation_mode:
            input_ids = input_ids[:, -1:]
            positions = positions[:, -1:]  # happens after we embed them
            assert input_ids.ne(self.padding_idx).any()

        x = self.embed_tokens(input_ids)
        x += positions

        x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)  # (seq_len, BS, model_dim)
        # decoder layers
        all_hidden_states = ()
        all_self_attns = ()
        next_decoder_cache = []

        for i, decoder_layer in enumerate(self.layers):
            decoder_layer  # type: DecoderLayer
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            layer_state = decoder_cached_states[i] if decoder_cached_states is not None else None
            x, layer_self_attn, layer_past = decoder_layer(
                x, encoder_hidden_states, encoder_padding_mask, layer_state=layer_state, attention_mask=combined_mask,
            )

            if self.output_past:
                next_decoder_cache.append(layer_past.copy())
            if self.output_hidden_states:
                all_hidden_states += (x,)
            if self.output_attentions:
                all_self_attns += (layer_self_attn,)

        # Convert shapes from (seq_len, BS, model_dim) to (BS, seq_len, model_dim)
        all_hidden_states = [hidden_state.transpose(0, 1) for hidden_state in all_hidden_states]
        x = x.transpose(0, 1)

        if self.output_past:
            next_cache = ((encoder_hidden_states, encoder_padding_mask), next_decoder_cache)
        else:
            next_cache = None
        return x, next_cache, all_hidden_states, list(all_self_attns)


def _reorder_buffer(attn_cache, new_order):
    for k, input_buffer_k in attn_cache.items():
        if input_buffer_k is not None:
            attn_cache[k] = input_buffer_k.index_select(0, new_order)
    return attn_cache


class SelfAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim,
        num_heads,
        dropout=0.0,
        bias=True,
        encoder_decoder_attention=False,  # otherwise self_attention
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.encoder_decoder_attention = encoder_decoder_attention
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.cache_key = "encoder_decoder" if self.encoder_decoder_attention else "self"

    def _shape(self, tensor, dim_0, bsz):
        return tensor.contiguous().view(dim_0, bsz * self.num_heads, self.head_dim).transpose(0, 1)

    def forward(
        self,
        query,
        key: Optional[Tensor],
        key_padding_mask: Optional[Tensor] = None,
        layer_state: Optional[Dict[str, Optional[Tensor]]] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Input shape: Time(SeqLen) x Batch x Channel"""
        static_kv = self.encoder_decoder_attention  # type: bool
        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]
        # get here for encoder decoder cause of static_kv
        if layer_state is not None:  # reuse k,v and encoder_padding_mask
            saved_state = layer_state.get(self.cache_key, {})
            if "prev_key" in saved_state:
                # previous time steps are cached - no need to recompute key and value if they are static
                if static_kv:
                    key = None
        else:
            saved_state = None
            layer_state = {}

        q = self.q_proj(query) * self.scaling
        if static_kv:
            if key is None:
                k = v = None
            else:
                k = self.k_proj(key)
                v = self.v_proj(key)
        else:
            k = self.k_proj(query)
            v = self.v_proj(query)

        q = self._shape(q, tgt_len, bsz)
        if k is not None:
            k = self._shape(k, -1, bsz)
        if v is not None:
            v = self._shape(v, -1, bsz)

        if saved_state is not None:
            k, v, key_padding_mask = self._use_saved_state(k, v, saved_state, key_padding_mask, static_kv, bsz)

        # Update cache
        layer_state[self.cache_key] = {
            "prev_key": k.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_value": v.view(bsz, self.num_heads, -1, self.head_dim),
            "prev_key_padding_mask": key_padding_mask if not static_kv else None,
        }

        assert k is not None
        src_len = k.size(1)
        attn_weights = torch.bmm(q, k.transpose(1, 2))
        assert attn_weights.size() == (bsz * self.num_heads, tgt_len, src_len)

        if attn_mask is not None:
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attn_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        # This is part of a workaround to get around fork/join parallelism not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.dim() == 0:
            key_padding_mask = None
        assert key_padding_mask is None or key_padding_mask.size()[:2] == (bsz, src_len,)

        if key_padding_mask is not None:  # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            reshaped = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(reshaped, float("-inf"))
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_probs = F.dropout(attn_weights, p=self.dropout, training=self.training,)

        assert v is not None
        attn_output = torch.bmm(attn_probs, v)
        assert attn_output.size() == (bsz * self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
        attn_output = self.out_proj(attn_output)
        attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
        return attn_output, attn_weights

    def _use_saved_state(self, k, v, saved_state, key_padding_mask, static_kv, bsz):
        # saved states are stored with shape (bsz, num_heads, seq_len, head_dim)
        if "prev_key" in saved_state:
            _prev_key = saved_state["prev_key"]
            assert _prev_key is not None
            prev_key = _prev_key.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                k = prev_key
            else:
                assert k is not None
                k = torch.cat([prev_key, k], dim=1)
        if "prev_value" in saved_state:
            _prev_value = saved_state["prev_value"]
            assert _prev_value is not None
            prev_value = _prev_value.view(bsz * self.num_heads, -1, self.head_dim)
            if static_kv:
                v = prev_value
            else:
                assert v is not None
                v = torch.cat([prev_value, v], dim=1)
        assert k is not None and v is not None
        prev_key_padding_mask = saved_state.get("prev_key_padding_mask", None)  # type: Optional[Tensor]
        key_padding_mask = self._cat_prev_key_padding_mask(
            key_padding_mask, prev_key_padding_mask, bsz, k.size(1), static_kv
        )
        return k, v, key_padding_mask

    @staticmethod
    def _cat_prev_key_padding_mask(
        key_padding_mask: Optional[Tensor],
        prev_key_padding_mask: Optional[Tensor],
        batch_size: int,
        src_len: int,
        static_kv: bool,
    ) -> Optional[Tensor]:
        # saved key padding masks have shape (bsz, seq_len)
        if prev_key_padding_mask is not None:
            if static_kv:
                new_key_padding_mask = prev_key_padding_mask
            else:
                new_key_padding_mask = torch.cat([prev_key_padding_mask, key_padding_mask], dim=1)

        elif key_padding_mask is not None:
            filler = torch.zeros(
                batch_size,
                src_len - key_padding_mask.size(1),
                dtype=key_padding_mask.dtype,
                device=key_padding_mask.device,
            )
            new_key_padding_mask = torch.cat([filler, key_padding_mask], dim=1)
        else:
            new_key_padding_mask = prev_key_padding_mask
        return new_key_padding_mask


class BartClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    # This can trivially be shared with RobertaClassificationHead

    def __init__(
        self, input_dim, inner_dim, num_classes, pooler_dropout,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class LearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    Padding ids are ignored by either offsetting based on padding_idx
    or by setting padding_idx to None and ensuring that the appropriate
    position ids are passed to the forward function.
    """

    def __init__(
        self, num_embeddings: int, embedding_dim: int, padding_idx: int,
    ):
        # if padding_idx is specified then offset the embedding ids by
        # this index and adjust num_embeddings appropriately
        assert padding_idx is not None
        num_embeddings += padding_idx + 1  # WHY?
        super().__init__(num_embeddings, embedding_dim, padding_idx=padding_idx)

    def forward(self, input, generation_mode=False):
        """Input is expected to be of size [bsz x seqlen]."""
        if generation_mode:  # the position is our current step in the decoded sequence
            pos = int(self.padding_idx + input.size(1))
            positions = input.data.new(1, 1).fill_(pos)
        else:
            positions = create_position_ids_from_input_ids(input, self.padding_idx)
        return super().forward(positions)


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm

            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def fill_with_neg_inf(t):
    """FP16-compatible function that fills a input_ids with -inf."""
    return t.float().fill_(float("-inf")).type_as(t)


def _filter_out_falsey_values(tup) -> Tuple:
    """Remove entries that are None or [] from an iterable."""
    return tuple(x for x in tup if isinstance(x, torch.Tensor) or x)


# Public API


@add_start_docstrings(
    "The bare BART Model outputting raw hidden-states without any specific head on top.", BART_START_DOCSTRING,
)
class BartModel(PretrainedBartModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)

        self.init_weights()

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        decoder_input_ids=None,
        encoder_outputs=None,  # type: Tuple
        decoder_attention_mask=None,
        decoder_cached_states=None,
        generation_mode=False,
    ):

        # make masks if user doesn't supply
        if not generation_mode:
            decoder_input_ids, decoder_attention_mask = _prepare_bart_decoder_inputs(
                self.config,
                input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attn_mask=decoder_attention_mask,
                mask_dtype=self.shared.weight.dtype,
            )
        assert decoder_input_ids is not None
        if encoder_outputs is None:
            encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        assert isinstance(encoder_outputs, tuple)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            decoder_input_ids,
            encoder_outputs[0],
            attention_mask,
            decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            generation_mode=generation_mode,
        )
        # Attention and hidden_states will be [] or None if they aren't needed
        decoder_outputs = _filter_out_falsey_values(decoder_outputs)  # type: tuple
        assert isinstance(decoder_outputs[0], torch.Tensor)
        encoder_outputs = _filter_out_falsey_values(encoder_outputs)  # type: tuple
        return decoder_outputs + encoder_outputs

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value

    def get_output_embeddings(self):
        return _make_linear_from_emb(self.shared)  # make it on the fly


@add_start_docstrings(
    "The BART Model with a language modeling head. Can be used for summarization.",
    BART_START_DOCSTRING + BART_GENERATION_EXAMPLE,
)
class BartForConditionalGeneration(PretrainedBartModel):
    base_model_prefix = "model"

    def __init__(self, config: BartConfig):
        super().__init__(config)
        # if base_model is None:
        base_model = BartModel(config)
        self.model = base_model
        self.lm_head = _make_linear_from_emb(self.model.shared)

    def tie_weights(self):
        pass  # hack to prevent changing lm_head.out_features. The input and output embeddings are still the same.

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_cached_states=None,
        lm_labels=None,
        generation_mode=False,
        **unused
    ):
        r"""
        masked_lm_labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the masked language modeling loss.
            Indices should either be in ``[0, ..., config.vocab_size]`` or -100 (see ``input_ids`` docstring).
            Tokens with indices set to ``-100`` are ignored (masked), the loss is only computed for the tokens
            with labels
            in ``[0, ..., config.vocab_size]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.RobertaConfig`) and inputs:
        masked_lm_loss (`optional`, returned when ``masked_lm_labels`` is provided) ``torch.FloatTensor`` of shape ``(1,)``:
            Masked language modeling loss.
        prediction_scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`)
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

            # Mask filling only works for bart-large
            from transformers import BartTokenizer, BartForConditionalGeneration
            tokenizer = BartTokenizer.from_pretrained('bart-large')
            TXT = "My friends are <mask> but they eat too many carbs."
            model = BartForConditionalGeneration.from_pretrained('bart-large')
            input_ids = tokenizer.batch_encode_plus([TXT], return_tensors='pt')['input_ids']
            logits = model(input_ids)[0]
            masked_index = (input_ids[0] == tokenizer.mask_token_id).nonzero().item()
            probs = logits[0, masked_index].softmax(dim=0)
            values, predictions = probs.topk(5)
            tokenizer.decode(predictions).split()
            # ['good', 'great', 'all', 'really', 'very']
        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            decoder_cached_states=decoder_cached_states,
            generation_mode=generation_mode,
        )
        lm_logits = self.lm_head(outputs[0])
        outputs = (lm_logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if lm_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            # TODO(SS): do we need to ignore pad tokens in lm_labels?
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs

    def prepare_inputs_for_generation(self, decoder_input_ids, past, attention_mask, **kwargs):
        assert past is not None, "past has to be defined for encoder_outputs"

        # first step, decoder_cached_states are empty
        if not past[1]:
            encoder_outputs, decoder_cached_states = past, None
        else:
            encoder_outputs, decoder_cached_states = past

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "decoder_cached_states": decoder_cached_states,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "generation_mode": True,
        }

    def prepare_scores_for_generation(self, scores, cur_len, max_length):
        if cur_len == 1:
            self._force_token_ids_generation(scores, self.config.bos_token_id)
        if cur_len == max_length - 1 and self.config.eos_token_id is not None:
            self._force_token_ids_generation(scores, self.config.eos_token_id)
        return scores

    @staticmethod
    def _reorder_cache(past, beam_idx):
        ((enc_out, enc_mask), decoder_cached_states) = past
        reordered_past = []
        for layer_past in decoder_cached_states:
            # get the correct batch idx from decoder layer's batch dim for cross and self-attn
            layer_past_new = {
                attn_key: _reorder_buffer(attn_cache, beam_idx) for attn_key, attn_cache in layer_past.items()
            }
            # reordered_layer_past = [layer_past[:, i].unsqueeze(1).clone().detach() for i in beam_idx]
            # reordered_layer_past = torch.cat(reordered_layer_past, dim=1)
            reordered_past.append(layer_past_new)
        new_enc_out = enc_out if enc_out is None else enc_out.index_select(1, beam_idx)
        new_enc_mask = enc_mask if enc_mask is None else enc_mask.index_select(0, beam_idx)

        past = ((new_enc_out, new_enc_mask), reordered_past)
        return past

    def get_encoder(self):
        return self.model.encoder

    def get_output_embeddings(self):
        return self.lm_head


@add_start_docstrings(
    """Bart model with a sequence classification/head on top (a linear layer on top of the pooled output) e.g. for GLUE tasks. """,
    BART_START_DOCSTRING,
)
class BartForSequenceClassification(PretrainedBartModel):
    def __init__(self, config: BartConfig, **kwargs):
        super().__init__(config, **kwargs)
        self.model = BartModel(config)
        self.classification_head = BartClassificationHead(
            config.d_model, config.d_model, config.num_labels, config.classif_dropout,
        )
        self.model._init_weights(self.classification_head.dense)
        self.model._init_weights(self.classification_head.out_proj)

    @add_start_docstrings_to_callable(BART_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids,
        attention_mask=None,
        encoder_outputs=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`, defaults to :obj:`None`):
            Labels for computing the sequence classification/regression loss.
            Indices should be in :obj:`[0, ..., config.num_labels - 1]`.
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BartConfig`) and inputs:
            loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when :obj:`label` is provided):
                Classification  loss (cross entropy)
            logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, config.num_labels)`):
                Classification (or regression if config.num_labels==1) scores (before SoftMax).
            hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
                Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
                of shape :obj:`(batch_size, sequence_length, hidden_size)`.
                Hidden-states of the model at the output of each layer plus the initial embedding outputs.
            attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
                Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.
                Attentions weights after the attention softmax, used to compute the weighted average in the
                self-attention
                heads.

    Examples::

        from transformers import BartTokenizer, BartForSequenceClassification
        import torch

        tokenizer = BartTokenizer.from_pretrained('bart-large')
        model = BartForSequenceClassification.from_pretrained('bart-large')
        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute",
        add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        labels = torch.tensor([1]).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids, labels=labels)
        loss, logits = outputs[:2]

        """
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            encoder_outputs=encoder_outputs,
        )
        x = outputs[0]  # last hidden state
        eos_mask = input_ids.eq(self.config.eos_token_id)
        if len(torch.unique(eos_mask.sum(1))) > 1:
            raise ValueError("All examples must have the same number of <eos> tokens.")
        sentence_representation = x[eos_mask, :].view(x.size(0), -1, x.size(-1))[:, -1, :]
        logits = self.classification_head(sentence_representation)
        # Prepend logits
        outputs = (logits,) + outputs[1:]  # Add hidden states and attention if they are here
        if labels is not None:  # prepend loss to output,
            loss = F.cross_entropy(logits.view(-1, self.config.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs
class BartForQuestionGeneration(PretrainedBartModel):
    def __init__(self, config):
        super(BartForQuestionGeneration, self).__init__(config)
        self.model = BartModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.config = config
        #self.cls = BertPreTrainingHeads(config, self.model.embeddings.word_embeddings.weight)
        self.qgen_outputs = nn.Linear(config.hidden_size, config.vocab_size)

        self.apply(self.init_bert_weights)
        self.MASKID = self.config.decoder_start_token_id
        self.vocab_size = config.vocab_size
        self.iscuda = True
        self.attentionsquare = nn.Linear(config.hidden_size, config.num_attention_heads * config.num_hidden_layers ) # config.n_heads * config.n_layers
        self.attentionsquare_lateral = nn.Linear(config.hidden_size, config.num_attention_heads * config.num_hidden_layers ) # config.n_heads * config.n_layers
        self.attention_over_attention = nn.Linear(config.hidden_size, 3 ) # 
        self.attention_over_attention2 = nn.Linear(config.hidden_size, 2 ) # 
        self.linear_copy_probs = nn.Linear(config.hidden_size, config.hidden_size ) # config.n_heads * config.n_layers

        self.p_value = nn.Linear(config.hidden_size, 1)

    def get_copy_probs_lateral(self,sequence_output, self_attentions,src_seq,hop=2): # This is 2-D Attention

        # Use sequence_output to choose the self_attentions wisely 
        attention_square = torch.softmax(self.attentionsquare(sequence_output),dim=-1) # batch_size x seq_size x (12*12)
        attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)

        self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size
        self_attentions = self_attentions.transpose(1,2) # batch_size x seq_size x (12*12) x seq_size

        # The below does weighted average over all the attentions of all the layers and heads
        #print(attention_square.size(), self_attentions.size())

        '''
        attention_square_lateral = self.attentionsquare_lateral(sequence_output) # batch_size x seq_size x (12*12)
        attention_square_lateral = attention_square_lateral.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)
        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 
        attention_lateral = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 
        '''
        # combine 144 attention matrices wisely:
        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 

        if hop == 3: # x*a + y* a^2 + z*a^3
          attention_square = torch.bmm(attention, attention)
          attention_cube = torch.bmm(attention, attention_square)

          #obtain values x, y, z as mention above
          attention_scores = torch.softmax(self.attention_over_attention(sequence_output),dim=-1)

          total_attention = torch.cat((attention.unsqueeze(1),attention_square.unsqueeze(1)),1)
          total_attention = torch.cat((total_attention,attention_cube.unsqueeze(1)),1)

          #multiply different hop-attentions with different probs
          total_attention = total_attention* attention_scores.transpose(1,2).unsqueeze(-1)
          total_attention = total_attention.sum(1) # batch x seq x seq
        elif hop == 2:
          attention_square = torch.bmm(attention, attention)

          #obtain values x, y, z as mention above
          attention_scores = torch.softmax(self.attention_over_attention2(sequence_output),dim=-1) #batch_size x seq x 2

          total_attention = torch.cat((attention.unsqueeze(2),attention_square.unsqueeze(2)),2) # b x s x 2 x s

          #multiply different hop-attentions with different probs
          total_attention = total_attention * attention_scores.unsqueeze(-1)
          total_attention = total_attention.sum(2)
          #total_attention = attention_scores.matmul(total_attention) # batchxseqx2 * batchx 2 x seq x seq 
          '''
          total_attention = (attention_square + attention) / 2 # average of 1-hop and 2-hop
          '''
        elif hop == 21:
          # combine wisely 144 self-attention in two different ways and use one for hop-1 and another for hop-2
          attention_square_lateral = torch.softmax(self.attentionsquare_lateral(sequence_output),dim=-1) # batch_size x seq_size x (12*12)
          attention_square_lateral = attention_square_lateral.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)
          attention_lateral = torch.matmul(attention_square_lateral,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 

          #two_hop_attention = torch.bmm(attention_lateral, attention)  #attention_square bxsxs
          two_hop_attention = torch.bmm(attention_lateral, attention_lateral)  #attention_square bxsxs

          attention_scores = torch.softmax(self.attention_over_attention2(sequence_output),dim=-1) #bxsx2 
          total_attention = torch.cat((attention.unsqueeze(2),two_hop_attention.unsqueeze(2)),2) # b x s x 2 x s
          total_attention = total_attention * attention_scores.unsqueeze(-1) # weighted average of 1-hop and 2-hop attentions
          total_attention = total_attention.sum(2)

        # attention value doesn't sum up to 1? why?
        # src_seq has the source vocab ids
        copy_probs=torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size)

        if self.iscuda==True:
          copy_probs = copy_probs.cuda()

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(total_attention, copy_probs) # copy scores

        #print(copy_probs.sum(-1))
        #input("copy probs")

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs
    def get_copy_probs(self,sequence_output, self_attentions,src_seq,cls_att=0,answer_ids=None,maxout_pointer=False,para_mask=None):

        # Use sequence_output to choose the self_attentions wisely 
        attention_square = torch.softmax(self.attentionsquare(sequence_output),dim=-1) # batch_size x seq_size x (12*12)
        attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)
        self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size
        self_attentions = self_attentions.transpose(1,2) # batch_size x seq_size x (12*12) x seq_size


        # The below does weighted average over all the attentions of all the layers and heads
        #print(attention_square.size(), self_attentions.size())

        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 

        if para_mask is not None:
           attention = attention * para_mask

        if cls_att == 1:
            attention = (attention + attention[:,0,:].unsqueeze(1)) / 2 # b x 1 x seq
        if cls_att == 2:
            answer_ids = answer_ids.float()
            answer_att = torch.bmm(answer_ids.unsqueeze(1), attention) # b x 1 x seq
            attention = (attention + answer_att) / (1+answer_ids.sum(-1).unsqueeze(-1).unsqueeze(-1)) # b x 1 x seq

        # attention value doesn't sum up to 1? why?
        # src_seq has the source vocab ids
        copy_probs=torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size)

        if self.iscuda==True:
          copy_probs = copy_probs.cuda()

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        #copy_probs = torch.bmm(attention, copy_probs) # copy scores # [b x seq x seq] * [b x seq x vocab]
        if maxout_pointer:
          #copy_probs = copy_probs.unsqueeze(1).expand(-1,copy_probs.size(1), -1, -1) # [ b x (s) x s x v] (_) is repeatition
          #copy_probs = copy_probs.unsqueeze(1) * attention.unsqueeze(-1) # [b x 1 x s x V] * [b x s x s x 1] = b x s x s x V
          #attention = attention.unsqueeze(-1).expand(-1,-1,-1,self.config.vocab_size) # b x s x s x V
          #attention = attention.masked_fill(1-copy_probs.byte(), 0)
          repeated_words = torch.bmm(copy_probs, copy_probs.transpose(1,2)) # b x seq x seq
          '''
          max_copy_attn_seqbyseq = repeated_words.unsqueeze(2) * attention.unsqueeze(3) # (b x seq x 1 x seq) * (b x seq x seq x 1) -- sequence by sequence
          max_copy_attn, _ = max_copy_attn_seqbyseq.max(-1)
          max_copy_attn = max_copy_attn / repeated_words.sum(-1).unsqueeze(2) # Distribute the maximum prob amongst all the instances 
          copy_probs = torch.bmm(max_copy_attn, copy_probs) # max_out_pointer probabilities
          '''
          max_copy_attn_seqbyseq = repeated_words.unsqueeze(1) * attention.unsqueeze(2).expand(-1,-1,attention.size(1),-1) # (b x 1 x seq x seq) * (b x seq x 1 x seq ) -- sequence by sequence
          max_copy_attn_seqbyseq = max_copy_attn_seqbyseq / repeated_words.sum(-1).unsqueeze(1).unsqueeze(1) #  divided by a tensor of size (b x 1 x 1 x seq) - Average the max value, 
                                                                                                             #  so that it can be directly multiplied
          max_copy_attn, _ = max_copy_attn_seqbyseq.max(-1)
          copy_probs = torch.bmm(max_copy_attn, copy_probs) # max_out_pointer probabilities : note they don't addd-up to 1
          copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # normalize? 

        else:
          copy_probs = torch.bmm(attention, copy_probs) # copy scores # [b x seq x seq] * [b x seq x vocab]


        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def get_copy_probs_sum(self,sequence_output, self_attentions,src_seq):

        # Use sequence_output to choose the self_attentions wisely 
        #attention_square = self.attentionsquare(sequence_output) # batch_size x seq_size x (12*12)
        #attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)
        self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size
        self_attentions = self_attentions.transpose(1,2) # batch_size x seq_size x (12*12) x seq_size

        # The below does weighted average over all the attentions of all the layers and heads
        #print(attention_square.size(), self_attentions.size())

        #attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 
        attention = self_attentions.mean(2) # batch_size x seq_size x seq_size 

        # attention value doesn't sum up to 1? why?
        # src_seq has the source vocab ids
        copy_probs=torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size)

        if self.iscuda==True:
          copy_probs = copy_probs.cuda()

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(attention, copy_probs) # copy scores

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire

        return copy_probs

    def get_copy_probs2(self,sequence_output, para_mask,src_seq, input_mask):

        # Use sequence_output to choose the self_attentions wisely 
        transformed_sequence_output = self.linear_copy_probs(sequence_output)
        affinity =  torch.bmm(transformed_sequence_output, sequence_output.transpose(1,2))  
        mask = (1-para_mask) * (-10000)

        #print("para_mask, breadth, height", para_mask[0].sum(-1), para_mask[0].sum(-2))
        #print((mask[0] == 0).sum(-1), (mask[0] == 0).sum(-2), affinity.size())
        #print((input_mask[0][0] == 0).sum())
        #print(src_seq[0])
        #input("check mask")

        affinity = affinity + mask
        probs = nn.Softmax(dim=-1)(affinity)

        # src_seq has the source vocab ids
        src_to_vocab_map=torch.zeros(src_seq.size(0), para_mask.size(2), self.config.vocab_size) #batch_size x seq_size x vocab_size


        if self.iscuda==True:
          src_to_vocab_map = src_to_vocab_map.cuda()

        src_to_vocab_map.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        #print(probs)
        copy_probs = torch.bmm(probs, src_to_vocab_map) # copy scores
        #print(copy_probs.sum(-1))
        #input("copy probs sum -1")

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,decoder_mask=True,evaluate=False, copymethod=1):

        #print("Forward for Question Generation")
        # Get Decoder Mask for Question Part
        #attention_mask = attention_mask.unsqueeze(1).expand(-1,attention_mask.size(-1),-1) # Use batch_size x seq_length x seq_length instead of batch_size x seq_length

        #question_mask_ids = token_type_ids.unsqueeze(-1).float()
        question_mask_ids = (token_type_ids == 1).unsqueeze(-1).float()
        input_mask_ids = (input_ids!=0).unsqueeze(-1).float()

        #Mask begin of question
        boq_mask = (input_ids != self.MASKID).unsqueeze(-1).float()
        question_mask_ids = question_mask_ids * boq_mask
        #torch.save(question_mask_ids,"question_mask_ids")
        question_mask = torch.bmm(question_mask_ids, question_mask_ids.transpose(1,2)) * torch.tril(torch.ones(question_mask_ids.size(1),question_mask_ids.size(1))).unsqueeze(0).cuda() 

        # For Question Generation use below line
        prediction_ids = (input_ids.unsqueeze(-1) * question_mask_ids.long()).squeeze(-1)
        # For simple LM use below line
        #prediction_ids = input_ids
        prediction_ids[:,0]=0 #self.PADID
        prediction_ids = torch.cat((prediction_ids[:,1:],prediction_ids[:,0:1]),-1) # batch_size x seq_length

        '''
        for b in range(prediction_ids.size(0)):
         print("question:")
         print("#target "," ".join(tokenizer.convert_ids_to_tokens([ prediction_ids[b][i].item() for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ])))
         print("#source "," ".join(tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ])))
         print("#ids"," ".join([ str(prediction_ids[b][i].item()) for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ]))
         input("source")
        '''

        #print
        para_mask = torch.bmm((1-question_mask_ids)*input_mask_ids, ((1-question_mask_ids)*input_mask_ids).transpose(1,2))
        question2para_mask = torch.bmm(input_mask_ids, ((1-question_mask_ids)*input_mask_ids).transpose(1,2))

        '''
        para_mask = para_mask.transpose(1,2)
        extra_mask = question_mask + para_mask
        extra_mask = extra_mask.long()
        '''
        '''
        for b in range(input_ids.size(0)):
         print("#para "," ".join(tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(input_ids[b].size(0)) if (input_mask_ids*(1-question_mask_ids))[b][i].item() ==1 ])))
         print("#source "," ".join(tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ])))
         print("para_mask, breadth, height", para_mask[b].sum(-1), para_mask[b].sum(-2))
        '''

        extra_mask = torch.bmm(input_mask_ids, input_mask_ids.transpose(1,2))
        extra_mask = extra_mask * torch.tril(torch.zeros_like(extra_mask[0]) + 1 )

        # The difference between LM and Question Generation is the below 2 lines of mask
        extra_mask = extra_mask + para_mask
        extra_mask = (extra_mask != 0).long()

        #extra_mask = attention_mask # lets check if without masking, prediction is good

        #print(extra_mask[0])
        #torch.save(extra_mask[0],"extra_mask_0")
        #input("extra mask")

        # attention_mask should not include future words for Question
        # token_typ_ids should contain Answer phrase

        answer_phrase_ids = torch.zeros_like(input_ids)
        '''
        '''
        #print(start_positions, end_positions)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,start_positions.unsqueeze(-1),1)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,(end_positions+1).unsqueeze(-1),-1)
        answer_phrase_ids = answer_phrase_ids.float().matmul(torch.triu(torch.ones(answer_phrase_ids.size(1), answer_phrase_ids.size(1)) ).cuda() ).long() # segment_id 1 for answer phrase

        # For Question Generation
        sequence_output, pooled_output, attns = self.model(input_ids, token_type_ids + answer_phrase_ids, extra_mask, output_all_encoded_layers=False,decoder_mask=decoder_mask,attn=True)


        # For LM
        #sequence_output, pooled_output, attns = self.model(input_ids, token_type_ids, extra_mask, output_all_encoded_layers=False,decoder_mask=decoder_mask,attn=True)
          

        #print(attns[0].size(),len(attns))
        #input("attn")

        '''
        logits = self.qa_outputs(sequence_output) 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        '''

        if copymethod > 0:
          p = torch.sigmoid(self.p_value(sequence_output))
          qgen_logits = p * torch.softmax(self.qgen_outputs(sequence_output),dim=-1)
          #qgen_logits = p * self.qgen_outputs(sequence_output)
          #print(qgen_logits[:,:,0:10])
          #input("qgen_logits look like this")

          if copymethod == 1:
            #Use a, obtain a by combining (head*layer) self-attentions
            copy_qgen_logits = (1-p) * self.get_copy_probs(sequence_output,attns,input_ids)
          elif copymethod == 2:
            # obtain a by linear transformation
            copy_qgen_logits = (1-p) * self.get_copy_probs2(sequence_output,question2para_mask,token_type_ids,question2para_mask) # Just create one attention layer on top of the encoded input
          elif copymethod == 21:
            # obtain a by linear transformation but from last 4 layers
            
            copy_qgen_logits = (1-p) * self.get_copy_probs2(sequence_output,question2para_mask,token_type_ids,question2para_mask) # Just create one attention layer on top of the encoded input
          elif copymethod == 3:
            #Use a^2
            copy_qgen_logits = (1-p) * self.get_copy_probs_lateral(sequence_output,attns,input_ids,hop=2) 
          elif copymethod == 4:
            #Use a,a^2,a^3
            copy_qgen_logits = (1-p) * self.get_copy_probs_lateral(sequence_output,attns,input_ids,hop=3) 
          elif copymethod == 5:
            #Use a, b 
            # This creates 2-D attention, vertical attention for obtaining relavant words per word; 
            # horizantal attention to weight-average per the attentions per question word 
            copy_qgen_logits = (1-p) * self.get_copy_probs_lateral(sequence_output,attns,input_ids,hop=21) 
          elif copymethod == 6:
            #Use a+[CLS] attention, obtain a by combining (head*layer) self-attentions
            copy_qgen_logits = (1-p) * self.get_copy_probs(sequence_output,attns,input_ids,cls_att=1)
          elif copymethod == 7:
            #Use a+[CLS] attention, obtain a by combining (head*layer) self-attentions
            copy_qgen_logits = (1-p) * self.get_copy_probs(sequence_output,attns,input_ids,cls_att=2,answer_ids=answer_phrase_ids)
          elif copymethod == 8: # max out pointer networks
            copy_qgen_logits = (1-p) * self.get_copy_probs(sequence_output,attns,input_ids,maxout_pointer=True)
          elif copymethod == 9:
            #Use a, obtain a by combining (head*layer) self-attentions
            copy_qgen_logits = (1-p) * self.get_copy_probs_sum(sequence_output,attns,input_ids)
          if copymethod == 11:
            #Use a, obtain a by combining (head*layer) self-attentions
            copy_qgen_logits = (1-p) * self.get_copy_probs(sequence_output,attns,input_ids,para_mask=question2para_mask)

          #print(copy_qgen_logits[:,:,0:10])
          #input("copy_qgen_logits look like this")

          qgen_logits = qgen_logits + copy_qgen_logits
        else:
          #print("no copy method")
          qgen_logits =  self.qgen_outputs(sequence_output)
          qgen_logits = torch.softmax(qgen_logits, dim=-1)
        

        # It becomes LogSoftMax after this step
        qgen_logits = torch.log(qgen_logits)

        #print(copy_qgen_logits.size(), qgen_logits.size())
        #input("sizes")
        #Getmax:
        #Debug

        '''
        val, ind = qgen_logits.max(-1)
        for b in range(prediction_ids.size(0)):
         print("#######")
         print("#source "," ".join(tokenizer.convert_ids_to_tokens([ input_ids[b][i].item() for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ])))
         print("#target "," ".join(tokenizer.convert_ids_to_tokens([ prediction_ids[b][i].item() for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ])))
         print("#predicted"," ".join(tokenizer.convert_ids_to_tokens([ ind[b][i].item() for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ])))
         input("predicted question")
         #print("#ids"," ".join([ str(ind[b][i].item()) for i in range(prediction_ids[b].size(0)) if prediction_ids[b][i].item() !=0 ]))
        '''

        #print(input_ids[0])
        #print(ind[0])
        #input("max")
        #print(start_positions[0], end_positions[0], input_ids[0][start_positions[0].item()],input_ids[0][end_positions[0].item()])
        #input("start and end positions")


        #Question prediction
        if not evaluate: #start_positions is not None and end_positions is not None:
         #loss_fct = CrossEntropyLoss(ignore_index=0)
         #loss = loss_fct(qgen_logits.view(-1,self.vocab_size), prediction_ids.view(-1))
         #print(qgen_logits[:,:,0:10])
         #input("wait")

         loss_fct = NLLLoss(ignore_index=0)
         loss = loss_fct(qgen_logits.view(-1,self.vocab_size), prediction_ids.view(-1))
         #print(loss)
         #input("loss")

          
         return loss
        else:
         return qgen_logits
