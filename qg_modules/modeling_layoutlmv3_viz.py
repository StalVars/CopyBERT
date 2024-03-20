# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
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
"""PyTorch BERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import copy
import json
import logging
import math
import os
import shutil
import tarfile
import tempfile
import sys
from io import open
from typing import List, Optional, Tuple, Union

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import NLLLoss

logger = logging.getLogger(__name__)
from transformers import *
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

#from transformers.tokenization_bert import (BasicTokenizer,
from transformers.models.bert.tokenization_bert import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

PRETRAINED_MODEL_ARCHIVE_MAP = {
    'bert-base-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz",
    'bert-large-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased.tar.gz",
    'bert-base-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased.tar.gz",
    'bert-large-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased.tar.gz",
    'bert-base-multilingual-uncased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased.tar.gz",
    'bert-base-multilingual-cased': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased.tar.gz",
    'bert-base-chinese': "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz",
}
BERT_CONFIG_NAME = 'bert_config.json'
TF_WEIGHTS_NAME = 'model.ckpt'

#tokenizer = BertTokenizer.from_pretrained("models/BioBERT_SQuAD_trained", do_lower_case=True)

def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print("Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    print("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `LayoutLMv3Model`.
    """
    def __init__(self,
                 vocab_size_or_config_json_file,
                 hidden_size=768,
                 num_hidden_layers=12,
                 num_attention_heads=12,
                 intermediate_size=3072,
                 hidden_act="gelu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0.1,
                 max_position_embeddings=512,
                 type_vocab_size=2,
                 initializer_range=0.02):
        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `LayoutLMv3Model`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `LayoutLMv3Model`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
 
        #Just incase
        self.alter_tok_embeddings = None

        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())

try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info("Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex .")
    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        #print(config)
        if config.alter_tok_embeddings is not None:
         if config.alter_tok_embeddings == 1:
          self.token_4_type_embeddings = nn.Embedding(4, config.hidden_size)
         elif config.alter_tok_embeddings != 1:
          self.embeddings4 = nn.Parameter(torch.Tensor(4, config.hidden_size))

        self.alter_tok_embeddings = config.alter_tok_embeddings # This suggests to use 3 token ids

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        #if not self.alter_tok_embeddings.isdigit():
        if self.alter_tok_embeddings is None:
          token_type_embeddings = self.token_type_embeddings(token_type_ids)
        elif self.alter_tok_embeddings == 1:
          token_type_embeddings = self.token_4_type_embeddings(token_type_ids)
        else: # weighted average of segment ids
          token_type_embeddings = torch.bmm(token_type_ids,self.embeddings4) # b x seq x 4 * b x 4 x hidden_size

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class BertOtherAttention(nn.Module):
    def __init__(self, config):
        super(BertOtherAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, input_encoder_states, attn=False):
        mixed_query_layer = self.query(hidden_states)
        #mixed_key_layer = self.key(hidden_states)
        #mixed_value_layer = self.value(hidden_states)
        mixed_key_layer = self.key(input_encoder_states)
        mixed_value_layer = self.value(input_encoder_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        # Apply the attention mask is (precomputed for all layers in LayoutLMv3Model forward() function)
        
        attention_scores = attention_scores + attention_mask
        #torch.save(attention_scores, "attention_scores")
        #input("")

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        #torch.save(attention_probs,"attention_probs")
        #input("attention probs")

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if attn:
         return context_layer, attention_probs
        else:
         return context_layer

class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask,attn=False):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in LayoutLMv3Model forward() function)
        #print("attention scores",attention_scores.size())
        #print("attention mask",attention_mask.size())
        
        attention_scores = attention_scores + attention_mask
        #torch.save(attention_scores, "attention_scores")
        #input("")

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        #torch.save(attention_probs,"attention_probs")
        #input("attention probs")

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if attn:
         return context_layer, attention_probs
        else:
         return context_layer

class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):

    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask,attn=False):

        if attn:
         self_output,self_attentions = self.self(input_tensor, attention_mask,attn=True)
        else:
         self_output = self.self(input_tensor, attention_mask)

        attention_output = self.output(self_output, input_tensor)

        if attn:
         return attention_output, self_attentions
        else:
         return attention_output

class BertDecoderAttention(nn.Module):
    def __init__(self, config):
        super(BertDecoderAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.enc2dec = BertOtherAttention(config)
        self.output = BertSelfOutput(config)
        self.output_2 = BertSelfOutput(config)

    def forward(self, input_tensor, encoder_tensor, output_mask, input_mask, attn=False):

        #if attn:
        # self_output,self_attentions = self.self(input_tensor, attention_mask,attn=True)
        #else:

        # attention mask should be lower triangle
        self_output = self.self(input_tensor, output_mask) # multi-head self-attention
        self_output = self.output(self_output, input_tensor)  # Add and Normalize

        if attn:
         dec_output, dec_attentions = self.enc2dec(self_output, input_mask, encoder_tensor,attn=True) # multi-head attention on input
         dec_output = self.output_2(dec_output, self_output) # Add and Normalize

         return dec_output, dec_attentions
        else:
         dec_output = self.enc2dec(attention_output, encoder_mask, encoder_tensor)
         return attention_output

class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

class BertDecoderLayer(nn.Module):
    def __init__(self, config):
        super(BertDecoderLayer, self).__init__()
        #self.attention = BertAttention(config)
        self.attention = BertDecoderAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, input_encoded_states, output_mask, input_mask, attn=False):

        if attn:
         attention_output, attentions = self.attention(hidden_states, input_encoded_states, output_mask, input_mask, attn=True)
        else:
         attention_output = self.attention(hidden_states, input_encoded_states, output_mask, input_mask)

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)

        if attn:
          return layer_output, attentions
        else:
          return layer_output

class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask,attn=False):

        if attn:
         attention_output, attentions = self.attention(hidden_states, attention_mask,attn=True)
        else:
         attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)

        layer_output = self.output(intermediate_output, attention_output)

        if attn:
          return layer_output, attentions
        else:
          return layer_output

class BertDecoder(nn.Module):
    def __init__(self, config):
        super(BertDecoder, self).__init__()
        layer = BertDecoderLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states,input_encoded_states, output_mask, input_mask, output_all_decoded_layers=True, attn=False):
        all_decoder_layers = []
        all_attentions = []

        #for layer_module,decoder_layer_module in zip(self.encoder_layer,self.layer):

        for layer_module in self.layer:

            if attn:
              hidden_states, attentions = layer_module(hidden_states, input_encoded_states, output_mask, input_mask, attn=True)
              all_attentions.append(attentions)
            else:
              hidden_states = layer_module(hidden_states, input_encoded_states, output_mask, input_mask)

            if output_all_decoded_layers:
                all_decoder_layers.append(hidden_states)

        if not output_all_decoded_layers:
            all_decoder_layers.append(hidden_states)

        if attn:
          return all_decoder_layers, all_attentions

        return all_decoder_layers

class BertEncoder(nn.Module):
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        layer = BertLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True, attn=False):
        all_encoder_layers = []
        all_attentions = []
        for layer_module in self.layer:
            #print(attention_mask.size())
            #input("layerwise attention")
            if attn:
              hidden_states, attentions = layer_module(hidden_states, attention_mask,attn=True)
              all_attentions.append(attentions)
            else:
              hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        if attn:
          return all_encoder_layers, all_attentions

        return all_encoder_layers


class BertPooler(nn.Module):
    def __init__(self, config):
        super(BertPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(bert_model_embedding_weights.size(1),
                                 bert_model_embedding_weights.size(0),
                                 bias=False)
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class LayoutLMv3PreTrainedModel_own(nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, config, *inputs, **kwargs):
        super(LayoutLMv3PreTrainedModel, self).__init__()
        if not isinstance(config, BertConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `BertConfig`. "
                "To create a model from a Google pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        self.config = config

    def init_bert_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """
        Instantiate a LayoutLMv3PreTrainedModel from a pre-trained model file or a pytorch state dict.
        Download and cache the pre-trained model file if needed.

        Params:
            pretrained_model_name_or_path: either:
                - a str with the name of a pre-trained model to load selected in the list of:
                    . `bert-base-uncased`
                    . `bert-large-uncased`
                    . `bert-base-cased`
                    . `bert-large-cased`
                    . `bert-base-multilingual-uncased`
                    . `bert-base-multilingual-cased`
                    . `bert-base-chinese`
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `pytorch_model.bin` a PyTorch dump of a LayoutLMv3ForPreTraining instance
                - a path or url to a pretrained model archive containing:
                    . `bert_config.json` a configuration file for the model
                    . `model.chkpt` a TensorFlow checkpoint
            from_tf: should we load the weights from a locally saved TensorFlow checkpoint
            cache_dir: an optional path to a folder in which the pre-trained models will be cached.
            state_dict: an optional state dictionnary (collections.OrderedDict object) to use instead of Google pre-trained models
            *inputs, **kwargs: additional input for the specific Bert class
                (ex: num_labels for LayoutLMv3ForSequenceClassification)
        """
        state_dict = kwargs.get('state_dict', None)
        kwargs.pop('state_dict', None)
        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)
        logger.info("Model config {}".format(config))
        # Instantiate model.
        model = cls(config, *inputs, **kwargs)
        if state_dict is None and not from_tf:
            weights_path = os.path.join(serialization_dir, WEIGHTS_NAME)
            state_dict = torch.load(weights_path, map_location='cpu')
        if tempdir:
            # Clean up temp dir
            shutil.rmtree(tempdir)
        if from_tf:
            # Directly load from a TensorFlow checkpoint
            weights_path = os.path.join(serialization_dir, TF_WEIGHTS_NAME)
            return load_tf_weights_in_bert(model, weights_path)
        # Load from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startswith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model

class LayoutLMv3Model_own(LayoutLMv3PreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.LayoutLMv3Model(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(LayoutLMv3Model, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,decoder_mask=False,attn=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #print(attention_mask.size())
        #input("attention mask")
        if decoder_mask:
           extended_attention_mask = attention_mask.unsqueeze(1)
        else:
           extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #print(extended_attention_mask.size())
        #input("extended attention mask 0")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #print(extended_attention_mask.size())
        #input("Extended attention mask 1")

        extended_attention_mask = extended_attention_mask.expand(-1,-1,extended_attention_mask.size(-1), -1)

        #print(extended_attention_mask.size())
        #naspurinput("Extended attention mask 2")

        embedding_output = self.embeddings(input_ids, token_type_ids)
        if attn:
         encoded_layers, attentions = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,attn=True)
        else:
         encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers,attn=False)

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        if attn:
         return encoded_layers, pooled_output, attentions
        else:
         return encoded_layers, pooled_output

class BertDecoderModel(LayoutLMv3PreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model
    # Usage below HAS TO BE CHANGED

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.LayoutLMv3Model(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(BertDecoderModel, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertDecoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, encoder_output, output_mask=None, input_mask=None, token_type_ids=None, output_all_decoded_layers=True,decoder_mask=False,attn=False):

        if output_mask is None:
            output_mask = torch.ones_like(input_ids)
        '''
        if input_mask is None:
            input_mask = torch.ones_like(encoder_output)
        '''
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.

        #if decoder_mask:
        #   extended_attention_mask = attention_mask.unsqueeze(1)
        #else:
        #    extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        extended_output_mask = output_mask.unsqueeze(1)
        extended_input_mask = input_mask.unsqueeze(1)


        # Since output_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.

        extended_output_mask = extended_output_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_output_mask = (1.0 - extended_output_mask) * -10000.0
        #extended_output_mask = extended_output_mask.expand(-1,-1,extended_output_mask.size(-1), -1)

        extended_input_mask = extended_input_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_input_mask = (1.0 - extended_input_mask) * -10000.0
        #print(extended_input_mask.size(), extended_output_mask.size())
        #extended_input_mask = extended_input_mask.expand(-1,-1,extended_output_mask.size(-1), -1) # You are right, extended_output_mask.size(-1), not extended_input_mask.size(-1)

        embedding_output = self.embeddings(input_ids, token_type_ids)

        if attn:
         decoded_layers, attentions = self.encoder(embedding_output,
                                      encoder_output,
                                      extended_output_mask,
                                      extended_input_mask,
                                      output_all_decoded_layers=output_all_decoded_layers,attn=True)
        else:
         decoded_layers = self.encoder(embedding_output,
                                      encoder_output,
                                      extended_output_mask,
                                      extended_input_mask,
                                      output_all_decoded_layers=output_all_decoded_layers,attn=False)

        sequence_output = decoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if not output_all_decoded_layers:
            decoded_layers = decoded_layers[-1]
        if attn:
         return decoded_layers, pooled_output, attentions
        else:
         return decoded_layers, pooled_output


class LayoutLMv3Model_prev(LayoutLMv3PreTrainedModel):
    """BERT model ("Bidirectional Embedding Representations from a Transformer").

    Params:
        config: a BertConfig class instance with the configuration to build a new model

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `output_all_encoded_layers`: boolean which controls the content of the `encoded_layers` output as described below. Default: `True`.

    Outputs: Tuple of (encoded_layers, pooled_output)
        `encoded_layers`: controled by `output_all_encoded_layers` argument:
            - `output_all_encoded_layers=True`: outputs a list of the full sequences of encoded-hidden-states at the end
                of each attention block (i.e. 12 full sequences for BERT-base, 24 for BERT-large), each
                encoded-hidden-state is a torch.FloatTensor of size [batch_size, sequence_length, hidden_size],
            - `output_all_encoded_layers=False`: outputs only the full sequence of hidden-states corresponding
                to the last attention block of shape [batch_size, sequence_length, hidden_size],
        `pooled_output`: a torch.FloatTensor of size [batch_size, hidden_size] which is the output of a
            classifier pretrained on top of the hidden state associated to the first character of the
            input (`CLS`) to train on the Next-Sentence task (see BERT's paper).

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = modeling.BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = modeling.LayoutLMv3Model(config=config)
    all_encoder_layers, pooled_output = model(input_ids, token_type_ids, input_mask)
    ```
    """

    def __init__(self, config):
        super(LayoutLMv3Model, self).__init__(config)
        self.embeddings = BertEmbeddings(config)
        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, output_all_encoded_layers=True,decoder_mask=False):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        #print(attention_mask.size())
        #input("attention mask")
        if decoder_mask:
           extended_attention_mask = attention_mask.unsqueeze(1)
        else:
           extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        #print(extended_attention_mask.size())
        #input("extended attention mask 0")

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #print(extended_attention_mask.size())
        #input("Extended attention mask 1")

        extended_attention_mask = extended_attention_mask.expand(-1,-1,extended_attention_mask.size(-1), -1)
        #print(extended_attention_mask.size())
        #input("Extended attention mask 2")

        embedding_output = self.embeddings(input_ids, token_type_ids)
        encoded_layers = self.encoder(embedding_output,
                                      extended_attention_mask,
                                      output_all_encoded_layers=output_all_encoded_layers)

        sequence_output = encoded_layers[-1]
        pooled_output = self.pooler(sequence_output)

        if not output_all_encoded_layers:
            encoded_layers = encoded_layers[-1]
        return encoded_layers, pooled_output


class LayoutLMv3ForPreTraining(LayoutLMv3PreTrainedModel):
    """BERT model with pre-training heads.
    This module comprises the BERT model followed by the two pre-training heads:
        - the masked language modeling head, and
        - the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: optional masked language modeling labels: torch.LongTensor of shape [batch_siz sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]
        `next_sentence_label`: optional next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `masked_lm_labels` and `next_sentence_label` are not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `masked_lm_labels` or `next_sentence_label` is `None`:
            Outputs a tuple comprising
            - the masked language modeling logits of shape [batch_size, sequence_length, vocab_size], and
            - the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForPreTraining(config)
    masked_lm_logits_scores, seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(LayoutLMv3ForPreTraining, self).__init__(config)
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.cls = BertPreTrainingHeads(config, self.layoutlmv3.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None, next_sentence_label=None):
        sequence_output, pooled_output = self.layoutlmv3(input_ids, token_type_ids, attention_mask,
                                                   output_all_encoded_layers=False)
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)

        if masked_lm_labels is not None and next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss = masked_lm_loss + next_sentence_loss
            return total_loss
        else:
            return prediction_scores, seq_relationship_score


class LayoutLMv3ForMaskedLM(LayoutLMv3PreTrainedModel):
    """BERT model with the masked language modeling head.
    This module comprises the BERT model followed by the masked language modeling head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `masked_lm_labels`: masked language modeling labels: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [-1, 0, ..., vocab_size]. All labels set to -1 are ignored (masked), the loss
            is only computed for the labels set in [0, ..., vocab_size]

    Outputs:
        if `masked_lm_labels` is  not `None`:
            Outputs the masked language modeling loss.
        if `masked_lm_labels` is `None`:
            Outputs the masked language modeling logits of shape [batch_size, sequence_length, vocab_size].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForMaskedLM(config)
    masked_lm_logits_scores = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(LayoutLMv3ForMaskedLM, self).__init__(config)
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.cls = BertOnlyMLMHead(config, self.layoutlmv3.embeddings.word_embeddings.weight)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, masked_lm_labels=None):
        sequence_output, _ = self.layoutlmv3(input_ids, token_type_ids, attention_mask,
                                       output_all_encoded_layers=False)
        prediction_scores = self.cls(sequence_output)

        if masked_lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), masked_lm_labels.view(-1))
            return masked_lm_loss
        else:
            return prediction_scores


class LayoutLMv3ForNextSentencePrediction(LayoutLMv3PreTrainedModel):
    """BERT model with next sentence prediction head.
    This module comprises the BERT model followed by the next sentence classification head.

    Params:
        config: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `next_sentence_label`: next sentence classification loss: torch.LongTensor of shape [batch_size]
            with indices selected in [0, 1].
            0 => next sentence is the continuation, 1 => next sentence is a random sentence.

    Outputs:
        if `next_sentence_label` is not `None`:
            Outputs the total_loss which is the sum of the masked language modeling loss and the next
            sentence classification loss.
        if `next_sentence_label` is `None`:
            Outputs the next sentence classification logits of shape [batch_size, 2].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForNextSentencePrediction(config)
    seq_relationship_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(LayoutLMv3ForNextSentencePrediction, self).__init__(config)
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.cls = BertOnlyNSPHead(config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, next_sentence_label=None):
        _, pooled_output = self.layoutlmv3(input_ids, token_type_ids, attention_mask,
                                     output_all_encoded_layers=False)
        seq_relationship_score = self.cls( pooled_output)

        if next_sentence_label is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-1)
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            return next_sentence_loss
        else:
            return seq_relationship_score


class LayoutLMv3ForSequenceClassification(LayoutLMv3PreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary. Items in the batch should begin with the special "CLS" token. (see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = LayoutLMv3ForSequenceClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(LayoutLMv3ForSequenceClassification, self).__init__(config)
        self.num_labels = num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        _, pooled_output = self.layoutlmv3(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits


class LayoutLMv3ForMultipleChoice(LayoutLMv3PreTrainedModel):
    """BERT model for multiple choice tasks.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_choices`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length]
            with the token types indices selected in [0, 1]. Type 0 corresponds to a `sentence A`
            and type 1 corresponds to a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, num_choices, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size]
            with indices selected in [0, ..., num_choices].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[[31, 51, 99], [15, 5, 0]], [[12, 16, 42], [14, 28, 57]]])
    input_mask = torch.LongTensor([[[1, 1, 1], [1, 1, 0]],[[1,1,0], [1, 0, 0]]])
    token_type_ids = torch.LongTensor([[[0, 0, 1], [0, 1, 0]],[[0, 1, 1], [0, 0, 1]]])
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_choices = 2

    model = LayoutLMv3ForMultipleChoice(config, num_choices)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_choices):
        super(LayoutLMv3ForMultipleChoice, self).__init__(config)
        self.num_choices = num_choices
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        flat_input_ids = input_ids.view(-1, input_ids.size(-1))
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        _, pooled_output = self.layoutlmv3(flat_input_ids, flat_token_type_ids, flat_attention_mask, output_all_encoded_layers=False)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, self.num_choices)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            return loss
        else:
            return reshaped_logits


class LayoutLMv3ForTokenClassification(LayoutLMv3PreTrainedModel):
    """BERT model for token-level classification.
    This module is composed of the BERT model with a linear layer on top of
    the full hidden state of the last layer.

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.
        `num_labels`: the number of classes for the classifier. Default = 2.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `labels`: labels for the classification output: torch.LongTensor of shape [batch_size, sequence_length]
            with indices selected in [0, ..., num_labels].

    Outputs:
        if `labels` is not `None`:
            Outputs the CrossEntropy classification loss of the output with the labels.
        if `labels` is `None`:
            Outputs the classification logits of shape [batch_size, sequence_length, num_labels].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    num_labels = 2

    model = LayoutLMv3ForTokenClassification(config, num_labels)
    logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config, num_labels):
        super(LayoutLMv3ForTokenClassification, self).__init__(config)
        self.num_labels = num_labels
        self.layoutlmv3 = LayoutLMv3Model(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, _ = self.layoutlmv3(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            return loss
        else:
            return logits

class LayoutLMv3ForPhraseExtraction(LayoutLMv3PreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(LayoutLMv3ForPhraseExtraction, self).__init__(config)
        self.layoutlmv3 = LayoutLMv3Model(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_start_outputs = nn.Linear(config.hidden_size, 1)
        self.qa_end_outputs = nn.Linear(config.hidden_size, 1)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.layoutlmv3(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        start_logits = self.qa_start_outputs(sequence_output)
        #start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            #print("input_ids",input_ids.size())
            #print("start_positions",start_positions.size())
            position_vec=torch.zeros_like(input_ids) 
            position_vec.scatter_(1,start_positions.unsqueeze(1), 1) #scatters vocab
            #print(token_type_ids[0])
            #input("token_type_ids")

            sequence_output2, _ = self.layoutlmv3(input_ids, position_vec, attention_mask, output_all_encoded_layers=False)
            end_logits = self.qa_end_outputs(sequence_output2)
            end_logits = end_logits.squeeze(-1)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

class LayoutLMv3ForQuestionAnswering(LayoutLMv3PreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(LayoutLMv3ForQuestionAnswering, self).__init__(config)
        self.layoutlmv3 = LayoutLMv3Model(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None):
        sequence_output, _ = self.layoutlmv3(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits

class LayoutLMv3ForQuestionGenerationFluidSegments(LayoutLMv3PreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(LayoutLMv3ForQuestionGenerationFluidSegments, self).__init__(config)
        self.layoutlmv3 = LayoutLMv3Model(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.config = config
        self.qgen_outputs = nn.Linear(config.hidden_size, config.vocab_size)
        self.apply(self.init_bert_weights)
        #self.MASKID = 103
        self.MASKID = 103
        self.vocab_size = config.vocab_size
        self.iscuda = True
        self.attentionsquare = nn.Linear(config.hidden_size, 12 * 12 ) # config.n_heads * config.n_layers
        self.attentionsquare_lateral = nn.Linear(config.hidden_size, 12 * 12 ) # config.n_heads * config.n_layers
        self.attention_over_attention = nn.SoftMax(nn.Linear(config.hidden_size, 3 ),dim=-1) # config.n_heads * config.n_layers
        self.linear_copy_probs = nn.Linear(config.hidden_size, config.hidden_size ) # config.n_heads * config.n_layers

        self.p_value = nn.Linear(config.hidden_size, 1)

    def get_copy_probs_lateral(self,sequence_output, self_attentions,src_seq): # This is 2-D Attention

        # Use sequence_output to choose the self_attentions wisely 
        attention_square = self.attentionsquare(sequence_output) # batch_size x seq_size x (12*12)
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

        combined_attention = torch.bmm(attention, attention)

        # attention value doesn't sum up to 1? why?
        # src_seq has the source vocab ids
        copy_probs=torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size)

        if self.iscuda==True:
          copy_probs = copy_probs.cuda()

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(combined_attention, copy_probs) # copy scores

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def get_copy_probs(self,sequence_output, self_attentions,src_seq):

        # Use sequence_output to choose the self_attentions wisely 
        attention_square = self.attentionsquare(sequence_output) # batch_size x seq_size x (12*12)
        attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)
        self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size
        self_attentions = self_attentions.transpose(1,2) # batch_size x seq_size x (12*12) x seq_size

        # The below does weighted average over all the attentions of all the layers and heads
        #print(attention_square.size(), self_attentions.size())

        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 

        # attention value doesn't sum up to 1? why?
        # src_seq has the source vocab ids
        copy_probs=torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size)

        if self.iscuda==True:
          copy_probs = copy_probs.cuda()

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(attention, copy_probs) # copy scores

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs


    def get_copy_probs2(self,sequence_output, input_mask,src_seq):

        # Use sequence_output to choose the self_attentions wisely 
        transformed_sequence_output = self.linear_copy_probs(sequence_output)
        affinity =  torch.bmm(transformed_sequence_output, sequence_output.transpose(1,2))  

        mask = (1 - input_mask) * -10000
 
        affinity = affinity + mask
        probs = nn.Softmax(dim=-1)(affinity)

        # src_seq has the source vocab ids
        src_to_vocab_map=torch.zeros(src_seq.size(0), input_mask.size(2), self.config.vocab_size) #batch_size x seq_size x vocab_size


        if self.iscuda==True:
          src_to_vocab_map = src_to_vocab_map.cuda()

        src_to_vocab_map.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(probs, src_to_vocab_map) # copy scores
        #print(copy_probs.sum(-1))
        #input("copy_probs")

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,decoder_mask=True,evaluate=False, copymethod=1):

        #print("Forward for Question Generation")
        # Get Decoder Mask for Question Part
        #attention_mask = attention_mask.unsqueeze(1).expand(-1,attention_mask.size(-1),-1) # Use batch_size x seq_length x seq_length instead of batch_size x seq_length

        #question_mask_ids = token_type_ids.unsqueeze(-1).float()
        question_mask_ids = (token_type_ids == 4).unsqueeze(-1).float() # to-be-changed
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
        '''
        para_mask = para_mask.transpose(1,2)
        extra_mask = question_mask + para_mask
        extra_mask = extra_mask.long()
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
        answer_phrase_ids = answer_phrase_ids.scatter(-1,start_positions.unsqueeze(-1),1)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,(end_positions+1).unsqueeze(-1),-1)
        answer_phrase_ids = answer_phrase_ids.float().matmul(torch.triu(torch.ones(answer_phrase_ids.size(1), answer_phrase_ids.size(1)) ).cuda() ).long() # segment_id 1 for answer phrase

        # For Question Generation

        #sequence_output, pooled_output, attns = self.layoutlmv3(input_ids, token_type_ids + answer_phrase_ids, extra_mask, output_all_encoded_layers=False,decoder_mask=decoder_mask,attn=True)
        outputs = self.layoutlmv3(input_ids, token_type_ids + answer_phrase_ids, extra_mask, output_all_encoded_layers=False,decoder_mask=decoder_mask,attn=True)
        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs["pooler_output"]
        attns = outputs["attentions"]



        # For LM
        #sequence_output, pooled_output, attns = self.layoutlmv3(input_ids, token_type_ids, extra_mask, output_all_encoded_layers=False,decoder_mask=decoder_mask,attn=True)

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
          qgen_logits = p * self.cls.predictions(sequence_output)

          if copymethod == 1:
            #print("no copy method 1")
            copy_qgen_logits = (1-p) * self.get_copy_probs(sequence_output,attns,input_ids)
          elif copymethod == 2:
            copy_qgen_logits = (1-p) * self.get_copy_probs2(sequence_output,para_mask,input_ids) # Just create one attention layer on top of the encoded input
          elif copymethod == 3:
            # This creates 2-D attention, vertical attention for obtaining relavant words per word; 
            # horizantal attention to weight-average per the attentions per question word 
            copy_qgen_logits = (1-p) * self.get_copy_probs_lateral(sequence_output,attns,input_ids) 
          qgen_logits = qgen_logits + copy_qgen_logits
        else:
          #print("no copy method")
          qgen_logits =  self.cls.predictions(sequence_output)

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
         loss_fct = CrossEntropyLoss(ignore_index=0)
         loss = loss_fct(qgen_logits.view(-1,self.vocab_size), prediction_ids.view(-1))
         return loss
        else:
         return qgen_logits

class LayoutLMv3ModelWithViz(LayoutLMv3Model):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        #self.embeddings = BertEmbeddings(config)
        #self.encoder = BertEncoder(config)

        #self.pooler = BertPooler(config) if add_pooling_layer else None

        # Initialize weights and apply final processing
        #self.post_init()
        self.coordinate_size=128
        self.shape_size=128
        self.max_2d_position_embeddings = 1024

        # Spatial 2d-position embeddings
        self.x_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.coordinate_size)
        self.y_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.coordinate_size)
        self.h_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.shape_size)
        self.w_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.shape_size)
        # spatial position embedding concat: 128 * 4 (coordinates) + 128 * 2 (shape) 
        #  = 128 * 6 = 768 

    def calculate_spatial_position_embeddings(self, bbox):
        try:
            #print(torch.max(bbox[:,:]))

            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))

        # below is the difference between LayoutLMEmbeddingsV2 (torch.cat) and LayoutLMEmbeddingsV1 (add)
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        bboxes: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:



        # Caculate bbox embeds:
        bbox_embeds = self.calculate_spatial_position_embeddings(bboxes)
        #bbox_embeds = self.spatial_linear(spatial_position_embeddings)

        # Add bbox embeds

        #print(input_ids)


        inputs_embeds = super().embeddings.word_embeddings(input_ids)
        inputs_embeds = inputs_embeds + bbox_embeds

        return_dict = super().forward(input_ids=None, 
                attention_mask = attention_mask,
                token_type_ids = token_type_ids, 
                position_ids = position_ids, 
                head_mask = head_mask,
                inputs_embeds = inputs_embeds,
                encoder_hidden_states = encoder_hidden_states, 
                encoder_attention_mask = encoder_attention_mask, 
                past_key_values = past_key_values, 
                use_cache = use_cache, 
                output_attentions = output_attentions, 
                output_hidden_states = output_hidden_states,
                return_dict = return_dict)
        return return_dict 


class TwoDPosEmbeddings(LayoutLMv3PreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(TwoDPosEmbeddings, self).__init__(config)

        #self.layoutlmv3 = LayoutLMv3ModelWithViz(config)
        self.layoutlmv3 = LayoutLMv3Model(config)

        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.config = config
        #self.cls = BertLMPredictionHead(config, self.layoutlmv3.embeddings.word_embeddings.weight)
        self.cls = BertPreTrainingHeads(config, self.layoutlmv3.embeddings.word_embeddings.weight)
        #self.qgen_outputs = nn.Linear(config.hidden_size, config.vocab_size)
        #self.qgen_outputs = self.cls.predictions # Make use of pretrained layer

        #self.apply(self.init_bert_weights)
        self.init_weights()
        self.MASKID = 103
        self.vocab_size = config.vocab_size
        self.iscuda = True
        self.attentionsquare = nn.Linear(config.hidden_size, config.num_attention_heads * config.num_hidden_layers ) # config.n_heads * config.n_layers
        self.attentionsquare_lateral = nn.Linear(config.hidden_size, config.num_attention_heads * config.num_hidden_layers ) # config.n_heads * config.n_layers
        self.attention_over_attention = nn.Linear(config.hidden_size, 3 ) # 
        self.attention_over_attention2 = nn.Linear(config.hidden_size, 2 ) # 
        self.linear_copy_probs = nn.Linear(config.hidden_size, config.hidden_size ) # config.n_heads * config.n_layers

        self.p_value = nn.Linear(config.hidden_size, 1)

        self.coordinate_size=128
        self.shape_size=128
        self.max_2d_position_embeddings = 1024
        # Spatial 2d-position embeddings
        self.x_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.coordinate_size)
        self.y_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.coordinate_size)
        self.h_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.shape_size)
        self.w_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.shape_size)
        # spatial position embedding concat: 128 * 4 (coordinates) + 128 * 2 (shape) 
        #  = 128 * 6 = 768 

    def calculate_2d_position_embeddings(self, point):
        try:
            #print(torch.max(bbox[:,:]))

            x_position_embeddings = self.x_position_embeddings(point[:, 0])
            y_position_embeddings = self.y_position_embeddings(point[:, 1])

        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        spatial_position_embeddings = torch.cat(
            [
                x_position_embeddings,
                y_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

    def forward(self, points, labels):


        point_embeddings = self.calculate_2d_position_embeddings(points)
        cosine_sim = point_embeddings.matmul(point_embeddings.transpose(-1,-2))
        loss = (cosine_sim - labels)**2
        loss = loss.mean()

        return loss, cosine_sim

class LayoutLMv3ForQuestionGeneration(LayoutLMv3PreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(LayoutLMv3ForQuestionGeneration, self).__init__(config)

        #self.layoutlmv3 = LayoutLMv3ModelWithViz(config)
        self.layoutlmv3 = LayoutLMv3Model(config)

        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.config = config
        #self.cls = BertLMPredictionHead(config, self.layoutlmv3.embeddings.word_embeddings.weight)
        self.cls = BertPreTrainingHeads(config, self.layoutlmv3.embeddings.word_embeddings.weight)
        #self.qgen_outputs = nn.Linear(config.hidden_size, config.vocab_size)
        #self.qgen_outputs = self.cls.predictions # Make use of pretrained layer

        #self.apply(self.init_bert_weights)
        self.init_weights()
        self.MASKID = 103
        self.vocab_size = config.vocab_size
        self.iscuda = True
        self.attentionsquare = nn.Linear(config.hidden_size, config.num_attention_heads * config.num_hidden_layers ) # config.n_heads * config.n_layers
        self.attentionsquare_lateral = nn.Linear(config.hidden_size, config.num_attention_heads * config.num_hidden_layers ) # config.n_heads * config.n_layers
        self.attention_over_attention = nn.Linear(config.hidden_size, 3 ) # 
        self.attention_over_attention2 = nn.Linear(config.hidden_size, 2 ) # 
        self.linear_copy_probs = nn.Linear(config.hidden_size, config.hidden_size ) # config.n_heads * config.n_layers

        self.p_value = nn.Linear(config.hidden_size, 1)

        self.coordinate_size=128
        self.shape_size=128
        self.max_2d_position_embeddings = 1024
        # Spatial 2d-position embeddings
        self.x_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.coordinate_size)
        self.y_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.coordinate_size)
        self.h_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.shape_size)
        self.w_position_embeddings = nn.Embedding(self.max_2d_position_embeddings, self.shape_size)
        # spatial position embedding concat: 128 * 4 (coordinates) + 128 * 2 (shape) 
        #  = 128 * 6 = 768 

    def calculate_rbf_spatial_position_embeddings(self, bbox):
            #
            x = bbox[:,:,0]

    def calculate_spatial_position_embeddings(self, bbox):
        try:
            #print(torch.max(bbox[:,:]))

            left_position_embeddings = self.x_position_embeddings(bbox[:, :, 0])
            upper_position_embeddings = self.y_position_embeddings(bbox[:, :, 1])
            right_position_embeddings = self.x_position_embeddings(bbox[:, :, 2])
            lower_position_embeddings = self.y_position_embeddings(bbox[:, :, 3])
        except IndexError as e:
            raise IndexError("The `bbox` coordinate values should be within 0-1000 range.") from e

        h_position_embeddings = self.h_position_embeddings(torch.clip(bbox[:, :, 3] - bbox[:, :, 1], 0, 1023))
        w_position_embeddings = self.w_position_embeddings(torch.clip(bbox[:, :, 2] - bbox[:, :, 0], 0, 1023))


        #print(left_position_embeddings.size())
        #print(right_position_embeddings.size())
        #print(upper_position_embeddings.size())
        #print(lower_position_embeddings.size())
        #print(h_position_embeddings.size())
        #print(w_position_embeddings.size())


        # below is the difference between LayoutLMEmbeddingsV2 (torch.cat) and LayoutLMEmbeddingsV1 (add)
        spatial_position_embeddings = torch.cat(
            [
                left_position_embeddings,
                upper_position_embeddings,
                right_position_embeddings,
                lower_position_embeddings,
                h_position_embeddings,
                w_position_embeddings,
            ],
            dim=-1,
        )
        return spatial_position_embeddings

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

        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 
        # takes 2 tensors: (1):batch_size x seq_size x 1 x (LXH) (2): batch_size x seq_size x (LXH) x seq_size
        # ==> for all batch,position : do a matrix multiplication (1x12) with (12x seq_size)

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


    def forward(self, 
            input_ids, 
            token_type_ids=None, 
            attention_mask=None, 
            start_positions=None, 
            end_positions=None,
            bboxes=None,
            pixel_values=None,
            decoder_mask=True,
            evaluate=False, 
            copymethod=1):

        # print("Forward for Question Generation")
        # Get Decoder Mask for Question Part
        # attention_mask = attention_mask.unsqueeze(1).expand(-1,attention_mask.size(-1),-1) # Use batch_size x seq_length x seq_length instead of batch_size x seq_length

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

        #print("answer phrase ids:", answer_phrase_ids)
        #input("-")


        #print(input_ids[0])

        # Make the answer token ids to [MASK]:
        #input_ids = input_ids * (1-answer_phrase_ids) + answer_phrase_ids * self.MASKID

        #print(input_ids[0])
        #input("-")


        # For Question Generation
        #sequence_output, pooled_output, attns = self.layoutlmv3(input_ids=input_ids, token_type_ids = token_type_ids + answer_phrase_ids, attention_mask=extra_mask)
        #bboxes=bboxes,
        #print("max bbox",torch.max(bboxes))
        #print("min bbox", torch.min(bboxes))
        #input("-")

        bbox_embeds = self.calculate_spatial_position_embeddings(bboxes)

        inputs_embeds = self.layoutlmv3.embeddings.word_embeddings(input_ids)
        new_inputs_embeds = inputs_embeds + bbox_embeds
       
        #        inputs_embeds=new_inputs_embeds,
        outputs = self.layoutlmv3(input_ids  = input_ids,
                token_type_ids = token_type_ids + answer_phrase_ids, 
                attention_mask=extra_mask,
                bbox=bboxes) 

                #pixel_values=pixel_values,

        sequence_output = outputs["last_hidden_state"]
        pooled_output = outputs["pooler_output"]
        attns = outputs["attentions"]

        #q_outputs =  self.layoutlmv3(input_ids = query_ids, token_type_ids=torch.zeros_like(query_ids), attention_mask=query_input_mask)
        # For LM
        #sequence_output, pooled_output, attns = self.layoutlmv3(input_ids, token_type_ids, extra_mask, output_all_encoded_layers=False,decoder_mask=decoder_mask,attn=True)
          

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
          qgen_logits = p * torch.softmax(self.cls.predictions(sequence_output),dim=-1)
          #qgen_logits = p * self.cls.predictions(sequence_output)
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
          qgen_logits =  self.cls.predictions(sequence_output)
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

        '''
        # for start position and end position prediction
        if start_positions is not None and end_positions is not None:

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)

            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
        '''
class LayoutLMv3ForQuestionLM(LayoutLMv3PreTrainedModel):
    """BERT model for Question Answering (span extraction).
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(LayoutLMv3ForQuestionLM, self).__init__(config)
        #self.layoutlmv3 = LayoutLMv3ModelWithViz(config)
        self.layoutlmv3 = LayoutLMv3Model(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.config = config
        self.qgen_outputs = nn.Linear(config.hidden_size, config.vocab_size)
        self.apply(self.init_bert_weights)
        self.MASKID = 103
        self.vocab_size = config.vocab_size
        self.iscuda = True
        self.attentionsquare = nn.Linear(config.hidden_size, 12 * 12 ) # config.n_heads * config.n_layers
        self.attentionsquare_lateral = nn.Linear(config.hidden_size, 12 * 12 ) # config.n_heads * config.n_layers
        self.linear_copy_probs = nn.Linear(config.hidden_size, config.hidden_size ) # config.n_heads * config.n_layers

        self.p_value = nn.Linear(config.hidden_size, 1)

    def get_copy_probs_lateral(self,sequence_output, self_attentions,src_seq): # This is 2-D Attention

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

        #combined_attention = torch.bmm(attention, attention_lateral)
        combined_attention = torch.bmm(attention, attention)
        combined_attention = torch.bmm(attention, combined_attention)# 3-hop

        # attention value doesn't sum up to 1? why?
        # src_seq has the source vocab ids
        copy_probs=torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size)

        if self.iscuda==True:
          copy_probs = copy_probs.cuda()

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(combined_attention, copy_probs) # copy scores

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def get_copy_probs(self,sequence_output, self_attentions,src_seq,cls_att=0):

        # Use sequence_output to choose the self_attentions wisely 
        attention_square = torch.softmax(self.attentionsquare(sequence_output),dim=-1) # batch_size x seq_size x (12*12)
        attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)
        self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size
        self_attentions = self_attentions.transpose(1,2) # batch_size x seq_size x (12*12) x seq_size

        # The below does weighted average over all the attentions of all the layers and heads
        #print(attention_square.size(), self_attentions.size())

        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 
        #cls_attention
        print(attention.size())
        input("attn size")
          

        # attention value doesn't sum up to 1? why?
        # src_seq has the source vocab ids
        copy_probs=torch.zeros(src_seq.size(0), src_seq.size(1), self.config.vocab_size)

        if self.iscuda==True:
          copy_probs = copy_probs.cuda()

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        print(copy_probs.size())
        input("copy_probs")
        copy_probs = torch.bmm(attention, copy_probs) # copy scores
        print(copy_probs.size())
        input("copy_probs")

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def get_copy_probs2(self,sequence_output, input_mask,src_seq):

        # Use sequence_output to choose the self_attentions wisely 
        transformed_sequence_output = self.linear_copy_probs(sequence_output)
        affinity =  torch.bmm(transformed_sequence_output, sequence_output.transpose(1,2))  

        mask = (1-input_mask) * (-10000)
        affinity = affinity + mask
        probs = nn.Softmax(dim=-1)(affinity)

        # src_seq has the source vocab ids
        src_to_vocab_map=torch.zeros(src_seq.size(0), input_mask.size(2), self.config.vocab_size) #batch_size x seq_size x vocab_size


        if self.iscuda==True:
          src_to_vocab_map = src_to_vocab_map.cuda()

        src_to_vocab_map.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(probs, src_to_vocab_map) # copy scores

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,decoder_mask=True,evaluate=False, copymethod=1):

        #print("Forward for Question Generation")
        # Get Decoder Mask for Question Part
        #attention_mask = attention_mask.unsqueeze(1).expand(-1,attention_mask.size(-1),-1) # Use batch_size x seq_length x seq_length instead of batch_size x seq_length

        question_mask_ids = token_type_ids.unsqueeze(-1).float()
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

        #para_mask = torch.bmm((1-question_mask_ids)*input_mask_ids, ((1-question_mask_ids)*input_mask_ids).transpose(1,2))
        adjusted_para_mask = torch.bmm(input_mask_ids, ((1-question_mask_ids)*input_mask_ids).transpose(1,2))
        '''
        para_mask = para_mask.transpose(1,2)
        extra_mask = question_mask + para_mask
        extra_mask = extra_mask.long()
        '''

        extra_mask = torch.bmm(input_mask_ids, input_mask_ids.transpose(1,2)) 
        extra_mask = extra_mask * torch.tril(torch.zeros_like(extra_mask[0]) + 1 )

        extra_mask = extra_mask * (1-adjusted_para_mask) #para has no effect
        extra_mask = (extra_mask != 0).long()

        #extra_mask = attention_mask # lets check if without masking, prediction is good

        #print(extra_mask[0])
        #torch.save(extra_mask[0],"extra_mask_0")
        #input("extra mask")

        # attention_mask should not include future words for Question
        # token_typ_ids should contain Answer phrase

        answer_phrase_ids = torch.zeros_like(input_ids)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,start_positions.unsqueeze(-1),1)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,(end_positions+1).unsqueeze(-1),-1)
        answer_phrase_ids = answer_phrase_ids.float().matmul(torch.triu(torch.ones(answer_phrase_ids.size(1), answer_phrase_ids.size(1)) ).cuda() ).long() # segment_id 1 for answer phrase

        # For Question Generation
        sequence_output, pooled_output, attns = self.layoutlmv3(input_ids, token_type_ids + answer_phrase_ids, extra_mask, output_all_encoded_layers=False,decoder_mask=decoder_mask,attn=True)

        # For LM
        #sequence_output, pooled_output, attns = self.layoutlmv3(input_ids, token_type_ids, extra_mask, output_all_encoded_layers=False,decoder_mask=decoder_mask,attn=True)

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
          qgen_logits = p * self.cls.predictions(sequence_output)

          if copymethod == 1:
            #print("no copy method 1")
            copy_qgen_logits = (1-p) * self.get_copy_probs(sequence_output,attns,input_ids)
          elif copymethod == 2:
            copy_qgen_logits = (1-p) * self.get_copy_probs2(sequence_output,para_mask,input_ids) # This creates a new attention over outputs
          elif copymethod == 3:
            # This creates 2-D attention, vertical attention for obtaining relavant words per word; 
            # horizantal attention to weight-average per the attentions per question word 
            copy_qgen_logits = (1-p) * self.get_copy_probs_lateral(sequence_output,para_mask,input_ids) 
          qgen_logits = qgen_logits + copy_qgen_logits
        else:
          #print("no copy method")
          qgen_logits =  self.cls.predictions(sequence_output)

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
         loss_fct = CrossEntropyLoss(ignore_index=0)
         loss = loss_fct(qgen_logits.view(-1,self.vocab_size), prediction_ids.view(-1))
         return loss
        else:
         return qgen_logits

        '''
        # for start position and end position prediction
        if start_positions is not None and end_positions is not None:

            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)

            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)

            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)

            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            return total_loss
        else:
            return start_logits, end_logits
        '''
        #return loss
#####
 
class BertEncoder2Decoder(LayoutLMv3PreTrainedModel):
    """BERT model for Encoder to Decoder (text generation)
    This module is composed of the BERT model with a linear layer on top of
    the sequence output that computes start_logits and end_logits

    Params:
        `config`: a BertConfig class instance with the configuration to build a new model.

    ### Usage below HAS TO BE CHANGED
    Inputs:
        `input_ids`: a torch.LongTensor of shape [batch_size, sequence_length]
            with the word token indices in the vocabulary(see the tokens preprocessing logic in the scripts
            `extract_features.py`, `run_classifier.py` and `run_squad.py`)
        `token_type_ids`: an optional torch.LongTensor of shape [batch_size, sequence_length] with the token
            types indices selected in [0, 1]. Type 0 corresponds to a `sentence A` and type 1 corresponds to
            a `sentence B` token (see BERT paper for more details).
        `attention_mask`: an optional torch.LongTensor of shape [batch_size, sequence_length] with indices
            selected in [0, 1]. It's a mask to be used if the input sequence length is smaller than the max
            input sequence length in the current batch. It's the mask that we typically use for attention when
            a batch has varying length sentences.
        `start_positions`: position of the first token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.
        `end_positions`: position of the last token for the labeled span: torch.LongTensor of shape [batch_size].
            Positions are clamped to the length of the sequence and position outside of the sequence are not taken
            into account for computing the loss.

    Outputs:
        if `start_positions` and `end_positions` are not `None`:
            Outputs the total_loss which is the sum of the CrossEntropy loss for the start and end token positions.
        if `start_positions` or `end_positions` is `None`:
            Outputs a tuple of start_logits, end_logits which are the logits respectively for the start and end
            position tokens of shape [batch_size, sequence_length].

    Example usage:
    ```python
    # Already been converted into WordPiece token ids
    input_ids = torch.LongTensor([[31, 51, 99], [15, 5, 0]])
    input_mask = torch.LongTensor([[1, 1, 1], [1, 1, 0]])
    token_type_ids = torch.LongTensor([[0, 0, 1], [0, 1, 0]])

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)

    model = LayoutLMv3ForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    def __init__(self, config):
        super(BertEncoder2Decoder, self).__init__(config)
        self.layoutlmv3_encoder = LayoutLMv3Model(config)
        self.layoutlmv3_decoder = BertDecoderModel(config)

        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.config = config
        self.qgen_outputs = nn.Linear(config.hidden_size, config.vocab_size)
        self.apply(self.init_bert_weights)
        self.MASKID = 103
        self.PADID = 0
        self.vocab_size = config.vocab_size
        self.iscuda = True
        self.attention_over_attentions = nn.Linear(config.hidden_size, 12 * 12 ) # config.n_heads * config.n_layers
        self.non_copy_prob = nn.Linear(config.hidden_size, 1)
        self.linear_copy_probs = nn.Linear(config.hidden_size, config.hidden_size ) # config.n_heads * config.n_layers


    '''
    @classmethod
    def from_pretrained (cls, pretrained_model_name_or_path_1, pretrained_model_name_or_path_2, *inputs, **kwargs):

        pretrained_model_name_or_path = pretrained_model_name_or_path_1

        cache_dir = kwargs.get('cache_dir', None)
        kwargs.pop('cache_dir', None)
        from_tf = kwargs.get('from_tf', False)
        kwargs.pop('from_tf', None)

        if pretrained_model_name_or_path in PRETRAINED_MODEL_ARCHIVE_MAP:
            archive_file = PRETRAINED_MODEL_ARCHIVE_MAP[pretrained_model_name_or_path]
        else:
            archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            logger.error(
                "Model name '{}' was not found in model name list ({}). "
                "We assumed '{}' was a path or url but couldn't find any file "
                "associated to this path or url.".format(
                    pretrained_model_name_or_path,
                    ', '.join(PRETRAINED_MODEL_ARCHIVE_MAP.keys()),
                    archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading archive file {}".format(archive_file))
        else:
            logger.info("loading archive file {} from cache at {}".format(
                archive_file, resolved_archive_file))
        tempdir = None
        if os.path.isdir(resolved_archive_file) or from_tf:
            serialization_dir = resolved_archive_file
        else:
            # Extract archive to temp dir
            tempdir = tempfile.mkdtemp()
            logger.info("extracting archive file {} to temp dir {}".format(
                resolved_archive_file, tempdir))
            with tarfile.open(resolved_archive_file, 'r:gz') as archive:
                archive.extractall(tempdir)
            serialization_dir = tempdir
        # Load config
        config_file = os.path.join(serialization_dir, CONFIG_NAME)
        if not os.path.exists(config_file):
            # Backward compatibility with old naming format
            config_file = os.path.join(serialization_dir, BERT_CONFIG_NAME)
        config = BertConfig.from_json_file(config_file)

        #logger.info("Model config {}".format(config))
        model = cls(config, *inputs, **kwargs)
        #Load encoder and decoder from pretrained models
        model.bert_encoder.from_pretrained(pretrained_model_name_or_path_1, *inputs, **kwargs) 
        print("loaded BERT encoder..")
        #input("loaded BERT encoder..")
        model.bert_decoder.from_pretrained(pretrained_model_name_or_path_2, *inputs, **kwargs)
        #input("loaded BERT decoder")
        print("loaded BERT decoder..")
        return model
    '''

    def get_copy_probs2(self,decoder_output, encoder_output, dec2enc_mask,src_seq): 

        # Use sequence_output to choose the self_attentions wisely 
        transformed_decoder_output = self.linear_copy_probs(decoder_output)
        affinity =  torch.bmm(transformed_decoder_output, encoder_output.transpose(1,2))  

        mask = (1-dec2enc_mask) * (-10000)
        affinity = affinity + mask
        probs = nn.Softmax(dim=-1)(affinity)

        # src_seq has the source vocab ids
        src_to_vocab_map=torch.zeros(src_seq.size(0), dec2enc_mask.size(2), self.config.vocab_size) #batch_size x seq_size x vocab_size

        if self.iscuda==True:
          src_to_vocab_map = src_to_vocab_map.cuda()

        src_to_vocab_map.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(probs, src_to_vocab_map) # copy scores

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def get_copy_probs3(self,sequence_output, self_attentions,para_mask, src_seq):

        # Use sequence_output to choose the self_attentions wisely 
        attention_square = self.attention_over_attentions(sequence_output) # batch_size x seq_size x (12*12)
        attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)

        self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size
        self_attentions = self_attentions.transpose(1,2) # batch_size x seq_size x (12*12) x seq_size

        # The below does weighted average over all the attentions of all the layers and heads

        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 

        attention = attention * para_mask + 1e-10

        attention = attention / attention.sum(-1).unsqueeze(-1)

        # src_seq has the source vocab ids
        copy_probs=torch.zeros(src_seq.size(0), attention.size(2), self.config.vocab_size) #batch_size x seq_size x vocab_size


        if self.iscuda==True:
          copy_probs = copy_probs.cuda()

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(attention, copy_probs) # copy scores

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def get_copy_probs(self,sequence_output, self_attentions,src_seq):

        # Use sequence_output to choose the self_attentions wisely 
        attention_square = self.attention_over_attentions(sequence_output) # batch_size x seq_size x (12*12)
        attention_square = attention_square.unsqueeze(2) #batch_size x seq_size x 1 x (12*12)

        self_attentions = torch.cat(self_attentions,dim=1) # batch_size x (12*12) x seq_size x seq_size
        self_attentions = self_attentions.transpose(1,2) # batch_size x seq_size x (12*12) x seq_size

        # The below does weighted average over all the attentions of all the layers and heads

        attention = torch.matmul(attention_square,self_attentions).squeeze(2) # batch_size x seq_size x seq_size 

        # src_seq has the source vocab ids
        copy_probs=torch.zeros(src_seq.size(0), attention.size(2), self.config.vocab_size) #batch_size x seq_size x vocab_size


        if self.iscuda==True:
          copy_probs = copy_probs.cuda()

        copy_probs.scatter_(2,src_seq.unsqueeze(2), 1) #scatters vocab
        copy_probs = torch.bmm(attention, copy_probs) # copy scores

        #Normalize
        # copy_probs = copy_probs / copy_probs.sum(-1).unsqueeze(-1) # pa neccessaire
        return copy_probs

    def forward(self, input_ids, output_ids, input_token_type_ids=None, output_token_type_ids=None, input_mask=None, output_mask=None, start_positions=None, end_positions=None,decoder_mask=True,evaluate=False,copymethod=1):



        #Encode the input
        answer_phrase_ids = torch.zeros_like(input_ids)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,start_positions.unsqueeze(-1),1)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,(end_positions+1).unsqueeze(-1),-1)
        answer_phrase_ids = answer_phrase_ids.float().matmul(torch.triu(torch.ones(answer_phrase_ids.size(1), answer_phrase_ids.size(1)) ).cuda() ).long() # segment_id 1 for answer phrase

        input_mask_ids = (input_ids != self.PADID).unsqueeze(2).float()
        input_mask = torch.bmm(input_mask_ids, input_mask_ids.transpose(1,2))

        encoder_output, pooled_encoder_output, self_attns = self.layoutlmv3_encoder(input_ids, input_token_type_ids + answer_phrase_ids, input_mask, output_all_encoded_layers=False,decoder_mask=decoder_mask,attn=True)

        #Decode the output
        output_mask_ids = (output_ids != self.PADID).unsqueeze(2).float()
        output_mask = torch.bmm(output_mask_ids, output_mask_ids.transpose(1,2))
        output_mask = output_mask * torch.tril(torch.ones_like(output_mask))

        prediction_ids = output_ids
        prediction_ids[:,0]=self.PADID
        prediction_ids = torch.cat((prediction_ids[:,1:],prediction_ids[:,0:1]),-1) # batch_size x seq_length

        dec2enc_mask = torch.bmm(output_mask_ids, input_mask_ids.transpose(1,2))

        decoder_output, pooled_decoder_output, attns = self.layoutlmv3_decoder(output_ids, encoder_output, output_mask, dec2enc_mask, output_token_type_ids, output_all_decoded_layers=False,attn=True)

        if copymethod > 0:
          p = torch.sigmoid(self.non_copy_prob(decoder_output))
          qgen_logits = p * self.cls.predictions(decoder_output)

          if copymethod == 1:
            #print("no copy method 1")
            copy_qgen_logits = (1-p) * self.get_copy_probs(decoder_output,attns,input_ids)
          elif copymethod == 2:
            #print("no copy method 2")
            copy_qgen_logits = (1-p) * self.get_copy_probs2(decoder_output,encoder_output, dec2enc_mask,input_ids)
          elif copymethod == 3:
            #print("no copy method 3")
            copy_qgen_logits = (1-p) * self.get_copy_probs(decoder_output,attns,para_mask, input_ids)
            #copy_qgen_logits = (1-p) * self.get_copy_probs3(decoder_output,encoder_output, dec2enc_mask,input_ids)
          qgen_logits = qgen_logits + copy_qgen_logits
        else:
          #print("no copy method")
          qgen_logits =  self.cls.predictions(decoder_output)

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

        #Question prediction
        if not evaluate: #start_positions is not None and end_positions is not None:
         loss_fct = CrossEntropyLoss(ignore_index=0)
         loss = loss_fct(qgen_logits.view(-1,self.vocab_size), prediction_ids.view(-1))
         return loss
        else:
         return qgen_logits

class AttMat(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            scores = x.matmul(x.transpose(-1,-2)) #
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias

