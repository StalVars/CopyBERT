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

import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import NLLLoss

from transformers import LongformerConfig, LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP

from transformers.tokenization_bert import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

from transformers.modeling_roberta import RobertaEmbeddings, RobertaModel
from transformers.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertPreTrainingHeads 
from transformers.modeling_longformer import LongformerSelfAttention 
from torch.nn import functional as F

logger = logging.getLogger(__name__)

LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP = {
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



class LongformerModel(RobertaModel):
    """
    This class overrides :class:`~transformers.RobertaModel` to provide the ability to process
    long sequences following the selfattention approach described in `Longformer: the Long-Document Transformer`_by
    Iz Beltagy, Matthew E. Peters, and Arman Cohan. Longformer selfattention combines a local (sliding window)
    and global attention to extend to long documents without the O(n^2) increase in memory and compute.

    The selfattention module `LongformerSelfAttention` implemented here supports the combination of local and
    global attention but it lacks support for autoregressive attention and dilated attention. Autoregressive
    and dilated attention are more relevant for autoregressive language modeling than finetuning on downstream
    tasks. Future release will add support for autoregressive attention, but the support for dilated attention
    requires a custom CUDA kernel to be memory and compute efficient.

    .. _`Longformer: the Long-Document Transformer`:
        https://arxiv.org/abs/2004.05150

    """

    config_class = LongformerConfig
    pretrained_model_archive_map = LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "longformer"

    def __init__(self, config):
        super().__init__(config)

        if isinstance(config.attention_window, int):
            assert config.attention_window % 2 == 0, "`config.attention_window` has to be an even value"
            assert config.attention_window > 0, "`config.attention_window` has to be positive"
            config.attention_window = [config.attention_window] * config.num_hidden_layers  # one value per layer
        else:
            assert len(config.attention_window) == config.num_hidden_layers, (
                "`len(config.attention_window)` should equal `config.num_hidden_layers`. "
                f"Expected {config.num_hidden_layers}, given {len(config.attention_window)}"
            )

        for i, layer in enumerate(self.encoder.layer):
            # replace the `modeling_bert.BertSelfAttention` object with `LongformerSelfAttention`
            layer.attention.self = LongformerSelfAttention(config, layer_id=i)

        self.init_weights()

    def _pad_to_window_size(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_window: int,
        pad_token_id: int,
    ):
        """A helper function to pad tokens and mask to work with implementation of Longformer selfattention."""

        assert attention_window % 2 == 0, f"`attention_window` should be an even value. Given {attention_window}"
        input_shape = input_ids.shape if input_ids is not None else inputs_embeds.shape
        batch_size, seqlen = input_shape[:2]

        padding_len = (attention_window - seqlen % attention_window) % attention_window
        if padding_len > 0:
            logger.info(
                "Input ids are automatically padded from {} to {} to be a multiple of `config.attention_window`: {}".format(
                    seqlen, seqlen + padding_len, attention_window
                )
            )
            if input_ids is not None:
                input_ids = F.pad(input_ids, (0, padding_len), value=pad_token_id)
            if attention_mask is not None:
                attention_mask = F.pad(
                    attention_mask, (0, padding_len), value=False
                )  # no attention on the padding tokens
            if token_type_ids is not None:
                token_type_ids = F.pad(token_type_ids, (0, padding_len), value=0)  # pad with token_type_id = 0
            if position_ids is not None:
                # pad with position_id = pad_token_id as in modeling_roberta.RobertaEmbeddings
                position_ids = F.pad(position_ids, (0, padding_len), value=pad_token_id)
            if inputs_embeds is not None:
                input_ids_padding = inputs_embeds.new_full(
                    (batch_size, padding_len), self.config.pad_token_id, dtype=torch.long,
                )
                inputs_embeds_padding = self.embeddings(input_ids_padding)
                inputs_embeds = torch.cat([inputs_embeds, inputs_embeds_padding], dim=-2)

        return padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds

    #@add_start_docstrings_to_callable(LONGFORMER_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        post_attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        masked_lm_labels=None,
    ):
        r"""

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

        import torch
        from transformers import LongformerModel, LongformerTokenizer

        model = LongformerModel.from_pretrained('longformer-base-4096')
        tokenizer = LongformerTokenizer.from_pretrained('longformer-base-4096')

        SAMPLE_TEXT = ' '.join(['Hello world! '] * 1000)  # long input document
        input_ids = torch.tensor(tokenizer.encode(SAMPLE_TEXT)).unsqueeze(0)  # batch of size 1

        # Attention mask values -- 0: no attention, 1: local attention, 2: global attention
        attention_mask = torch.ones(input_ids.shape, dtype=torch.long, device=input_ids.device) # initialize to local attention
        attention_mask[:, [1, 4, 21,]] = 2  # Set global attention based on the task. For example,
                                            # classification: the <s> token
                                            # QA: question tokens
                                            # LM: potentially on the beginning of sentences and paragraphs
        sequence_output, pooled_output = model(input_ids, attention_mask=attention_mask)
        """

        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )
        padding_len, input_ids, attention_mask, token_type_ids, position_ids, inputs_embeds = self._pad_to_window_size(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            attention_window=attention_window,
            pad_token_id=self.config.pad_token_id,
        )

        '''
        attention_mask = attention_mask.unsqueeze(-1).float()
        attention_mask_3d = torch.bmm(attention_mask, attention_mask.transpose(1,2)) # batch x seqlen x seqlen 
        attention_mask  = attention_mask_3d.masked_fill((1-post_attention_mask).byte(), 0)
        '''

        # embed
        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=None,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
        )

        # undo padding
        if padding_len > 0:
            # `output` has the following tensors: sequence_output, pooled_output, (hidden_states), (attentions)
            # `sequence_output`: unpad because the calling function is expecting a length == input_ids.size(1)
            # `pooled_output`: independent of the sequence length
            # `hidden_states`: mainly used for debugging and analysis, so keep the padding
            # `attentions`: mainly used for debugging and analysis, so keep the padding
            output = output[0][:, :-padding_len], *output[1:]

        return output

class LongformerForQuestionGeneration(BertPreTrainedModel):
    config_class = LongformerConfig
    pretrained_model_archive_map = LONGFORMER_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "longformer"

    def __init__(self, config):
        super(LongformerForQuestionGeneration, self).__init__(config)
        #self.bert = BertModel(config)
        self.longformer = LongformerModel(config)
        # TODO check with Google if it's normal there is no dropout on the token classifier of SQuAD in the TF version
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.config = config
        self.cls = BertPreTrainingHeads(config) # self.longformer.embeddings.word_embeddings.weight)
        #self.qgen_outputs = nn.Linear(config.hidden_size, config.vocab_size)
        #self.qgen_outputs = self.cls.predictions # Make use of pretrained layer

        #self.apply(self.init_bert_weights)
        self.init_weights()
        self.MASKID = 50264
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
    def _get_question_end_index(self, input_ids):
        sep_token_indices = (input_ids == self.config.mask_token_id).nonzero()
        batch_size = input_ids.shape[0]

        assert sep_token_indices.shape[1] == 2, "`input_ids` should have two dimensions"
        assert (
            sep_token_indices.shape[0] == 2 * batch_size
        ), f"There should be exactly two sep(mask) tokens: {self.config.mask_token_id} in every sample for questions answering"

        return sep_token_indices.view(batch_size, 3, 2)[:, 0, 1]
    def _compute_global_attention_mask(self, input_ids):
        question_end_index = self._get_question_end_index(input_ids)
        question_end_index = question_end_index.unsqueeze(dim=1)  # size: batch_size x 1
        # bool attention mask with True in locations of global attention
        attention_mask = torch.arange(input_ids.shape[1], device=input_ids.device)
        attention_mask = attention_mask.expand_as(input_ids) < question_end_index

        attention_mask = attention_mask.int() + 1  # True => global attention; False => local attention
        return attention_mask.long()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, start_positions=None, end_positions=None,decoder_mask=True,evaluate=False, copymethod=1):


        # Save attention_mask as input_mask for later use
        input_mask = attention_mask

        '''
        global_attention_mask = self._compute_global_attention_mask(input_ids)

        if attention_mask is None:
            attention_mask = global_attention_mask
        else:
            # combine global_attention_mask with attention_mask
            # global attention on question tokens, no attention on padding tokens
            attention_mask = global_attention_mask * attention_mask
        '''

        ##### QUESTION SEGMENTS IDS = 1 #####
        question_mask_ids = (token_type_ids == 1).unsqueeze(-1).float()
        input_mask_ids = input_mask.unsqueeze(-1).float()
        #input_mask_ids = (input_ids!=0).unsqueeze(-1).float()

        #Mask begin of question
        boq_mask = (input_ids != self.MASKID).unsqueeze(-1).float()
        question_mask_ids = question_mask_ids * boq_mask

        tril = torch.tril(torch.ones(question_mask_ids.size(1),question_mask_ids.size(1))).unsqueeze(0) 
        if self.iscuda:
          tril = tril.cuda() 

        question_mask = torch.bmm(question_mask_ids, question_mask_ids.transpose(1,2)) * tril 

        # For Question Generation use below line
        prediction_ids = (input_ids.unsqueeze(-1) * question_mask_ids.long()).squeeze(-1)
        prediction_ids[:,0]=0 #self.PADID : ignored_index
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

        # attention_mask should not include future words for Question
        # token_typ_ids should contain Answer phrase

        answer_phrase_ids = torch.zeros_like(input_ids)
        '''
        '''
        #print(start_positions, end_positions)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,start_positions.unsqueeze(-1),1)
        answer_phrase_ids = answer_phrase_ids.scatter(-1,(end_positions+1).unsqueeze(-1),-1)
        triu = torch.triu(torch.ones(answer_phrase_ids.size(1), answer_phrase_ids.size(1)))
        if self.iscuda:
          triu = triu.cuda()
        answer_phrase_ids = answer_phrase_ids.float().matmul(triu).long() # segment_id 1 for answer phrase

        # For Question Generation
        #token_type_ids = token_type_ids + answer_phrase_ids

        sequence_output, pooled_output, attns = self.longformer(input_ids=input_ids, 
                                                attention_mask=input_mask,
                                                post_attention_mask=extra_mask) 


        '''
        logits = self.qa_outputs(sequence_output) 
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        '''

        if copymethod > 0:
          p = torch.sigmoid(self.p_value(sequence_output))
          qgen_logits = p * torch.softmax(self.cls.predictions(sequence_output),dim=-1)
          #if copymethod == 1:
          copy_qgen_logits = (1-p) * self.get_copy_probs(sequence_output,attns,input_ids)
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

