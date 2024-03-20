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

from transformers import *

from transformers.tokenization_bert import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)

from transformers.modeling_roberta import RobertaEmbeddings, RobertaModel
from transformers.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertPreTrainingHeads

logger = logging.getLogger(__name__)


class TransformersForQuestionGeneration(BertPreTrainedModel):
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

    model = BertForQuestionAnswering(config)
    start_logits, end_logits = model(input_ids, token_type_ids, input_mask)
    ```
    """
    config_class = RobertaConfig
    pretrained_model_archive_map = ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP
    base_model_prefix = "roberta"

    def __init__(self, config):
        super(TransformersForQuestionGeneration, self).__init__(config)
        #self.bert = BertModel(config)
        self.roberta = RobertaModel(config)
        # self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.qa_outputs = nn.Linear(config.hidden_size, 2)
        self.config = config
        self.cls = BertPreTrainingHeads(config)
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

        #token_type_ids = token_type_ids + answer_phrase_ids
        sequence_output, pooled_output, attns = self.roberta(input_ids=input_ids, attention_mask=extra_mask, token_type_ids=token_type_ids)


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

