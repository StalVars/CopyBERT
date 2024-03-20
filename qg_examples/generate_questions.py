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
"""Run BERT on SQuAD."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import json

import math
import os
import random
import sys
from io import open
import sys
import re

sys.path.insert(0,'./')
import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import gc, psutil
from torch.utils.data.distributed import DistributedSampler
from qg_utils.utils_qg import *
from tqdm import tqdm, trange
import torch.nn.functional as F

'''
from pytorch_pretrained_bert_local.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert_local.modeling import BertForQuestionGeneration, BertConfig
from pytorch_pretrained_bert_local.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert_local.modeling import BertForQuestionLM

from pytorch_pretrained_bert_local.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert_local.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)
'''

#from transformers.tokenization_bert import whitespace_tokenize
from transformers.models.bert.tokenization_bert import whitespace_tokenize
from qg_modules.optimization import BertAdam
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
#from transformers.tokenization_bert import *

from transformers.models.bert.tokenization_bert import *
from transformers.models.roberta.tokenization_roberta import *

#from transformers.tokenization_roberta import *

from transformers import *
from qg_modules.modeling_bert import BertForQuestionGeneration
from qg_modules.modeling_roberta import RobertaForQuestionGeneration


#falcon
import falcon
import socket
from waitress import serve

from qg_utils.beam_search import BeamSearch

#from nltk.translate.bleu_score import corpus_bleu

import datetime

MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionGeneration, BertTokenizer, "[CLS]","[MASK]","[SEP]"),
    "span": (BertConfig, BertForQuestionGeneration, BertTokenizer, "[CLS]","[MASK]","[SEP]"),
    "roberta": (RobertaConfig, RobertaForQuestionGeneration, RobertaTokenizer,"<s>","<mask>","</s>"),
}
    #"longformer": (LongformerConfig, LongformerForQuestionGeneration, LongformerTokenizer,"<s>","<mask>","</s>"),
    #"bart": (BartConfig, BartForQuestionGeneration, BartTokenizer, "<s>","<mask>","</s>"),
    #"albert": (AlbertConfig, AlbertForQuestionGeneration, AlbertTokenizer, "[CLS]","[MASK]","[SEP]"),
    #"xlnet": (XLNetConfig, XLNetForQuestionGeneration, XLNetTokenizer, "<s>","<mask>","</s>"),
'''
'''

now = datetime.datetime.now()
filetag = str(now.hour)+"."+str(now.minute)+"."+str(now.second)

'''
'''

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

import logging
logger = logging.getLogger(__name__)






class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 doc_span_index,
                 tokens,
                 token_to_orig_map,
                 token_is_max_context,
                 input_ids,
                 input_mask,
                 segment_ids,
                 start_position=None,
                 end_position=None,
                 is_impossible=None):
        self.unique_id = unique_id
        self.example_index = example_index
        self.doc_span_index = doc_span_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.token_is_max_context = token_is_max_context
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

def read_enc2dec_examples(input_file, is_training, version_2_with_negative):

    if is_training:
      enc_file = input_file+"/from.txt"
      dec_file = input_file+"/to.txt"
    else:
      enc_file = input_file+"/from.dev.txt"
      dec_file = input_file+"/to.dev.txt"

    input_lines=[]
    output_lines=[]
    with open(enc_file, "r", encoding='utf-8') as reader:
        for line in reader:
             line=line.strip()
             input_lines.append(line)
    with open(dec_file, "r", encoding='utf-8') as reader:
        for line in reader:
             line=line.strip()
             output_lines.append(line)


 
    examples=[]
    for idx in range(len(input_lines)):

        start_position = 0 
        end_position = 0 
        orig_answer_text = ""
        is_impossible=False

        if len(output_lines[idx])!=0 and len(input_lines[idx])!=0: #Ensure the input and output are not empty strings
          example = SquadExample(
                    qas_id=idx,
                    question_text=input_lines[idx],
                    doc_tokens=output_lines[idx],
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
          examples.append(example)
    return examples


def translate_batch(
            model,
            tokenizer,
            batch,
            max_length,
            args,
            qamodel,
            qlm_model=None,
            min_length=0,
            ratio=0.,
            n_best=1,
            beam_size=10,
            return_attention=False,split=False):

        # TODO: support these blacklisted features.

        # (0) Prep the components of the search.
        #beam_size = 5 #20
        batch_size = 1
        n_best = args.n_best_size

        # (1) Run the encoder on the src.
        input_ids, segment_ids, input_mask, start_positions, end_positions, references = batch
        #qgen_logits = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod,evaluate=True)

        results = {
            "predictions": None,
            "scores": None,
            "attention": None,
            #"batch": batch,
            "gold_score": 0}

        # (2) Repeat src objects `beam_size` times.
        # We use batch_size x beam_size
        #memory_bank = tile(memory_bank, beam_size, dim=1)
        #mb_device = memory_bank.device
        #memory_lengths = tile(src_lengths, beam_size)


        cls_id,sep_id = tokenizer.convert_tokens_to_ids([cls_token,sep_token])

        # (0) pt 2, prep the beam object
        beam = BeamSearch(
            beam_size,
            tokenizer,
            batch_size=batch_size,
            n_best=n_best,
            min_length=min_length,
            max_length=max_length,
            block_ngram_repeat=True,bos=cls_id, eos=sep_id)

        model.eval()

        log_probs = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod,evaluate=True)
        if qlm_model is not None:
            lm_logits = qlm_model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod,evaluate=True)
            lm_log_probs = torch.log(torch.softmax(lm_logits,dim=-1))
            print("reducing log_probs according to second model")
            log_probs = log_probs - 0.2 * lm_log_probs

        log_probs = log_probs[:,-1,:].unsqueeze(1) # batch_size x 1 x vocab_size
        dummy_beam_probs = torch.zeros_like(log_probs).expand(-1,beam_size-1,-1) 
                                                       # batch_size x (beam_size-1)x vocab_Size
        log_probs = torch.cat((log_probs, dummy_beam_probs),1).view(beam_size*batch_size,-1) # (batch_size*beam_size x vocab_size)
        beam_input_ids  = input_ids.unsqueeze(1).expand(-1,beam_size,-1).view(beam_size*batch_size,-1)
        beam_segment_ids  = segment_ids.unsqueeze(1).expand(-1,beam_size,-1).view(beam_size*batch_size,-1)
        beam_input_mask  = input_mask.unsqueeze(1).expand(-1,beam_size,-1).view(beam_size*batch_size,-1)
        beam.assign_whats_alive(input_ids, segment_ids, input_mask, start_positions, end_positions)

        with torch.no_grad():
          for step in range(max_length):
            #decoder_input = beam.current_predictions
            attn=None
            beam.advance(log_probs, attn)

            any_beam_is_finished = beam.is_finished.any()
            #=1 if there is some eos in the beam, i.e [SEP]

            #To print beam
            #print([ tokenizer.convert_ids_to_tokens([beam.alive_seq[j][i].item() for i in range(beam.alive_seq.size(1)-1) if beam.alive_segment[j][i] > -2 ]) for j in range(beam.alive_seq[:,1:].size(0))])
            #print( beam.alive_segment)
            #input("beam progress")
            #print([ tokenizer.convert_ids_to_tokens([beam.alive_seq[j][i].item() for i in range(beam.alive_seq.size(1)-1) if beam.alive_segment[j][i] == 1 ]) for j in range(beam.alive_seq[:,1:].size(0))])
            #input("segments 1")

            if any_beam_is_finished:
                beam.update_finished()
                #print("update finished")
                if beam.done:
                    break

            input_ids, segment_ids, input_mask, start_positions, end_positions = beam.get_whats_alive()

            # split here for smaller batch sizes
            if split:
              batch_size = input_ids.size(0) 
              #log_probs=torch.tensor().cuda()
              log_probs1 = model(input_ids[:5], segment_ids[:5], input_mask[:5], start_positions[:5], end_positions[:5],copymethod=args.copymethod,evaluate=True)
              log_probs2 = model(input_ids[5:], segment_ids[5:], input_mask[5:], start_positions[5:], end_positions[5:],copymethod=args.copymethod,evaluate=True)
              log_probs = torch.cat((log_probs1,log_probs2),0)
            else:
              log_probs = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod,evaluate=True)

            if qlm_model is not None:
              lm_logits = qlm_model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=0,evaluate=True) 
              lm_log_probs = torch.log(torch.softmax(lm_logits,dim=-1))
              alpha = 0.2/(step+1) if step < 4 else 0
              log_probs = log_probs - alpha * lm_log_probs
              #log_probs = lm_log_probs

            log_probs = log_probs[:,-1,:]# (batch_size*beam_size) x vocab_size

            if any_beam_is_finished:
                # Reorder states.
                pass

        #print((beam.segments).sum())
        #results["scores"] = beam.scores
        #print([sum(beam.segments[0][i]) for i in range(len(beam.scores[0])) ])
        #input("")
        results["scores"] = [[ beam.scores[0][i]/beam.segments[0][i].sum() for i in range(len(beam.scores[0])) ]]
        results["predictions"] = beam.predictions
        results["segments"] = beam.segments
        #results["attention"] = beam.attention
        return results


def sample_sequence(model, tokenizer, inputs, length, args, num_samples=5, temperature=1, top_k=0, top_p=0.0, is_xlnet=False, device='cpu'):

    #inputs = [ input_ids, segment_ids,input_mask, start_positions, end_positions ]

    generated = inputs[0]
    segment_ids = inputs[1] 
    input_mask = inputs[2] 
    start_positions = inputs[3] 
    end_positions = inputs[4] 
    references = inputs[5] 
    scores = [0 for i in range(num_samples) ]
    preds=[]
    contexts=[]
    segments=[]
    answer_phrases=[]
    results=dict()

    for current_sample in range(num_samples):
     with torch.no_grad():
        for _ in trange(length):
            #inputs = {'input_ids': generated}
            #print("generated size", generated.size())

            outputs = model(generated, segment_ids, input_mask, start_positions, end_positions,evaluate=True,copymethod=args.copymethod)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)

            #print("outputs dim", outputs.dim())
            #print("outputs size", outputs.size())
            next_token_logits = outputs[0,-1,:] / temperature
            #print(next_token_logits.dim())
            #input("next tokens dim")

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = torch.log(F.softmax(filtered_logits, dim=-1))
            next_token = torch.multinomial(F.softmax(filtered_logits, dim=-1), num_samples=1)
            scores[current_sample] +=   probs[next_token]
            if next_token.item() == tokenizer.convert_tokens_to_ids(["[SEP]"])[0]:
                #print("next_token", next_token.item())
                #Found end-of-question [SEP]
                break;

            generated = torch.cat((generated, next_token.unsqueeze(0)), dim=1)
            segment_ids = torch.cat((segment_ids, torch.ones_like(next_token.unsqueeze(0))),dim=1)
            input_mask = torch.cat((input_mask, torch.ones_like(next_token.unsqueeze(0))),dim=1)
     input_ids = generated

     preds += [input_ids]
     #preds += [" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] != 0][1:])) for b in range(input_ids.size(0)) ] 
     contexts +=  [" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0])) for b in range(input_ids.size(0)) ]
     answer_phrases += [ " ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0 \
             and  (s >= start_positions.item() and s <= end_positions.item()) ])) for b in range(input_ids.size(0))  ]
     segments.append(segment_ids)


     '''
     print("######")
     print("#Context:\n",context)
     print("#Answer Phrase:\n",answer_phrase, start_positions.item(), end_positions.item())
     print("#Preds:\n",preds)
     print("#Reference\n",references[0])
     print("#######")
     ref_file.write(references[0]+"\n")
     pred_file.write(preds[0]+"\n")
     context_file.write(answer_phrase[0]+"|"+context[0]+"\n")
     predlog_file.write("######\n")
     predlog_file.write("#Context:\n"+context[0]+"\n")
     predlog_file.write("#Answer Phrase:\n"+answer_phrase[0] +";"+ str(start_positions.item())+"-"+str(end_positions.item()) )
     predlog_file.write("#Preds:\n"+preds[0]+"\n")
     predlog_file.write("#Reference\n"+references[0]+"\n")
     predlog_file.write("#######\n")
     '''

    #ngram_match, denom = calculate_ngram_match([references[0]], [preds[0]],n=2)
    #input("sample_sequence")

    #return ngram_match, denom  
    results["scores"] = scores
    results["predictions"] = preds
    results["segments"] = segments 
    return results


def print_and_file(*text):
    print(*text)
    #print(text)
    predlog_file.write(text[0]+"\n")
    predlog_file.flush()
    #input("ok?")

def print_beam(tokenizer, preds, segments,scores=None,start_indices=None,end_indices=None,secondary_scores=None):
    return_pred=""
    best_score=0

    # For List
    #if type(preds) == list or type(: 
    return_preds=[]
    return_scores=[]
    for j in range(len(preds)):

          if True:
            return_pred = " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])[1:-1])
            return_preds.append(return_pred)
            return_scores.append(scores[j])
          if scores is None:
            print_and_file("%d %s" % (j," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )
          else:
            if secondary_scores is None:
             if start_indices is not None:
                 print_and_file("%d %f %s | Answer: %s" % (j,scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])), " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(start_indices[j].item(), end_indices[j].item()+1)  ]))  ))
             else:
               print_and_file("%d %f %s" % (j,scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )
            else:
             if start_indices is not None:
                 print_and_file("%d %f %f %s | Answer: %s" % (j,scores[j],secondary_scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])), " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(start_indices[j].item(), end_indices[j].item()+1)  ]))  ))
             else:
               print_and_file("%d %f %f %s" % (j,scores[j],secondary_scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )
    return return_preds, return_scores

class QuestionGenerator():

    def __init__(self,model, tokenizer, args, device):

        self.model = model
        self.tokenizer  = tokenizer
        self.args = args
        self.device = device

    def on_post(self, req, resp):
        body = req.stream.read()
        data=pickle.loads(body)

        context = data["context"]
        print(data)

        preds = self.generate_questions(self.model, self.tokenizer, self.args, self.device, context=context) 

        if type(preds) != list:
            preds=[]

        print("Got context:" , context)
        print("Predicted:" , "\n".join( str(enumerate(preds)) ))

        resp.body = pickle.dumps(preds)



    def generate_questions(self, model, tokenizer, args, device, context="sample context", no_of_datapoints=-1,train=False,qlm_model=None,test_examples=None):

        predict_batch_size = 1
        data_file = args.predict_file
        qamodel=None
        model.eval()


        '''
        if args.read_enc2dec_file:
          eval_dev_examples,eval_test_examples = read_enc2dec_examples(
            input_file=data_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
        else:
           eval_test_examples = read_squad_examples(
            input_file=data_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
        '''


        #context=input("Enter Input Context:")
        #context="And the actual reality, if you actually train a phrase translation table as we discussed last lecture on the European Parliament proceedings,.you get for this particular sentence, 2007 or  2007 matching phrase pairs..For each phrase translation probability, especially for common words, there's a huge tale of junk.. So it doesn't make sense to say, well, okay, let's just ignore all that junk at the end..We can only look at the top 20 translations per phrase that's probably good enough..And then in this case, still 22 translation options remain.. Given why this is very short sentence, average news, paper sentences, maybe 20, 30 words, this one is six words..And you have 200 translation options to deal with..These are the ones you should have picked..So these are the ones that the machine has to pick..And not only does the machine has to pick these, it also has to pick them in the right order..And making this picking them in the right order  makes this kind of a naive setup, an exponential problem..This exponential number of orders, how you can go through the sentence..So what we're going to do, since we're not going to solve it in P complete problem here, is to come up with a heuristic algorithm.. " 
        end_position=len(context.split()) - 3
        example = SquadExample(
                    qas_id="01",
                    question_text="question",
                    doc_tokens=context.strip().split(" "),
                    orig_answer_text="",
                    start_position=0,
                    end_position=end_position,
                    is_impossible=False)
        examples = [example]
        eval_features = convert_examples_to_features(
            examples=examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=1390,
            max_query_length=args.max_query_length,
            is_training=True,no_question=True,is_generate=True)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)
        all_tokens = [f.tokens for f in eval_features]
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=predict_batch_size)

    

        with torch.no_grad():
         for input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):

  
            if example_indices.item() < args.genfrom:
                continue
            if example_indices.item() >= args.gento:
                continue

            #if example_indices.item() not in print_only:
            #    continue

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            #Batch size = 1, remove the pads
            input_length = (input_mask!=0).sum(-1).item() 
            input_ids = input_ids[:,:input_length]
            input_mask = input_mask[:,:input_length]
            segment_ids = segment_ids[:,:input_length]
            #start_positions = start_positions[:,:input_length]
            #end_positions = end_positions[:,:input_length]

            references = [ re.sub("[     ]*\[SEP\]","",re.sub(".*\[MASK\]",""," ".join(all_tokens[i]))) for i in example_indices ]
            #references = [ re.sub(".*\[MASK\]",""," ".join(all_tokens[i])) for i in example_indices ]
            inputs = [ input_ids, segment_ids,input_mask, start_positions, end_positions, references ]

            '''
            print(input_ids.size(), segment_ids.size(), input_mask.size(), start_positions.size(), end_positions.size(), len(references))
            print(input_ids[0])
            input("input ids")
            print("#input "," ".join(tokenizer.convert_ids_to_tokens([ input_ids[0][i].item() for i in range(input_ids[0].size(0)) if input_ids[0][i].item() !=0 ])))
            print("#answer "," ".join(tokenizer.convert_ids_to_tokens([ input_ids[0][i].item() for i in range(input_ids[0].size(0)) if i >=start_positions[0].item() and i<=end_positions[0].item() ])))
            print(start_positions, end_positions)
            input("input")
            print(segment_ids[0])
            input("segment ids")
            print(input_mask[0])
            input("input_mask")
            '''

            #predlog_file.write("---"+" ".join([str(i) for i in example_indices ])+"\n")
            #gen_tup  = sample_sequence(model, tokenizer, inputs, 20, args)
            gen_tup = translate_batch(model, tokenizer, inputs, 20, args,qamodel, beam_size=args.beam_size, qlm_model=qlm_model)
            preds = gen_tup['predictions']
            segments = gen_tup['segments']
            scores = gen_tup['scores']

            #sort scores
            sorted_indices = [i[0] for i in sorted(enumerate(scores[0]), key=lambda x:x[1].item(),reverse=True)]

            preds = [[ preds[0][ind] for ind in sorted_indices ]]
            scores = [[ scores[0][ind] for ind in sorted_indices ]]
            segments = [[ segments[0][ind] for ind in sorted_indices ]]
            

            new_start_indices=None
            new_end_indices=None
            prev_scores = None


            if args.rerank:
              prev_scores = scores[0]
              preds,segments,scores,new_start_indices, new_end_indices = rerank(preds, segments, scores,input_ids,input_mask,qamodel,start_positions,end_positions,top=5) # only rerank top 5 
            else:
              new_start_indices = torch.zeros(len(preds[0])).long().cuda() + start_positions - 1 # -1 because [CLS] token is removed in the preds
              new_end_indices = torch.zeros(len(preds[0])).long().cuda() + end_positions - 1 # -1 because [CLS] token is removed in the preds
            

            print_and_file("##### Context ###")
            print_and_file([" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0])) for b in range(input_ids.size(0)) ][0])
            print_and_file("##### N-Best Predictions ###")
            #To print beam

            best_preds, best_scores = print_beam(tokenizer, preds[0], segments[0],scores=scores[0],start_indices = new_start_indices,end_indices=new_end_indices,secondary_scores=prev_scores)

            trimmed_preds=[]
            for best_pred_text in best_preds:
             best_pred_text = re.sub("( \[SEP\])+","",best_pred_text)
             best_pred_text = re.sub("( \[PAD\])+","",best_pred_text)
             best_pred_text = re.sub("( \?)+"," ?",best_pred_text)
             trimmed_preds.append(best_pred_text)

            print_and_file("%s: %s"%("#Reference",references[0]))
            print_and_file("#Predicted: %s " %(trimmed_preds[0]))
            #print(best_score)
            pred_file.write(best_pred_text+"\n")
            ref_file.write(references[0]+"\n")
            pred_file.flush()
            ref_file.flush()

            return best_preds


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--qlm_model", default=None, type=str, required=False,
                        help="a language model on questions- used while decoding for pmi")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--genfrom", default=-1, type=int,
                        help="generate from the datapoint (.)")
    parser.add_argument("--gento", default=99999, type=int,
                        help="generate till the datapoint (.)")
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--doc_stride", default=128, type=int,
                        help="When splitting up a long document into chunks, how much stride to take between chunks.")
    parser.add_argument("--max_query_length", default=64, type=int,
                        help="The maximum number of tokens for the question. Questions longer than this will "
                             "be truncated to this length.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--rerank", action='store_true', help="rerank while generating")
    parser.add_argument("--do_predict", action='store_true', help="to generate on dev set.")
    parser.add_argument("--testset", action='store_true', help="in-addition to do_predict: to run generate on the test set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--read_enc2dec_file", action='store_true', help="Whether to run training.")
    parser.add_argument("--read_qg", action='store_true', help="Whether to run training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--n_best_size", default=20, type=int,
                        help="The total number of n-best predictions to generate in the nbest_predictions.json "
                             "output file.")
    parser.add_argument("--beam_size", default=5, type=int,
                        help="The beam width")
    parser.add_argument("--copymethod", default=1, type=int,
                        help="copymethod=0 means no copymechanism;copymethod=1(default) means use selfattentions for copy"
                              "copymethod=2 means use a separate copymechanism")
    parser.add_argument("--max_answer_length", default=30, type=int,
                        help="The maximum length of an answer that can be generated. This is needed because the start "
                             "and end predictions are not conditioned on one another.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed. "
                             "A number of warnings are expected for a normal SQuAD evaluation.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--training_sanity_check_steps',
                        type=int,
                        default=10000,
                        help="Number of updates steps for sanity check on training.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--version_2_with_negative',
                        action='store_true',
                        help='If true, the SQuAD examples contain some that do not have an answer.')
    parser.add_argument('--null_score_diff_threshold',
                        type=float, default=0.0,
                        help="If null_score - best_non_null is greater than the threshold predict null.")
    parser.add_argument('--server_ip', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="Can be used for distant debugging.")
    parser.add_argument('--genset', type=str, default='test', help="generation set")
    parser.add_argument("--data_cache_name", default=None, type=str, required=True,
                        help="data cache name")
    parser.add_argument("--model_type", default=None, type=str, required=True,
                        help="bert/roberta/..")
    parser.add_argument('--gen_during_train', action='store_true', help="generate few questions during evaluation of training")
    parser.add_argument('--tag', type=str, default='0', help="tag for prediction")
    parser.add_argument('--answer_at_end_of_context', action='store_true', help="answer to be added at the end of context")

    args = parser.parse_args()
    print(args)

    global ref_file 
    global pred_file 
    global context_file 
    global predlog_file 
    global log_file
    global filetag

    if args.qlm_model is not None:
       pmi="True"
    else:
       pmi="False"
    if args.do_train:
       filetag = "_train_cp_"+str(args.copymethod)+"_"+re.sub("/","_",str(args.output_dir))+"."+filetag
    elif args.do_predict:
       filetag = "_"+re.sub("/","_",str(args.output_dir))+"_beam_"+str(args.beam_size)+"_nbest_"+str(args.n_best_size)+"_cp_"+str(args.copymethod)+"."+filetag

       filetag = filetag +"."+args.tag

       ref_file = open(args.output_dir+"/references."+filetag+".txt","w")
       pred_file = open(args.output_dir+"/preds."+filetag+".txt","w")
       context_file = open(args.output_dir+"/contexts."+filetag+".txt","w")
       predlog_file = open(args.output_dir+"/qgeneration."+filetag+".txt","w")
    log_file = open("qg_logs/log."+filetag+".txt","w")

    if args.gen_during_train:
     ref_file = open("generations/references."+filetag+".txt","w")
     pred_file = open("generations/preds."+filetag+".txt","w")
     context_file = open("generations/contexts."+filetag+".txt","w")
     predlog_file = open("generations/qgeneration."+filetag+".txt","w")

    log_file.write(str(args)+"\n")

    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')

    logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%m/%d/%Y %H:%M:%S',
                        level = logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
                            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    #print("lower case",args.do_lower_case)
    #input("tokenizer case")

    global cls_token
    global mask_token
    global sep_token
    config_class, model_class, tokenizer_class, cls_token, mask_token, sep_token = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    #num_train_optimization_steps = None
    num_train_optimization_steps = 5474

    model = model_class.from_pretrained(args.bert_model,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))
    model.to(device)

    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        warmup_linear = WarmupLinearSchedule(warmup=args.warmup_proportion,
                                             t_total=num_train_optimization_steps)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        if os.path.isfile(os.path.join(args.bert_model, "optimizer.pt")):
           optimizer.load_state_dict(torch.load(os.path.join(args.bert_model, "optimizer.pt")))

    model = model_class.from_pretrained(args.bert_model) #(,do_lower_case=args.do_lower_case)
    tokenizer = tokenizer_class.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)#, do_basic_tokenize=False)

    model.to(device)
    # dummy qlm-model
    qlm_model=None
    #Sample context

    qgen=QuestionGenerator(model=model, tokenizer=tokenizer,args=args, device=device)

    # falcon api
    hostname=socket.gethostname()
    hostname=str(hostname)

    print("hostname:", hostname)

    ''' Start the falcon server '''

    api = application = falcon.API()
    api.req_options.auto_parse_form_urlencoded = True

    api.add_route('/generate', qgen)
    serve(api, host=hostname, port='1755')
    print("serving..")



if __name__ == "__main__":
    main()


