# coding=utf-8

'''

  This script is to impolement the loss function by : https://openreview.net/pdf?id=Bkx0RjA9tX ( GENERATIVE QUESTION ANSWERING: LEARNING TO ANSWER THE WHOLE QUESTION )  
  using CopyBERT

'''

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
from os import listdir

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
import gc, psutil
from torch.utils.data.distributed import DistributedSampler, DistributedSampler
from tqdm import tqdm, trange
import torch.nn.functional as F
import tracemalloc
import copy

sys.path.insert(0,'./')
#from qg_utils.utils_gap import (read_hotpot_examples, read_hotpotpara_examples, convert_hotpotexamples_to_features)
#from qg_utils.utils_gap import convert_hotpotexamples_to_features, HotPot_Supp_Example

from transformers import  AutoTokenizer, AutoConfig



''' For Calculating SQuAD QA results '''
from transformers.data.processors.squad import SquadResult, SquadV1Processor, SquadV2Processor

#from qg_modules.hf_squad_processors import SquadResult, SquadV1Processor, SquadV2Processor, squad_convert_examples_to_features as convert_examples_to_features

#from transformers.data.metrics.squad_metrics import (
from qg_modules.squad_metrics import (
    compute_predictions_log_probs,
    compute_predictions_logits,
    squad_evaluate,
)

'''
    AdamW,
    get_linear_schedule_with_warmup,
    squad_convert_examples_to_features,
'''

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
from qg_modules.optimization import BertAdam
from transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME

#from transformers.tokenization_bert import *
#from transformers.tokenization_roberta import *

from transformers import *
from qg_modules.modeling_bert_qg_decoder import BertForQuestionGeneration
import logging
'''
from qg_modules.modeling_roberta import RobertaForQuestionGeneration
from qg_modules.modeling_xlnet import XLNetForQuestionGeneration
from qg_modules.modeling_albert import AlBertForQuestionGeneration
from qg_modules.modeling_longformer import LongformerForQuestionGeneration
'''

from qg_utils.beam_search import BeamSearch

import nltk
nltk.download('punkt')
from nltk.translate.bleu_score import corpus_bleu

import datetime

ALLOWED_LANGS = ['en', 'es', 'de'] # Add more languages later here 
MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionGeneration, BertTokenizer, "[CLS]","[MASK]","[SEP]"),
    "span": (BertConfig, BertForQuestionGeneration, BertTokenizer, "[CLS]","[MASK]","[SEP]"),
}
'''
MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionGeneration, BertTokenizer, "[CLS]","[MASK]","[SEP]"),
    "albert": (AlbertConfig, AlBertForQuestionGeneration, AlbertTokenizer, "[CLS]","[MASK]","[SEP]"),
    "span": (BertConfig, BertForQuestionGeneration, BertTokenizer, "[CLS]","[MASK]","[SEP]"),
    "roberta": (RobertaConfig, RobertaForQuestionGeneration, RobertaTokenizer,"<s>","<mask>","</s>"),
    "xlnet": (XLNetConfig, XLNetForQuestionGeneration, XLNetTokenizer, "<s>","<mask>","</s>"),
    "longformer": (LongformerConfig, LongformerForQuestionGeneration, LongformerTokenizer,"<s>","<mask>","</s>"),
}
    #"bart": (BartConfig, BartForQuestionGeneration, BartTokenizer, "<s>","<mask>","</s>"),
'''
'''
MODEL_CLASSES = {
    "bert": (BertConfig, BertForQuestionGeneration, BertTokenizer),
    "roberta": (RobertaConfig, RobertaForQuestionGeneration, RobertaTokenizer),
    "xlnet": (XLNetConfig, XLNetForQuestionGeneration, XLNetTokenizer),
    "xlm": (XLMConfig, XLMForQuestionGeneration, XLMTokenizer),
    "distilbert": (DistilBertConfig, DistilBertForQuestionGeneration, DistilBertTokenizer),
    "albert": (AlbertConfig, AlBertForQuestionGeneration, AlbertTokenizer),
}
'''

now = datetime.datetime.now()
filetag = str(now.day)+"_"+str(now.month)+"_"+str(now.year)+"_"+str(now.hour)+"."+str(now.minute)+"."+str(now.second)

'''
ref_file = open("generations/references."+filetag+".txt","w")
pred_file = open("generations/preds."+filetag+".txt","w")
context_file = open("generations/contexts."+filetag+".txt","w")
predlog_file = open("generations/qgeneration."+filetag+".txt","w")
log_file = open("logs/log."+filetag+".txt","w")
'''

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle


''' Setup Logging '''
fileh = logging.FileHandler('qg_logs/logfile', 'a')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#formatter = logging.Formatter('%(asctime)-15s::%(levelname)s::%(filename)s::%(funcName)s::%(lineno)d::%(message)s')
fileh.setFormatter(formatter)
logger = logging.getLogger(__name__)
log = logging.getLogger()  # root logger - Good to get it only once.
for hdlr in log.handlers[:]:  # remove the existing file handlers
    if isinstance(hdlr,logging.FileHandler):
        log.removeHandler(hdlr)
log.addHandler(fileh)      # set the new handler
# set the log level to INFO, DEBUG as the default is ERROR
log.setLevel(logging.DEBUG)

#logging.basicConfig(filename='logs/logfile.txt', filemode='w')

class SquadEvalExample(object):
    """
    A single training/test example for the Squad dataset, as loaded from disk.

    Args:
        qas_id: The example's unique identifier
        question_text: The question string
        context_text: The context string
        answer_text: The answer string
        start_position_character: The character position of the start of the answer
        title: The title of the example
        answers: None by default, this is used during evaluation. Holds answers as well as their start positions.
        is_impossible: False by default, set to True if the example has no possible answer.
    """

    def __init__(
        self,
        qas_id,
        question_text,
        context_text,
        answer_text,
        doc_tokens,
        start_position_character,
        answers=[],
        is_impossible=False
    ):
        self.qas_id = qas_id
        self.question_text = question_text
        self.context_text = context_text
        self.answer_text = answer_text
        self.doc_tokens = doc_tokens
        self.is_impossible = is_impossible
        self.answers = answers
        self.start_position, self.end_position = 0, 0
        


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
                 phr_start_positions=None,
                 phr_end_positions=None,
                 is_impossible=None,
                 pretrain_id=0):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible
        self.pretrain_id = pretrain_id
        self.phr_start_positions = phr_start_positions
        self.phr_end_positions = phr_end_positions
        

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
                 is_impossible=None,
                 pretrain_id=-1,
                 query_ids=None,
                 query_input_mask=None
                 ):

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
        self.pretrain_id = pretrain_id

        self.query_ids = query_ids
        self.query_input_mask = query_input_mask

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens



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
                    question_text=output_lines[idx],
                    doc_tokens=input_lines[idx],
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible)
          examples.append(example)
    return examples
def read_lm_examples(input_file, is_training, version_2_with_negative):
    """Read a txt file into a list of SquadExample."""
    #with open(input_file, "r", encoding='utf-8') as reader:
    #    input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    # If there exists a dumped examples file for dev and test separately, load them and ignore further processing:
    input_lines=[]
    examples=[]
    with open(input_file, "r", encoding='utf-8') as reader:
        for line in reader:
             line=line.strip()
             input_lines.append(line)
    for idx in range(len(input_lines)):
        example = SquadExample(
                    qas_id=idx,
                    question_text=input_lines[idx],
                    doc_tokens="",
                    orig_answer_text="",
                    start_position=0,
                    end_position=0,
                    is_impossible=True)
        examples.append(example)
    return examples



def squad_json_2_multiple_answers_hotpot_example(input_data, args,clm):






    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    #clm - the type of Conditional Language Model (clm)
    def get_answer_starts(c,p,train=True):
        #p_re=re.sub(r"([()+.?*\$\[\]\{\}])",r"\\\1",p) # make phrase searchable
        p_re=re.escape(p) #
        starts=[m.start(0) for m in re.finditer(p_re,c)]
        if train:
          starts=starts[0:1]
        return [{"text":p,"answer_start":start} for start in starts ]


    # If there exists a dumped examples file for dev and test separately, load them and ignore further processing:
    if args.no_mask:
      start_tok="[START]"
      end_tok="[END]"
    else:
      start_tok="[MASK]"
      end_tok="[MASK]"

    examples = []
    skipped=[]
    phrases_frequency=[]
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            #if java_script(paragraph["context"]):
            #    Leave this paragraph
            #    continue

            #paragraph_text = re.sub("\[MASK\]","[MARK_0] [MARK_1]",paragraph_text) # Keep MARK_0 stead of MASK


            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True
            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            if "pretrain" in paragraph.keys():
               pretrain_id = paragraph["pretrain"]
            elif clm == "question":  
                  pretrain_id = 0
            else:
                 pretrain_id = 0 # -1


            #split paragraph texts in to sents

            #paragraph_texts = 
            paragraph_texts = nltk.tokenize.sent_tokenize(paragraph_text)
            new_paragraph_text = " ".join(paragraph_texts) #paragraph_text
            for qa in paragraph["qas"]:


                doc_tokens = []
                char_to_word_offset = []
                prev_is_whitespace = True

                for c in new_paragraph_text:
                   if is_whitespace(c):
                       prev_is_whitespace = True
                   else:
                       if prev_is_whitespace:
                           doc_tokens.append(c)
                       else:
                           doc_tokens[-1] += c
                       prev_is_whitespace = False
                   char_to_word_offset.append(len(doc_tokens) - 1)

                #if qa["answers"][0]["text"] == "[MASK]":
                #   qa["answers"][0]["text"] = "[MARK_0] [MARK_1]"


                try:
                  qas_id = qa["id"]
                except:
                  qas_id = qa["qid"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False


                
                   
                if "is_impossible" in qa.keys(): 
                  is_impossible = qa["is_impossible"]

                '''
                if (len(qa["answers"]) != 1) and (not is_impossible):
                    raise ValueError(
                        "For training, each question should have exactly 1 answer.")
                '''
                start_positions=[]
                end_positions=[]
                phrases=[]

                skip = 0
                no_of_phrases = 0

                if len(qa["answers"]) != 0:
                  unique_answers=[]
                  for answer in qa['answers']:

                    ''' if answer is he in 'the' ' - ignore ''' 
                    prev_char = new_paragraph_text[answer["answer_start"]-1:answer["answer_start"]]
                    if re.match("[a-zA-Z]",prev_char):
                        #print("%s, %s, %s,%s"%(answer["text"], prev_char, "ignoring..", new_paragraph_text[answer["answer_start"]-1:answer["answer_start"]+5]))
                        continue
                    else:
                        #print("Not ignoring %s"%(answer["text"]))
                        pass

                    #print(qa)
                    #answer = qa["answers"][0]
                    orig_answer_text = answer["text"]
                    answer_offset = answer["answer_start"]
                    answer_length = len(orig_answer_text)


                    if answer in unique_answers:
                        #print("answers found, continuing..", answer) #, unique_answers)
                        continue
                    unique_answers.append(answer)

                    if new_paragraph_text[answer_offset:answer_offset + answer_length] != orig_answer_text:
                          skip += 1
                          print("%s , %s, %s : %s/%s" % (answer_offset, len(char_to_word_offset), answer_length, new_paragraph_text[answer_offset:answer_offset + answer_length], orig_answer_text))
                          #print(skip)  
                          #print(no_of_phrases)

                          continue
                    else:
                          no_of_phrases += 1

                    start_position = char_to_word_offset[answer_offset]
                    end_position = char_to_word_offset[answer_offset + answer_length - 1]
                    # Only add answers where the text can be exactly recovered from the
                    # document. If this CAN'T happen it's likely due to weird Unicode
                    # stuff so we will just skip the example.
                    #
                    # Note that this means for training mode, every example is NOT
                    # guaranteed to be preserved.
                    actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                    cleaned_answer_text = " ".join(
                        whitespace_tokenize(orig_answer_text))
                    if actual_text.find(cleaned_answer_text) == -1:
                        logger.warning("Could not find answer: '%s' vs. '%s'",
                                       actual_text, cleaned_answer_text)
                        continue
                    start_positions.append(start_position)
                    end_positions.append(end_position)
                    phrases.append(orig_answer_text)

                else:

                    start_position = -1
                    end_position = -1
                    orig_answer_text = ""


                skipped.append(skip)
                phrases_frequency.append(no_of_phrases)
                '''
                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_positions[0],
                    end_position=end_positions[0],
                    phr_start_positions=start_positions,
                    phr_end_positions=end_positions,
                    is_impossible=is_impossible,
                    pretrain_id=pretrain_id)
                examples.append(example)
                '''

                # Defining inputs for hotpotsupp object
                #paragraph_texts =  
                #phrases =

                supp_sents = [] 
                # dummy answer
                orig_answer_text = phrases[0] 
                start_position = start_positions[0] 
                end_position = end_positions[0] 
                q_match_len = 0

                #print("#phr start positions: %s, phrases length: %s" %(len(start_positions), len(phrases)))

                example = HotPot_Supp_Example(
                    qas_id=qas_id,
                    question_text=question_text,
                    para_sents=paragraph_texts,
                    supp_sents=supp_sents,
                    start_position=start_position,
                    end_position=end_position,
                    orig_answer_text=orig_answer_text,
                    phrase=phrases,
                    phrase_start_position=start_positions,
                    phrase_end_position=end_positions,
                    q_match_len=q_match_len,
                    pretrain_id = pretrain_id,
                    is_impossible=False)

                examples.append(example)
    
    
    print("skipped phrases(max):%s\nnumber of phrases(max) : %s\n number of examples:%s\n" %(max(skipped), max(phrases_frequency), len(examples)))
    return examples


def squad_json_2_squad_example(input_data, args,clm):


    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    #clm - the type of Conditional Language Model (clm)
    def get_answer_starts(c,p,train=True):
        #p_re=re.sub(r"([()+.?*\$\[\]\{\}])",r"\\\1",p) # make phrase searchable
        p_re=re.escape(p) #
        starts=[m.start(0) for m in re.finditer(p_re,c)]
        if train:
          starts=starts[0:1]
        return [{"text":p,"answer_start":start} for start in starts ]


    # If there exists a dumped examples file for dev and test separately, load them and ignore further processing:
    if args.no_mask:
      start_tok="[START]"
      end_tok="[END]"
    else:
      start_tok="[MASK]"
      end_tok="[MASK]"

    examples = []
    for entry in tqdm(input_data, desc="reading squad json.."):
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]

            #paragraph_text = re.sub("\[MASK\]","[MARK_0] [MARK_1]",paragraph_text) # Keep MARK_0 stead of MASK

            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            if "pretrain" in paragraph.keys():
               pretrain_id = paragraph["pretrain"]
            elif clm == "question":  
                  pretrain_id = 0
            else:
                 pretrain_id = 0 # -1


            for qa in paragraph["qas"]:


                start_present = re.match(".*\[START\].*", paragraph_text)
                end_present = re.match(".*\[END\].*", paragraph_text)
                # Remove [START] and [END] tokens from json file text
                if not args.encaps_ans and (start_present or end_present):
                    paragraph_text = re.sub("\[START\]","", paragraph_text)
                    paragraph_text = re.sub("\[END\]","", paragraph_text)
                    paragraph_text = re.sub("  *"," ", paragraph_text)
                    answer_text = qa["answers"][0]["text"]
                    answer_text = re.sub("\[START\]","", answer_text)
                    answer_text = re.sub("\[END\]","", answer_text)
                    answer_text = answer_text.strip()


                    answers = get_answer_starts(paragraph_text, answer_text)
                    qa["answers"] = answers



                if args.encaps_ans and not(start_present and end_present): #True: # encapsulate - true
                 # Encapsulate mark-0 and mark-1 around the answer phrase
                 answer_start = qa["answers"][0]["answer_start"]
                 answer_text = qa["answers"][0]["text"]
                 answer_end = answer_start + len(answer_text)
                 #print(answer_start,answer_text)
                 new_paragraph_text = paragraph_text[:answer_start]
                 if answer_text == "[MASK]" and args.no_mask: 
                   answer_text = "" 
                 #qa["answers"][0]["text"] == "[MARK_0] "+answer_text+" [MARK_1]"
                 qa["answers"][0]["text"] = start_tok +" "+ answer_text + " "+end_tok #"[MARK_0] "+answer_text+" [MARK_1]"
                 #new_paragraph_text += "[MARK_0]"
                 #new_paragraph_text += answer_text
                 #new_paragraph_text += "[MARK_1]"
                 new_paragraph_text += qa["answers"][0]["text"]
                 new_paragraph_text += paragraph_text[answer_end:]

                 if args.debug:# and False:
                   print("before:",paragraph_text)
                   print("after:", new_paragraph_text)
                   print("answer:", qa["answers"][0]["text"])
                   #input("wait")

                 doc_tokens = []
                 char_to_word_offset = []
                 prev_is_whitespace = True
                 for c in new_paragraph_text:
                     if is_whitespace(c):
                         prev_is_whitespace = True
                     else:
                         if prev_is_whitespace:
                             doc_tokens.append(c)
                         else:
                             doc_tokens[-1] += c
                         prev_is_whitespace = False
                     char_to_word_offset.append(len(doc_tokens) - 1)

                #if qa["answers"][0]["text"] == "[MASK]":
                #   qa["answers"][0]["text"] = "[MARK_0] [MARK_1]"


                try:
                  qas_id = qa["id"]
                except:
                  qas_id = qa["qid"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if True: #is_training:
                   
                    if "is_impossible" in qa.keys(): 
                      is_impossible = qa["is_impossible"]

                    '''
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    '''
                    if not is_impossible or len(qa["answers"]) != 0:
                        #print(qa)
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        print(len(char_to_word_offset), answer_offset, answer_length)
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    pretrain_id=pretrain_id)
                examples.append(example)

    return examples


def squad_jsonl_2_squad_example(input_data, args,clm):


    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    #clm - the type of Conditional Language Model (clm)
    def get_answer_starts(c,p,train=True):
        #p_re=re.sub(r"([()+.?*\$\[\]\{\}])",r"\\\1",p) # make phrase searchable
        p_re=re.escape(p) #
        starts=[m.start(0) for m in re.finditer(p_re,c)]
        if train:
          starts=starts[0:1]
        return [{"text":p,"answer_start":start} for start in starts ]


    # If there exists a dumped examples file for dev and test separately, load them and ignore further processing:
    if args.no_mask:
      start_tok="[START]"
      end_tok="[END]"
    else:
      start_tok="[MASK]"
      end_tok="[MASK]"

    examples = []
    for paragraph in tqdm(input_data, desc="reading squad json.."):
            #for paragraph in entry["paragraphs"]:
            #print("entry:", entry)
            #for paragraph in entry:
            #print("paragraph:", paragraph)

            if "header" in paragraph.keys():
                continue

            paragraph_text = paragraph["context"]

            #paragraph_text = re.sub("\[MASK\]","[MARK_0] [MARK_1]",paragraph_text) # Keep MARK_0 stead of MASK

            doc_tokens = []
            char_to_word_offset = []
            prev_is_whitespace = True

            for c in paragraph_text:
                if is_whitespace(c):
                    prev_is_whitespace = True
                else:
                    if prev_is_whitespace:
                        doc_tokens.append(c)
                    else:
                        doc_tokens[-1] += c
                    prev_is_whitespace = False
                char_to_word_offset.append(len(doc_tokens) - 1)

            if "pretrain" in paragraph.keys():
               pretrain_id = paragraph["pretrain"]
            elif clm == "question":  
                  pretrain_id = 0
            else:
                 pretrain_id = 0 # -1


            for qa in paragraph["qas"]:


                start_present = re.match(".*\[START\].*", paragraph_text)
                end_present = re.match(".*\[END\].*", paragraph_text)
                # Remove [START] and [END] tokens from json file text
                if not args.encaps_ans and (start_present or end_present):
                    paragraph_text = re.sub("\[START\]","", paragraph_text)
                    paragraph_text = re.sub("\[END\]","", paragraph_text)
                    paragraph_text = re.sub("  *"," ", paragraph_text)
                    answer_text = qa["detected_answers"][0]["text"]
                    answer_text = re.sub("\[START\]","", answer_text)
                    answer_text = re.sub("\[END\]","", answer_text)
                    answer_text = answer_text.strip()


                    answers = get_answer_starts(paragraph_text, answer_text)
                    qa["answers"] = answers


                #answers = get_answer_starts(paragraph_text, answer_text)

                '''
                new_answers=[]
                for answer_text in [ a["text"] for a in qa["detected_answers"] ] : 
                  new_answers += get_answer_starts(paragraph_text, answer_text)
                '''
                
                new_answers=[]
                for answer in qa["detected_answers"]:
                    new_answers.append({"answer_start":answer["char_spans"][0][0], "text":answer["text"]})

                qa["detected_answers"] = new_answers

                if args.encaps_ans and not(start_present and end_present): #True: # encapsulate - true
                 # Encapsulate mark-0 and mark-1 around the answer phrase
                 answer_start = qa["answers"][0]["answer_start"]
                 answer_text = qa["answers"][0]["text"]
                 answer_end = answer_start + len(answer_text)
                 #print(answer_start,answer_text)
                 new_paragraph_text = paragraph_text[:answer_start]
                 if answer_text == "[MASK]" and args.no_mask: 
                   answer_text = "" 
                 #qa["answers"][0]["text"] == "[MARK_0] "+answer_text+" [MARK_1]"
                 qa["answers"][0]["text"] = start_tok +" "+ answer_text + " "+end_tok #"[MARK_0] "+answer_text+" [MARK_1]"
                 #new_paragraph_text += "[MARK_0]"
                 #new_paragraph_text += answer_text
                 #new_paragraph_text += "[MARK_1]"
                 new_paragraph_text += qa["answers"][0]["text"]
                 new_paragraph_text += paragraph_text[answer_end:]

                 if args.debug:# and False:
                   print("before:",paragraph_text)
                   print("after:", new_paragraph_text)
                   print("answer:", qa["answers"][0]["text"])
                   #input("wait")

                 doc_tokens = []
                 char_to_word_offset = []
                 prev_is_whitespace = True
                 for c in new_paragraph_text:
                     if is_whitespace(c):
                         prev_is_whitespace = True
                     else:
                         if prev_is_whitespace:
                             doc_tokens.append(c)
                         else:
                             doc_tokens[-1] += c
                         prev_is_whitespace = False
                     char_to_word_offset.append(len(doc_tokens) - 1)

                #if qa["answers"][0]["text"] == "[MASK]":
                #   qa["answers"][0]["text"] = "[MARK_0] [MARK_1]"


                try:
                  qas_id = qa["id"]
                except:
                  qas_id = qa["qid"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if True: #is_training:
                   
                    if "is_impossible" in qa.keys(): 
                      is_impossible = qa["is_impossible"]

                    '''
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    '''
                    if not is_impossible or len(qa["detected_answers"]) != 0:
                        #print(qa)
                        answer = qa["detected_answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
                        start_position = char_to_word_offset[answer_offset]
                        print(len(char_to_word_offset), answer_offset, answer_length)
                        end_position = char_to_word_offset[answer_offset + answer_length - 1]
                        # Only add answers where the text can be exactly recovered from the
                        # document. If this CAN'T happen it's likely due to weird Unicode
                        # stuff so we will just skip the example.
                        #
                        # Note that this means for training mode, every example is NOT
                        # guaranteed to be preserved.
                        actual_text = " ".join(doc_tokens[start_position:(end_position + 1)])
                        cleaned_answer_text = " ".join(
                            whitespace_tokenize(orig_answer_text))
                        if actual_text.find(cleaned_answer_text) == -1:
                            logger.warning("Could not find answer: '%s' vs. '%s'",
                                           actual_text, cleaned_answer_text)
                            continue
                    else:
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""

                example = SquadExample(
                    qas_id=qas_id,
                    question_text=question_text,
                    doc_tokens=doc_tokens,
                    orig_answer_text=orig_answer_text,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=is_impossible,
                    pretrain_id=pretrain_id)
                examples.append(example)
    return examples


def read_squad_examples_prev(input_file, is_training, version_2_with_negative,args,clm=""):


    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        json_data = json.load(reader)

    

    print("- %s -" %(input_file))

    if 'version' in json_data.keys() and json_data["version"] == "hotpot": # previously: json_data["version"] == 0
      examples = squad_json_2_multiple_answers_hotpot_example(json_data["data"], args,clm )
      version=0
    else: #elif input_data["version"] == 1.1 or anything else:
      examples = squad_json_2_squad_example(json_data["data"], args,clm )
      version=1.1
    
    print("     version:%s" %(version))

    return examples, version




def read_squad_examples( input_file, is_training, version_2_with_negative,args,clm=""):


    if re.match("^.*jsonl$", input_file):

      """Read a SQuAD json file into a list of SquadExample."""
      json_data=[]
      print("Input jsonl file:%s"%(input_file))
      with open(input_file, "r", encoding='utf-8') as reader:
          for line in reader:
             line=line.strip()
             json_line_data = json.loads(line) #reader)
             json_data.append(json_line_data)


    else:
      with open(input_file, "r", encoding='utf-8') as reader:
        json_data = json.load(reader)
    

    print("- %s -" %(input_file))
    if re.match("^.*jsonl$", input_file):
      examples = squad_jsonl_2_squad_example(json_data, args,clm)
      version=1.1
    elif 'version' in json_data.keys() and json_data["version"] == "hotpot": # previously: json_data["version"] == 0
      examples = squad_json_2_multiple_answers_hotpot_example(json_data["data"], args,clm )
      version=0
    else: #elif input_data["version"] == 1.1 or anything else:
      examples = squad_json_2_squad_example(json_data["data"], args,clm )
      version=1.1
    
    print("     version:%s" %(version))

    return examples, version


def convert_lmexamples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, 
                                 is_training,no_question=False,is_generate=False,
                                 eop_token=None):
    """Loads a data file into a list of `InputBatch`s."""

    bop_token = cls_token
    eop_token = sep_token
    unique_id = 1000000000
    num_out_of_span = 0
    features=[]

    for (example_index, example) in enumerate(examples):
       tokens = [ bop_token ] # Start with [CLS]
       tokens.append("[QUESTION]") # currently lm json only for questions
       query_tokens = tokenizer.tokenize(example.question_text)
       for (i, token) in enumerate(query_tokens): #example.doc_tokens):
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tokens.append(sub_token)
       tokens.append(eop_token)
       #print(tokens)
       #input("query tokens")

       input_ids = tokenizer.convert_tokens_to_ids(tokens) 


       if len(input_ids) > max_seq_length:
           input_ids = input_ids[:max_seq_length]
       input_mask = [ 1 for i in input_ids ]
       segment_ids = [ 1 for i in input_ids ]

       while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

       assert len(input_ids) == max_seq_length
       assert len(input_mask) == max_seq_length
       assert len(segment_ids) == max_seq_length
       features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=0, #doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=[],
                    token_is_max_context=[],
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=0,
                    end_position=0,
                    is_impossible=example.is_impossible))
       unique_id += 1

    return features


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, 
                                 is_training,no_question=False,is_generate=False,
                                 eop_token=None,args=None, max_token_length=40):
    """Loads a data file into a list of `InputBatch`s."""
    #print("Converting examples to features.. function..")
    #input("")

    in_span_unique_ids=[] 
    out_span_unique_ids=[] 

    unique_id = 1000000000
    num_out_of_span = 0

    bop_token = cls_token
    if eop_token is  None:
     eop_token = tokenizer.convert_ids_to_tokens([args.boq_id])[0]
    else:
     # let eop_token be eop_token; modify the boq_id
     args.boq_id = tokenizer.convert_tokens_to_ids([eop_token])[0]
     pass

    eoq_token = sep_token



    features = []

    print("total number of examples = ", len(examples))

    for (example_index, example) in tqdm(enumerate(examples), desc="converting examples to features.."):


        ''' should we take example index from the #example from list 'examples' or should I use example's id attribute? '''
        exempt=False

        if len(example.doc_tokens) > 1000:
            continue

        #if example_index < 908:
        #    continue

        remove_this=False
        for tok in example.doc_tokens:
           pass
            
           #if len(tok) > 15:
           #     remove_this=True

        #if remove_this:
        #    continue

        #example.doc_tokens = new_tokens
        #print("calculating..",example_index,len(examples), len(example.doc_tokens), len(example.question_text),"\r")

        if len(example.doc_tokens) > 1000:
            #print("Skipping..",example_index,len(examples), example.doc_tokens, example.question_text, "\r")
            continue

 
        #print("tokenizing query tokens")
        query_tokens = tokenizer.tokenize(example.question_text)

        #eop_token = "[GAP]"
        eop_token = tokenizer.convert_ids_to_tokens([args.boq_id])[0]
        #print("eop token chosen:%s"%(eop_token), "is_training:", is_training)

        '''
        if example.pretrain_id == 1:
            eop_token = "[GAP]"
        elif example.pretrain_id == 0:
            eop_token = "[QUESTION]"
        '''
        #print("example pretrain id %s"%(example.pretrain_id))


        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []



        ''' To add [MASK] before and after the answer phrase '''
        #example.doc_tokens = example.doc_tokens[:example.start_position] +["[MASK]"] + example.doc_tokens[example.start_position:example.end_position+1] + ["[MASK]"] + example.doc_tokens[example.end_position+2:]
        #example.end_position = example.end_position + 2
        # Add one [MASK] just-before the answer-phrase

        '''
        example.doc_tokens = example.doc_tokens[:example.start_position] +["[MASK]"] + example.doc_tokens[example.start_position:example.end_position+1] + example.doc_tokens[example.end_position+2:]
        example.end_position = example.end_position + 1
        '''

        

        #print(example.__dict__.keys())
        #print("tokenizing doc tokens")
        for (i, token) in enumerate(example.doc_tokens):

            orig_to_tok_index.append(len(all_doc_tokens))
            #if len(token) > max_token_length:
            #   break
            #print("length of token:", len(token))

            if len(token) > max_token_length:
                exempt=True
                break
                #print(token)


            try:
              sub_tokens = tokenizer.tokenize(token)
            except:
              exempt=True
              break

            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)
                #print("--", sub_token)


        # Datapoint exempted because of lengthy token or  java script or that sort
        if exempt:
            print("Exempting:", example.doc_tokens)
            continue

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            (tok_start_position, tok_end_position) = _improve_answer_span(
                all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
                example.orig_answer_text)

        #max_tokens_for_doc = max_seq_length - len(query_tokens) - 3
        # The -4 accounts for [CLS], [MASK],[MASK] and [SEP] -- if you append 'answer' to the context before question
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 4 - (tok_end_position - tok_start_position + 1) # answer words

        # The -5 accounts for [CLS], [MASK] (phrase) [MASK] [QUESTION] and [SEP] -- if you append 'answer' to the context before question

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0

        if is_generate:
         '''
         '''
         # Take the span that contains the answer:
         while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            if start_offset + length == len(all_doc_tokens):
                break
            end_offset = min(length, doc_stride)
            if tok_start_position >=start_offset and tok_end_position < start_offset + end_offset:
              doc_spans.append(_DocSpan(start=start_offset, length=length))
            start_offset += min(length, doc_stride)

         length = len(all_doc_tokens) - start_offset
         out_of_span_in_gen = False
         if length > max_tokens_for_doc:
                length = max_tokens_for_doc
                '''
                if not (tok_start_position >=start_offset and tok_end_position < start_offset + length):
                   out_of_span_in_gen = True
                   num_out_of_span += 1
                   start_offset = len(all_doc_tokens) - length
                '''
                while not (tok_start_position >=start_offset and tok_end_position < start_offset + length):
                   out_of_span_in_gen = True
                   num_out_of_span += 1
                   print("---is generate---", start_offset, length, tok_start_position, tok_end_position)
                   # check for adjacent windows of length #max_tokens_for_doc
                   if start_offset+(2*length) <= len(all_doc_tokens):
                      start_offset = start_offset + length
                   else:
                      start_offset = len(all_doc_tokens) - length # take the last #length(384?) words

         doc_spans.append(_DocSpan(start=start_offset, length=length))

        else:
         while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            #print("%s, start:%s, length:%s" %(unique_id, start_offset, length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span ) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append(bop_token)
            segment_ids.append(0)

            if not no_question and not args.reverse_seq:


             for token in query_tokens:
                tokens.append(token)
                segment_ids.append(1)
             tokens.append(eoq_token)
             segment_ids.append(1)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(0)

            ''' Add answer phrase at the end '''
            '''
            tokens.append(eop_token)
            segment_ids.append(0) 
            if not example.is_impossible:
             tokens = tokens + all_doc_tokens[tok_start_position:tok_end_position+1] + [eop_token]
             segment_ids = segment_ids + [ 0 for i in range(tok_end_position-tok_start_position+1) ]
             segment_ids.append(1) 
            else:
             tokens = tokens + all_doc_tokens[0:1] + [eop_token]
             segment_ids.append(0) #only cls_token
             segment_ids.append(1) 
            '''

            #print("eop token:", eop_token)
            tokens = tokens + [eop_token]
            segment_ids.append(1) 
            #segment_ids.append(0) # Keep eop_token's segment id as 0 instead of 1

            # Question to come after the context
            if not no_question and args.reverse_seq:
             for token in query_tokens:
                tokens.append(token)
                segment_ids.append(1)
             tokens.append(eoq_token)
             segment_ids.append(1)

            print(example_index, doc_span_index, tokens, "\n")
            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if no_question:
             for token in query_tokens:
                tokens.append(token)
             tokens.append(eoq_token)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            if args is not None and args.debug:
              new_tokens = tokenizer.convert_ids_to_tokens(input_ids)
              print(new_tokens)
              #input("new tokens")





            query_ids = tokenizer.convert_tokens_to_ids([cls_token] + query_tokens +[sep_token])
            query_input_mask = [1] * len(query_ids)

            # Zero-pad up to the sequence length.
            #print(max_seq_length, len(input_ids))
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            while len(query_ids) < max_query_length+2:
                query_ids.append(0)
                query_input_mask.append(0)

            assert len(query_ids) == max_query_length + 2
            assert len(query_input_mask) == max_query_length + 2

            start_position = None
            end_position = None
            if not example.is_impossible and is_training:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                    #print("out of span")
                if out_of_span:
                    #print("---train-- out of span:--", unique_id, tok_start_position, tok_end_position, doc_start, doc_end)
                    start_position = 0
                    end_position = 0
                    num_out_of_span += 1
                else:
                    if not args.reverse_seq:
                      doc_offset = len(query_tokens) + 2
                    else:
                      doc_offset =  1 #Just [CLS] token
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            if is_training and example.is_impossible:
                start_position = 0
                end_position = 0


            if example_index < 20:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and example.is_impossible:
                    logger.info("impossible example")
                if is_training and not example.is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info(
                        "answer: %s" % (answer_text))

            if not out_of_span: # or not is_generate is removed 

             # Get the qas_id as exmaple index - one example per question
             #example_id=int(example.qas_id)
             example_id=example.qas_id

             #print(example_index, example.qas_id)
             #      example_index=example_index,

             in_span_unique_ids.append(example_index)

             #print("length of the features: %s"%(len(features)))
             
             features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_id, 
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible,
                    pretrain_id = example.pretrain_id,
                    query_ids = query_ids,
                    query_input_mask = query_input_mask))
             unique_id += 1
            else:
                out_span_unique_ids.append(example_index)


    not_present=0
    for uid in out_span_unique_ids:
        if uid not in in_span_unique_ids:
            #print("unique id:%s, ignored"% (uid))
            not_present+=1

    print("Number of out of spans:%s, features size:%s, examples size:%s, not_present:%s" %(num_out_of_span, 
        len(features), 
        len(examples), not_present) )
    if not_present != 0:
        print("Not all examples are included")
    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])


def write_predictions(all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature.example_index].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature.unique_id]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature.tokens):
                        continue
                    if end_index >= len(feature.tokens):
                        continue
                    if start_index not in feature.token_to_orig_map:
                        continue
                    if end_index not in feature.token_to_orig_map:
                        continue
                    if not feature.token_is_max_context.get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                tok_tokens = feature.tokens[pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature.token_to_orig_map[pred.start_index]
                orig_doc_end = feature.token_to_orig_map[pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                tok_text = tok_text.replace(" ##", "")
                tok_text = tok_text.replace("##", "")

                # Clean whitespace
                tok_text = tok_text.strip()
                tok_text = " ".join(tok_text.split())
                orig_text = " ".join(orig_tokens)

                final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))
                
            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))
                
        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            # predict "" iff the null score - the score of best non-null > threshold
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff
            if score_diff > null_score_diff_threshold:
                all_predictions[example.qas_id] = ""
            else:
                all_predictions[example.qas_id] = best_non_null_entry.text
            all_nbest_json[example.qas_id] = nbest_json

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4) + "\n")


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs

def calculate_ngram_match(candidates,references,n=3):

    if len(candidates) != len(references):
        print("mismatch in the count of sentences to calculate ngram match")
        return 0,100
    if n == 3:
     weights = (0,0,1,0) 
    if n == 4:
     weights = (0,0,0,1) 
    if n == 2:
     weights = (0,1,0,0) 

    denom=0
    for i in range(len(candidates)):
     text1=candidates[i]
     #text2=references[i]
     ngrams1=[ [ candidates[i][k+j] for j in range(n)] for k in range(len(text1) - n + 1) ]
     #ngrams2=[ [ references[i][k+j] for j in range(n)] for k in range(len(text2) - n + 1) ]
     denom += len(ngrams1)

    if denom == 0:
        denom = 1

    return denom * corpus_bleu([ [x] for x in references], candidates,weights=weights) , denom



def generate_questions(model, tokenizer, input_ids, segment_ids,input_mask, start_positions, end_positions, references,args):
     with torch.no_grad():
          batch_qgen_logits = model(input_ids, segment_ids, input_mask, start_positions,end_positions,evaluate=True,copymethod=args.copymethod)

     #Best word at each step
     eoses_found = False
     step=1
     while not eoses_found:

                #Get the best words
                #best_probs, best_words = batch_qgen_logits[:,-1:,:].max(-1)
                best_probs, sample_words = batch_qgen_logits[:,-1,:].max(-1)

                #probs = F.softmax(batch_qgen_logits, dim=-1)
                #probs = probs[:,-1,:]
                #sample_words = torch.multinomial(probs, num_samples=1)

                #print(step, [" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] != 0])) for b in range(input_ids.size(0)) ] )
                #input("debug")

                #create new input_ids, segment_ids and input_mask
                #input_ids = torch.cat((input_ids,best_words), -1)

                # Get lengths
                input_lengths = (input_ids != 0 ).sum(-1)  # subtract 1 to get index
                #print(input_ids[0],input_lengths.size(), input_ids.size(),sample_words.size() )

                input_ids = input_ids.scatter(1,input_lengths.unsqueeze(-1),sample_words)
                segment_ids = segment_ids.scatter(1,input_lengths.unsqueeze(-1),torch.zeros_like(sample_words)+1)
                input_mask = input_mask.scatter(1,input_lengths.unsqueeze(-1),torch.zeros_like(sample_words)+1)

                # Run model: find new batch_qgen_logits
                batch_qgen_logits = model(input_ids, segment_ids, input_mask, start_positions,end_positions,evaluate=True)
                #print([ tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] != 0]) for b in range(input_ids.size(0)) ])

                #Check if the input_ids contain eos in all batch or if the length is max_length
                if (input_ids == 102).sum() >= input_ids.size(0) or torch.max(segment_ids.sum(1)) > 40 :
                    eoses_found = True


     #Print predictions on file
     preds=[ " ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] != 0])) for b in range(input_ids.size(0))  ]
     contexts=[ " ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0 and input_mask[b][s] == 1])) for b in range(input_ids.size(0))  ]

     #Find Bleu score later
     ngram_match, denom = calculate_ngram_match(references, preds,n=2)

     wf=open("debug_gens.txt","a") 
     for i in range(min(len(preds),10)):
                print("###-",i)
                print(preds[i])
                print(references[i])
                wf.write("###\n")
                wf.write("###-context"+str(i)+"\n")
                wf.write(contexts[i]+"\n")
                wf.write("###-answer phrase"+str(i)+"\n")
                wf.write(str(start_positions[i])+"-"+str(end_positions[i])+"\n")
                wf.write("###-reference"+str(i)+"\n")
                wf.write(references[i]+"\n")
                wf.write("###-prediction:"+str(i)+"\n")
                wf.write(preds[i]+"\n")
                wf.write("###\n")

     return ngram_match, denom

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
        Args:
            logits: logits distribution shape (vocabulary size)
            top_k > 0: keep only top k tokens with highest probability (top-k filtering).
            top_p > 0.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
        From: https://gist.github.com/thomwolf/1a5a29f6962089e871b94cbd09daf317
    """
    assert logits.dim() == 1  # batch size 1 for now - could be updated for more but the code would be less clear
    top_k = min(top_k, logits.size(-1))  # Safety check
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def rerank(preds, segments, scores, input_ids, input_mask,qamodel,start_positions,end_positions,top=10):
    #preds,segments, scores are list
    maxlen=0
    preds = preds[0][:top]
    segments = segments[0][:top]
    scores = scores[0][:top]
    for i in range(len(preds)):
       if maxlen <=preds[i].size(0):
          maxlen =preds[i].size(0)
         
    new_preds = torch.zeros(len(preds),maxlen+2).long().cuda()
    new_segments = torch.zeros(len(preds),maxlen+2).long().cuda() #maxlen+1[CLS] +1[SEP]
    new_start_indices = torch.zeros(len(preds)).long().cuda() 
    new_end_indices = torch.zeros(len(preds)).long().cuda() 

    for i in range(len(preds)):
        inp_len=(1-segments[i]).sum()
        q_len=segments[i].sum()
        new_preds[i][0]=101 # [CLS] token
        new_preds[i][1:q_len] = preds[i][inp_len+1:]

        #new_preds[i][q_len] = 102 #[SEP] 
        #new_preds[i][q_len+1:q_len+1+inp_len] = preds[i][:inp_len]
        new_preds[i][q_len:q_len+inp_len] = preds[i][:inp_len]

        #new_preds[i][q_len+1+inp_len] = 102 #[SEP] 
        new_preds[i][q_len+inp_len] = 102 #[SEP] 

        #new_segments[i][q_len+1:q_len+1+inp_len] = 1-segments[i][:inp_len]
        #new_segments[i][q_len+1+inp_len] = 1 #[SEP]
        new_segments[i][q_len:q_len+inp_len] = 1-segments[i][:inp_len]
        new_segments[i][q_len+inp_len] = 1 #[SEP]
        new_start_indices[i] = start_positions+q_len-1
        new_end_indices[i] = end_positions+q_len-1

    start_logits,end_logits = qamodel(new_preds, new_segments)

    #log probs
    start_logits = torch.log(torch.softmax(start_logits,dim=-1)) # Convert logits to log_probs; sorry for the naming convention
    end_logits = torch.log(torch.softmax(end_logits,dim=-1)) # Convert logits to log_probs; sorry for the naming convention
 
    new_scores = start_logits.gather(1,new_start_indices.unsqueeze(-1)) + end_logits.gather(1,new_end_indices.unsqueeze(-1))
    
    val, ind = torch.sort(new_scores,dim=0,descending=True)
    ind = ind.squeeze(-1)
    new_preds = new_preds.index_select(0,ind) 
    new_scores = new_scores.index_select(0,ind) 
    new_segments = new_segments.index_select(0,ind) 
    new_start_indices = new_start_indices.index_select(0,ind) 
    new_end_indices = new_end_indices.index_select(0,ind) 
    

    #return [preds], [segments], [scores]
    return [new_preds], [1-new_segments], [new_scores],new_start_indices, new_end_indices

def generate_from_embedding(

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
        input_ids, segment_ids, input_mask, pretrain_ids, query_ids, query_input_mask, start_positions, end_positions, references = batch

        #print(input_ids)
        #input("input ids gen")

        #qgen_logits = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod,evaluate=True,args=args)


        if args.do_train:
          core_model = model.module
        elif args.do_predict:
          core_model = model

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


        def sample_next_word(log_probs):
              

            categ = torch.distributions.Categorical(log_probs)

            sampled = categ.sample()

            #print(sampled.size())
            #input("sampled size")

            return sampled




        # (0) pt 2, prep the beam object
        #print(query_input_mask.size(), query_ids.size(), )

        #q_outputs = model.module.bert(input_ids = query_ids, token_type_ids = torch.zeros_like(query_ids), attention_mask = query_input_mask)

        q_outputs = core_model.bert(input_ids = query_ids, token_type_ids = torch.zeros_like(query_ids), attention_mask = query_input_mask)
        q_emb = q_outputs[0][:,0:1]



        #print_input_ids(tokenizer, query_ids, index="Q-emb  :")
        #input("")


        q_encode_outputs=model(input_ids=input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                pretrain_ids=pretrain_ids, 
                query_ids=query_ids,
                query_input_mask=query_input_mask,
                start_positions=start_positions, 
                end_positions=end_positions,
                copymethod=args.copymethod,evaluate=True, args=args)
        q_emb = q_encode_outputs[1] 


        query_new_ids = query_ids [:, 0:1]
        token_type_ids = torch.zeros_like(query_new_ids)
        attention_mask = torch.ones_like(query_new_ids)

 
        #log_probs  = core_model.log_likelihood(query_new_ids, outputs, query_new_ids,  args, True, pretrain_ids)


        q_decode_outputs=model(input_ids=input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                pretrain_ids=pretrain_ids, 
                query_ids=query_ids,
                query_input_mask=query_input_mask,
                start_positions=start_positions, 
                end_positions=end_positions,
                copymethod=args.copymethod,evaluate=True, args=args, given_q_emb=q_emb)

        log_probs = q_decode_outputs[0] 


        # Sample and get the next word

        #print(query_new_ids)
        #print(log_probs.size())

        # Get new input ids etc to save in the beam ( Beam )

        #JUST renaming
        input_ids = query_new_ids
        segment_ids = token_type_ids
        input_mask = attention_mask

        #else:
        '''
        model.bert(input_ids=input_ids,
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                pretrain_ids=pretrain_ids, 
                query_ids=query_ids,
                query_input_mask=query_input_mask,
                start_positions=start_positions, 
                end_positions=end_positions,
        '''

        #log_probs = outputs[0]

        log_probs = log_probs[:,-1,:].unsqueeze(1) # batch_size x 1 x vocab_size
        sampled = sample_next_word(log_probs)

        #print_input_ids(tokenizer, [sampled], index="1sample:")
        #input("")

        # Adjust input_ids, segment_ids, input_mask etc

        input_ids = torch.cat((input_ids, sampled), dim=-1)
        segment_ids = torch.cat((segment_ids, torch.ones_like(sampled)), dim=-1)
        input_mask = torch.ones_like(segment_ids)

        with torch.no_grad():
          for step in range(max_length):

            attn=None

            if args.debug:

             for j in range(beam.alive_seq[:,1:].size(0)):
              print(tokenizer.convert_ids_to_tokens([beam.alive_seq[j][i].item() for i in range(beam.alive_seq.size(1)-1) if beam.alive_segment[j][i] > -2 ])) 


            '''
            word_embs = core_model.bert.embeddings.word_embeddings(input_ids) #, token_type_ids=token_type_ids) #+answer_phrase_ids)
            word_embs = word_embs  +  q_emb # b x seq x 1 * bx1xhidden_size


            # Add diag mask here?:
            diag_mask = torch.ones_like(input_ids)
            diag_mask = diag_mask.unsqueeze(-1).expand(-1,-1,input_ids.size(-1))
            diag_mask = torch.tril(diag_mask) # b x q_seq  x q_seq
            '''

            #outputs = model.module.bert(inputs_embeds=word_embs,
            #print("word emb size, segment ids size, diag mask", word_embs.size(), segment_ids.size(), diag_mask.size()) 

            #outputs = model(inputs_embeds=word_embs,
            #                        token_type_ids=segment_ids,
            #                        attention_mask=diag_mask) 

            q_decode_outputs=model(input_ids=input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                pretrain_ids=pretrain_ids, 
                query_ids=input_ids, #query_ids,
                query_input_mask=input_mask, #query_input_mask,
                start_positions=start_positions, 
                end_positions=end_positions,
                copymethod=args.copymethod,evaluate=True, args=args, given_q_emb=q_emb)

            log_probs = q_decode_outputs[0] 
            #print("log probs size:",log_probs.size())




            #log_probs  = core_model.log_likelihood(input_ids, outputs, input_ids,  args, True, pretrain_ids)


            log_probs = log_probs[:,-1,:]# (batch_size*beam_size) x vocab_size
            sampled = sample_next_word(log_probs)

            #print_input_ids(tokenizer, [sampled], index="sampled:")

            if step == 0:
             scores = log_probs[:,sampled]
            else:
             scores += log_probs[:,sampled]

            #print_input_ids(tokenizer, input_ids,index=step)
            # Adjust input_ids, segment_ids, input_mask etc

            input_ids = torch.cat((input_ids, sampled.unsqueeze(-1)), dim=-1)
            segment_ids = torch.cat((segment_ids, torch.ones_like(sampled.unsqueeze(-1))), dim=-1)

            #segment_ids = torch.zeros_like(input_ids)

            input_mask = torch.ones_like(input_ids)

            if sampled.item() == sep_token:
                pass

        #print((beam.segments).sum())
        #results["scores"] = beam.scores
        #print([sum(beam.segments[0][i]) for i in range(len(beam.scores[0])) ])
        #input("")
        #print(beam.predictions)
        #input("beam predictions")

        results["scores"] = [[ scores[i] for i in range(len(input_ids)) ]]
        results["predictions"] = [ input_ids.view(1,-1) ] # nullifying 1st dim: ok for batch_size 1
        results["segments"] = [ torch.ones_like(segment_ids.view(1,-1)) ] # nullifying 1st dim: ok for batch_size 1

        #results["segments"] = [ torch.ones_like(segment_ids.view(1,-1))] # For printing
        #results["attention"] = beam.attention

        return results


def generate_from_embedding_doesntwork(
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
        input_ids, segment_ids, input_mask, pretrain_ids, query_ids, query_input_mask, start_positions, end_positions, references = batch

        #print(input_ids)
        #input("input ids gen")

        #qgen_logits = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod,evaluate=True,args=args)


        if args.do_train:
          core_model = model.module
        elif args.do_predict:
          core_model = model

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


        def sample_next_word(log_probs):
              

            categ = torch.distributions.Categorical(log_probs)

            sampled = categ.sample()

            #print(sampled.size())
            #input("sampled size")

            return sampled




        # (0) pt 2, prep the beam object
        #print(query_input_mask.size(), query_ids.size(), )

        #q_outputs = model.module.bert(input_ids = query_ids, token_type_ids = torch.zeros_like(query_ids), attention_mask = query_input_mask)
        q_outputs = core_model.bert(input_ids = query_ids, token_type_ids = torch.zeros_like(query_ids), attention_mask = query_input_mask)
        q_emb = q_outputs[0][:,0:1]

        #print_input_ids(tokenizer, query_ids, index="Q-emb  :")
        #input("")



        '''

                model(input_ids=input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                pretrain_ids=pretrain_ids, 
                query_ids=query_ids,
                query_input_mask=query_input_mask,
                start_positions=start_positions, 
                end_positions=end_positions,
                copymethod=args.copymethod,evaluate=True, args=args)
        '''


        query_new_ids = query_ids [:, 0:1]
        token_type_ids = torch.zeros_like(query_new_ids)
        attention_mask = torch.ones_like(query_new_ids)

        word_embs = core_model.bert.embeddings.word_embeddings(query_new_ids) #, token_type_ids=token_type_ids) #+answer_phrase_ids)

        word_embs = word_embs  +  q_emb # b x seq x 1 * bx1xhidden_size
 
        #outputs = model.module.bert(inputs_embeds=word_embs,
        outputs = core_model.bert(inputs_embeds=word_embs,
                                    token_type_ids=token_type_ids,
                                    attention_mask=attention_mask)

        # Add diag mask here:
        #log_probs  = model.module.log_likelihood(query_new_ids, outputs, query_new_ids,  args, True, pretrain_ids)

        log_probs  = core_model.log_likelihood(query_new_ids, outputs, query_new_ids,  args, True, pretrain_ids)

        # Sample and get the next word

        #print(query_new_ids)
        #print(log_probs.size())

        # Get new input ids etc to save in the beam ( Beam )

        input_ids = query_new_ids
        segment_ids = token_type_ids
        input_mask = attention_mask

        #else:
        '''
        model.bert(input_ids=input_ids,
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                pretrain_ids=pretrain_ids, 
                query_ids=query_ids,
                query_input_mask=query_input_mask,
                start_positions=start_positions, 
                end_positions=end_positions,
        '''

        #log_probs = outputs[0]

        log_probs = log_probs[:,-1,:].unsqueeze(1) # batch_size x 1 x vocab_size
        sampled = sample_next_word(log_probs)
        #print_input_ids(tokenizer, [sampled], index="1sample:")
        #input("")

        # Adjust input_ids, segment_ids, input_mask etc
        input_ids = torch.cat((input_ids, sampled), dim=-1)
        segment_ids = torch.cat((segment_ids, torch.ones_like(sampled)), dim=-1)
        input_mask = torch.ones_like(segment_ids)

        with torch.no_grad():
          for step in range(max_length):
            attn=None

            if args.debug:
             for j in range(beam.alive_seq[:,1:].size(0)):
              print(tokenizer.convert_ids_to_tokens([beam.alive_seq[j][i].item() for i in range(beam.alive_seq.size(1)-1) if beam.alive_segment[j][i] > -2 ])) 

            word_embs = core_model.bert.embeddings.word_embeddings(input_ids) #, token_type_ids=token_type_ids) #+answer_phrase_ids)
            word_embs = word_embs  +  q_emb # b x seq x 1 * bx1xhidden_size


            # Add diag mask here?:
            diag_mask = torch.ones_like(input_ids)
            diag_mask = diag_mask.unsqueeze(-1).expand(-1,-1,input_ids.size(-1))
            diag_mask = torch.tril(diag_mask) # b x q_seq  x q_seq

            #outputs = model.module.bert(inputs_embeds=word_embs,

            #print("word emb size, segment ids size, diag mask", word_embs.size(), segment_ids.size(), diag_mask.size()) 

            outputs = core_model.bert(inputs_embeds=word_embs,
                                    token_type_ids=segment_ids,
                                    attention_mask=diag_mask) #input_mask)


            log_probs  = core_model.log_likelihood(input_ids, outputs, input_ids,  args, True, pretrain_ids)


            log_probs = log_probs[:,-1,:]# (batch_size*beam_size) x vocab_size
            sampled = sample_next_word(log_probs)

            #print_input_ids(tokenizer, [sampled], index="sampled:")
            #input("")

            if step == 0:
             scores = log_probs[:,sampled]
            else:
             scores += log_probs[:,sampled]

            #print_input_ids(tokenizer, input_ids,index=step)

            # Adjust input_ids, segment_ids, input_mask etc
            input_ids = torch.cat((input_ids, sampled.unsqueeze(-1)), dim=-1)
            segment_ids = torch.cat((segment_ids, torch.ones_like(sampled.unsqueeze(-1))), dim=-1)
            #segment_ids = torch.zeros_like(input_ids)
            input_mask = torch.ones_like(input_ids)

            if sampled.item() == sep_token:
                pass

        
        #print((beam.segments).sum())
        #results["scores"] = beam.scores
        #print([sum(beam.segments[0][i]) for i in range(len(beam.scores[0])) ])
        #input("")
        #print(beam.predictions)
        #input("beam predictions")

        results["scores"] = [[ scores[i] for i in range(len(input_ids)) ]]
        results["predictions"] = [ input_ids.view(1,-1) ] # nullifying 1st dim: ok for batch_size 1
        results["segments"] = [ torch.ones_like(segment_ids.view(1,-1)) ] # nullifying 1st dim: ok for batch_size 1

        #results["segments"] = [ torch.ones_like(segment_ids.view(1,-1))] # For printing
        #results["attention"] = beam.attention

        return results

def translate_batch(
            model,
            tokenizer,
            batch,
            max_length,
            args,
            qamodel,
            qlm_model=None,
            encoder_model=None,
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
        input_ids, segment_ids, input_mask, pretrain_ids, query_ids, query_input_mask, start_positions, end_positions, references = batch

        #print(input_ids)
        #input("input ids gen")

        #qgen_logits = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod,evaluate=True,args=args)

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






        #encoder_model:
        if encoder_model is not None:

          if args.encoder_loss is not None:
            encoding_loss = args.encoder_loss # This loss can vary from args.loss, add such provision later
          else:
            encoding_loss = args.loss # This loss can vary from args.loss, add such provision later


           

          encoder_args = copy.deepcopy(args)
          encoder_args.add_target_only = True

          encoded_vector = encoder_model(input_ids=input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                pretrain_ids=pretrain_ids, 
                query_ids=query_ids,
                query_input_mask=query_input_mask,
                start_positions=start_positions, 
                end_positions=end_positions, 
                copymethod=encoder_args.copymethod,
                encoding_only=True, 
                args=encoder_args,
                loss=encoding_loss)

          if type(encoded_vector) == tuple:
              encoded_vector = encoded_vector[1] # take the second argument as encoded vector


          #input("encoding finished")

        else:
          encoded_vector = None


        outputs = model(input_ids=input_ids, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                pretrain_ids=pretrain_ids, 
                query_ids=query_ids,
                query_input_mask=query_input_mask,
                start_positions=start_positions, 
                end_positions=end_positions,
                copymethod=args.copymethod,
                evaluate=True, 
                args=args,
                encoded_vector=encoded_vector)

        log_probs = outputs[0]
        
        if qlm_model is not None:
            lm_logits = qlm_model(input_ids=input_ids, 
                    segment_ids=segment_ids, 
                    attention_mask=input_mask, 
                    start_positions=start_positions, 
                    end_positions=end_positions,
                    copymethod=args.copymethod,
                    evaluate=True)
            lm_log_probs = torch.log(torch.softmax(lm_logits,dim=-1))
            print("reducing log_probs according to second model")
            log_probs = log_probs - 0.2 * lm_log_probs

        log_probs = log_probs[:,-1,:].unsqueeze(1) # batch_size x 1 x vocab_size
        dummy_beam_probs = torch.zeros_like(log_probs).expand(-1,beam_size-1,-1) 
                                                       # batch_size x (beam_size-1)x vocab_Size
        log_probs = torch.cat((log_probs, dummy_beam_probs),1).view(beam_size*batch_size,-1) # (batch_size*beam_size x vocab_size)

        #print(input_ids)
        #input("input ids")

        beam_input_ids  = input_ids.unsqueeze(1).expand(-1,beam_size,-1).view(beam_size*batch_size,-1)

        #beam_query_ids  = query_ids.unsqueeze(1).expand(-1,beam_size,-1).view(beam_size*batch_size,-1)
        #beam_query_input_mask  = query_input_mask.unsqueeze(1).expand(-1,beam_size,-1).view(beam_size*batch_size,-1)

        beam_segment_ids  = segment_ids.unsqueeze(1).expand(-1,beam_size,-1).view(beam_size*batch_size,-1)
        beam_input_mask  = input_mask.unsqueeze(1).expand(-1,beam_size,-1).view(beam_size*batch_size,-1)
        #beam.assign_whats_alive(input_ids, segment_ids, input_mask, start_positions, end_positions)
        beam.assign_whats_alive(input_ids, segment_ids, input_mask, start_positions, end_positions)

        with torch.no_grad():
          for step in range(max_length):
            #decoder_input = beam.current_predictions
            attn=None

            #print(log_probs.size())
            #input("log probs size")
            beam.advance(log_probs, attn)

            any_beam_is_finished = beam.is_finished.any()
            #=1 if there is some eos in the beam, i.e [SEP]

            #To print beam
            if args.debug:
             for j in range(beam.alive_seq[:,1:].size(0)):
              print(tokenizer.convert_ids_to_tokens([beam.alive_seq[j][i].item() for i in range(beam.alive_seq.size(1)-1) if beam.alive_segment[j][i] > -2 ])) 
             #input("beam progress")
            #print([ tokenizer.convert_ids_to_tokens([beam.alive_seq[j][i].item() for i in range(beam.alive_seq.size(1)-1) if beam.alive_segment[j][i] == 1 ]) for j in range(beam.alive_seq[:,1:].size(0))])
            #input("segments 1")

            if any_beam_is_finished:
                beam.update_finished()
                #print("update finished")
                if beam.done:
                    break

            input_ids, segment_ids, input_mask, start_positions, end_positions = beam.get_whats_alive()

            #model.print(args, input_ids, prefix="input ids",  token_type_ids = segment_ids)
            #print("segment_ids", segment_ids[0])
            #input("---")

            # split here for smaller batch sizes
            if split:
              batch_size = input_ids.size(0) 
              #log_probs=torch.tensor().cuda()
              log_probs1 = model(input_ids[:5], segment_ids[:5], input_mask[:5], start_positions[:5], end_positions[:5],copymethod=args.copymethod,evaluate=True,args=args)
              log_probs2 = model(input_ids[5:], segment_ids[5:], input_mask[5:], start_positions[5:], end_positions[5:],copymethod=args.copymethod,evaluate=True)
              log_probs = torch.cat((log_probs1,log_probs2),0)
            else:

              #Added pretrain_ids
              outputs = model(input_ids=input_ids, 
                      token_type_ids=segment_ids, 
                      attention_mask=input_mask, 
                      pretrain_ids=pretrain_ids, 
                      query_ids=query_ids,
                      query_input_mask=query_input_mask,
                      start_positions=start_positions, 
                      end_positions=end_positions,
                      copymethod=args.copymethod,
                      evaluate=True,args=args,
                      encoded_vector=encoded_vector)

              #print(outputs[0].size())
              #input("outputs[0].size")

              if type(outputs) == tuple:
                log_probs = outputs[0]
              else:
                log_probs = outputs

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

            #outputs = model(generated, segment_ids, input_mask, start_positions, end_positions,evaluate=True,copymethod=args.copymethod)  # Note: we could also use 'past' with GPT-2/Transfo-XL/XLNet (cached hidden-states)
            outputs = model(input_ids=generated, 
                token_type_ids=segment_ids, 
                attention_mask=input_mask, 
                pretrain_ids=pretrain_ids, 
                query_ids=generated,
                query_input_mask=input_mask,
                start_positions=start_positions, 
                end_positions=end_positions,
                copymethod=args.copymethod,evaluate=True, args=args)

        log_probs = outputs[0]
        if qlm_model is not None:

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
     #contexts +=  [" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0])) for b in range(input_ids.size(0)) ]
     #answer_phrases += [ " ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0 \
     #        and  (s >= start_positions.item() and s <= end_positions.item()) ])) for b in range(input_ids.size(0))  ]

     #segments.append(segment_ids)


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


def evaluate_genqa(model, tokenizer, args, device,no_of_datapoints=-1,train=True,test_examples=None):



        eval_dev_examples = read_hotpotpara_examples(input_file=args.predict_file_w_sent_bounds,
                                                is_training=train,
                                                version_2_with_negative=args.version_2_with_negative)

        # Get cache name
        if args.data_cache_name:
             cache_name = args.data_cache_name
        else:
             cache_name = args.bert_model
        cache_name = cache_name+".genqa"
        cached_eval_features_file = args.predict_file_w_sent_bounds+'_{0}_{1}_{2}_{3}'.format(
            list(filter(None, cache_name.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))

        try:
            with open(cached_eval_features_file, "rb") as reader:
                eval_features = pickle.load(reader)
        except:

            eval_features = convert_hotpotexamples_to_features(examples=eval_dev_examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=train,
                                                is_evaluate=True,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                pad_token_segment_id=3 if args.model_type in ['xlnet'] else 0,
                                                cls_token_at_end=True if args.model_type in ['xlnet'] else False,
                                                sequence_a_is_doc=True if args.model_type in ['bert'] else False)
            with open(cached_eval_features_file, "wb") as writer:
                    pickle.dump(eval_features, writer)


        if no_of_datapoints != -1:
            eval_dev_examples = eval_dev_examples[:no_of_datapoints]
            eval_features = eval_features[:no_of_datapoints]

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_dev_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        logger.info("Start evaluating")
        eval_loss = 0
        eval_hits = 0
        eval_dist = 0
        total_tokens = 0

        no_of_examples=0
        examples=[]
        predictions=dict()
        count_wrong_sentmarkers = 0
        with torch.no_grad():
          for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):
            if no_of_examples % 1000 == 0:
                logger.info("Processing example: %d" % (no_of_examples))

            batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
            input_ids, input_mask, segment_ids, sent_lens, sent_supps, start_positions, end_positions, phr_start_positions, phr_end_positions, q_match_len  = batch

            '''
            print("input ids", input_ids)
            print("input mask", input_mask)
            print("segment ids", segment_ids)
            print("start positions", start_positions)
            print("end positions", end_positions)
            input("input_ids")
            '''

            '''
            #Batch size = 1, remove the pads
            input_length = (input_ids!=0).sum(-1).item() 
            input_ids = input_ids[:,:input_length]
            input_mask = input_mask[:,:input_length]
            segment_ids = segment_ids[:,:input_length]
            #start_positions = start_positions[:,:input_length]
            #end_positions = end_positions[:,:input_length]
            '''

            #references = [ re.sub("[     ]*\[SEP\]","",re.sub(".*\[MASK\]",""," ".join(all_tokens[i]))) for i in example_indices ]

            ''' Here we have to use different 'answer phrases' that yields the minimum loss by the genqa model '''

            # Get Best Word 
            word_scores=[]
            for b in range(0):
             for s in range(input_ids.size(1)):
              if input_ids[b][s] == 0: # Ignore padd token positions and further
                 break

              new_segment_ids = torch.zeros_like(segment_ids)
              new_segment_ids[b][s] == 1
              new_segment_ids = segment_ids + new_segment_ids

              if (new_segment_ids  > 1).sum().item() != 0:
                  print("->",s)
                  print("0",segment_ids)
                  print("+",sent_lens[:,s,:])
                  print("1",new_segment_ids)
                  count_wrong_sentmarkers += 1
                  word_scores.append(0)
                  continue

              loss = model(input_ids, new_segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod, args=args)
              word_scores.append(-loss.item())

            ground_loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod, args=args)

            word_val, word_ind = torch.Tensor(word_scores).max(-1)
            print("word scores: %s, %s" %(word_scores[word_ind], ground_loss.item()))


            #input("check the predicted word for answer")

            # Get predicted sentence
            sent_ind = word_ind.unsqueeze(-1).unsqueeze(-1) * sent_lens # b x 1 x 1  * b x #sent x seq
            


            #predicted_start = phr_start_positions.gather(-1,phrase_ind.unsqueeze(-1))
            #predicted_end = phr_end_positions.gather(-1,phrase_ind.unsqueeze(-1))

            for j in range(input_ids.size(0)):
             #print("-->"," ".join(tokenizer.convert_ids_to_tokens([input_ids[j][i].item() for i in range(len(input_ids[j])) if soft_token_ids[j][i] > 0.5 ])))

             question = " ".join(tokenizer.convert_ids_to_tokens([input_ids[j][s].item() for s in range(input_ids.size(1)) if segment_ids[j][s] != 0][:]))
             para = " ".join(tokenizer.convert_ids_to_tokens([input_ids[j][s].item() for s in range(input_ids.size(1)) if segment_ids[j][s] == 0][:]))
             predicted=" ".join(tokenizer.convert_ids_to_tokens([input_ids[j][i].item() for i in range(input_ids.size(1)) if i >=predicted_start[j].item() and i<= predicted_end[j].item() ]))
             ground_truth=" ".join(tokenizer.convert_ids_to_tokens([input_ids[j][i].item() for i in range(len(input_ids[j])) if start_positions[j].item() <=i and end_positions[j].item() >=i ]))

             print(" ".join(tokenizer.convert_ids_to_tokens([input_ids[j][s].item() for s in range(input_ids.size(1)) if input_mask[j][s] == 1])))

             print("Question:")
             print(question)

             print("Answer Prases:")
             print("#########")
             print("predict:", predicted)
             print("ground :",ground_truth)
             print("#########")
             print("positions:")
             print("#########")
             print("predict:", predicted_start[j].item(), predicted_end[j].item())
             print("ground :", start_positions[j].item(), end_positions[j].item())
             print("#########")

             predictions[step*input_ids.size(0)+j] = predicted

             #examples[step*input_ids.size(0)+j] = {"answers":[{"text":ground_truth}]}

             example = SquadEvalExample(
                    qas_id=step*input_ids.size(0)+j,
                    question_text=question,
                    context_text=para,
                    answer_text=ground_truth,
                    doc_tokens=para,
                    start_position_character=start_positions[j].item(),
                    answers=[{"text":ground_truth}],
                    is_impossible=False
                    )

             examples.append(example)

             #input("check")

            val, pred_ind = sent_val, sent_ind
            val, ground_ind = sent_supps.max(-1)

            #print(pred_ind.size(), sent_supps.size(),sent_supps, pred_ind)
            #input("ind size")

            if pred_ind.item() == ground_ind.item():
                eval_hits += 1


            '''
            else:
                sign = lambda x: math.copysign(1, x)
                left=start_positions.item()-start_ind.item()
                left=sign(left)*left
                right=end_positions.item()-start_ind.item()
                right=sign(right)*right
                eval_dist+=min(left, right)
            '''

            #print(eval_hits, eval_dist, min(left,right), no_of_examples)

            eval_loss += eval_hits
            #total_tokens += (segment_ids == 1).sum()
            no_of_examples += args.predict_batch_size

        print("Eval hits percentage:", eval_hits/no_of_examples)
        print("Wrong sent markers: %s "%  (count_wrong_sentmarkers))
        print("Eval loss:", eval_loss)
        print("Eval Total Points:", no_of_examples)

        results = squad_evaluate(examples, predictions)
        print("Eval-Squad Results:", results)
        return (100*eval_hits/no_of_examples), eval_loss, results

def evaluate_qa(model, tokenizer, args, device, step=0, no_of_datapoints=-1,train=False,test_examples=None, ppl_only=False, predict_dir="", encoder_model=None):
     
     def to_list(tensor):
       return tensor.detach().cpu().tolist()



     dev_files = [f for f in listdir(predict_dir) if re.match("^.*json$|^.*txt$", f) ] # isfile(join(mypath, f))]
     dev_files = sorted(dev_files)
     dev_losses = {} #losses per dev file
     dev_results = {} #losses per dev file

     print("predict dir:%s, files: {{  %s  }}\n"%(predict_dir, dev_files))
     
     for devf in dev_files:
        data_file = predict_dir +"/" + devf

        ''' Get the suffixes for lang and the type of conditional language modeling : question/gap '''
        try:
          lang = re.sub(".json|.txt","",devf).split("_")[-2]
        except:
          lang = "en"
        if lang != "":
            args.lang = lang

        clm = re.sub(".json|.txt","",devf).split("_")[-1]

        if clm == "question":
            EOP="[QUESTION]"
        elif clm == "lm":
            EOP="[QUESTION]"
        elif clm == "gap":
            #EOP="[GAP]"
            EOP="[QUESTION]" #QUESITON always
        else:
            #print("using SEP as EOP token")
            EOP="[SEP]"  #default question
            #EOP="[QUESTION]"  #default question


        if clm == "lm":

          eval_dev_examples = read_lm_examples(
            input_file=data_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
        else:

          eval_dev_examples, version = read_squad_examples(
            input_file=data_file, 
            is_training=False, 
            version_2_with_negative=args.version_2_with_negative, 
            args=args,clm=clm)

        print("%s datapoints in %s"%(len(eval_dev_examples), data_file))

        if no_of_datapoints != -1:
            eval_dev_examples = eval_dev_examples[:no_of_datapoints]

        print("data cache name:", args.data_cache_name)
        cache_name = "_qa_" + args.data_cache_name #bert_model
        if args.reverse_seq:
            cache_name += "rev_for_run_ae.py"
        # if tokenizer
        cached_eval_features_file = data_file+'_{0}_{1}_{2}_{3}'.format(
            list(filter(None, cache_name.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))

        try:
            with open(cached_eval_features_file, "rb") as reader:
                eval_features = pickle.load(reader)
        except:

          if clm == "lm":
           eval_features = convert_lmexamples_to_features(
            examples=eval_dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True,
            eop_token=EOP) 
            #,no_question=True)
          else:  

            print("version:",version)
            #input("version")

            if version == 0:
              print("version 0 for evaluation, converting into hotpot examples")
              eval_features = convert_hotpotexamples_to_features(examples=eval_dev_examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=True,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                pad_token_segment_id=3 if args.model_type in ['xlnet'] else 0,
                                                cls_token_at_end=True if args.model_type in ['xlnet'] else False,
                                                sequence_a_is_doc=True if args.model_type in ['bert'] else False)


            else:
             eval_features = convert_examples_to_features(
             examples=eval_dev_examples,
             tokenizer=tokenizer,
             max_seq_length=args.max_seq_length,
             doc_stride=args.doc_stride,
             max_query_length=args.max_query_length,
             is_training=True,
             eop_token=EOP,
             args=args) 
             #,no_question=True)
          with open(cached_eval_features_file, "wb") as writer:
                    pickle.dump(eval_features, writer)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_dev_examples))
        logger.info("  Num split examples(features) = %d", len(eval_features))
        print("  Num split examples(features) = %d"% (len(eval_features)) )
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_unique_ids = torch.tensor([f.unique_id for f in eval_features], dtype=torch.long)
        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)

        all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)
        all_pretrain_ids = torch.tensor([f.pretrain_id for f in eval_features], dtype=torch.long)

        all_query_ids = torch.tensor([f.query_ids for f in eval_features], dtype=torch.long)
        all_query_input_mask = torch.tensor([f.query_input_mask for f in eval_features], dtype=torch.long)


        #all_tokens = [f.tokens for f in eval_features]
        #all_ids = torch.tensor([ i for i in range(len(eval_features)) ], dtype=torch.long)
        #all_example_index = torch.tensor([f.example_index for f in eval_features], dtype=torch.long)

        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        eval_data = TensorDataset(all_input_ids,
                                  all_input_mask, 
                                  all_segment_ids, 
                                  all_start_positions, 
                                  all_end_positions, 
                                  all_example_index,
                                  all_pretrain_ids, 
                                  all_query_ids, 
                                  all_query_input_mask, 
                                  all_unique_ids)

        # Run prediction for full data

        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        logger.info("Start evaluating")
        eval_loss = 0
        total_tokens = 0
        all_results=[]
        output_prediction_file = args.output_dir +"/predicted_"+str(step)+"."+devf
        output_nbest_file = args.output_dir +"/nbest_"+str(step)+"."+devf

        prefix = str(step)+"."+devf
        output_prediction_file = os.path.join(args.output_dir, "predictions_{}.json".format(prefix))
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions_{}.json".format(prefix))
        output_null_log_odds_file = os.path.join(args.output_dir, "null_log_odds_{}.json".format(prefix))

        eval_examples=[]
        no_of_tokens = 0
        for example in eval_dev_examples:
        #? for (example_index, example) in enumerate(eval_dev_examples):

            #print("qas:%s\n q-text:%s \n answer:%s \n doc_tokens: %s" % (example.qas_id, example.question_text , example.orig_answer_text, example.doc_tokens))
            #input("-eval example-, qas_id, question_text, context_text, answer_text, start_position_character, answers,is_impossible ")

            new_example = SquadEvalExample(
                    qas_id=example.qas_id, # example_index? 
                    question_text=example.question_text, 
                    context_text=example.doc_tokens,
                    answer_text=example.orig_answer_text,
                    doc_tokens=example.doc_tokens,
                    start_position_character=example.start_position,
                    answers=[{"text":example.orig_answer_text}],
                    is_impossible=False)

            eval_examples.append(new_example)

        eval_features_by_ids = dict()
        feature_best_scores = dict()

        no_of_examples=0
        tokens = 0
        with torch.no_grad():

          for batch in tqdm(eval_dataloader, desc="Evaluating by qa "+devf, disable=args.local_rank not in [-1, 0]):


            batch = [t.to(device) for t in  batch ]
            input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices, pretrain_ids, query_ids, query_input_mask, unique_ids = batch
            if no_of_examples % 1000 == 0:
                logger.info("Processing example: %d" % (no_of_examples))

            '''
            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)
            '''

            tokens += segment_ids.sum()

            '''
            Question: does it make sense to reduced the length of the padded passage to 
               run it fast?
            '''

            #temp_args = copy.deepcopy(args) 
            #temp_args.loss == "qae"
            #print(start_positions, end_positions)

            if ppl_only: #args.loss in ["qa","prior", "genqa", "aae"] and not ppl_only:
             ''' To calculate PPL loss '''

             outputs = model(input_ids=input_ids, 
                    token_type_ids=segment_ids, 
                    attention_mask=input_mask, 
                    pretrain_ids=pretrain_ids, 
                    start_positions=start_positions, 
                    end_positions=end_positions,
                    copymethod=args.copymethod, args=args, loss = args.loss) # qa_by_qemb

             loss = outputs[0]
             eval_loss += loss.mean().item()
             no_of_examples += args.predict_batch_size

            #if args.loss in ["qa","prior", "genqa", "aae"] and not ppl_only:
            if True:


              ''' 
              def forward(self,
                 input_ids,
                 token_type_ids=None,
                 attention_mask=None,
                 pretrain_ids=None,
                 start_positions=None,
                 end_positions=None,
                 decoder_mask=True,
                 evaluate=False,
                 copymethod=1,
                 args=None):

              '''
              ''' To calculate EM using simple QA model '''

              encoded_embedding=None
              if args.encoder_model and encoder_model is not None:

                encoding_loss=args.encoder_loss if args.encoder_loss is not None else args.loss

                encoded_embedding = encoder_model(input_ids=input_ids, 
                      token_type_ids=segment_ids, 
                      attention_mask=input_mask, 
                      query_ids=query_ids, 
                      query_input_mask=query_input_mask, 
                      start_positions=start_positions, 
                      end_positions=end_positions, 
                      pretrain_ids=pretrain_ids, 
                      copymethod=args.copymethod, 
                      args=args, 
                      loss=encoding_loss,
                      evaluate=True, 
                      encoding_only=True)

              #encoded_embedding=None
              # Fix the loss if random loss is given
              if args.loss not in ["qa", "aae", "qa_by_qemb"]:
                  loss="qa_by_qemb"
                  #print("args.loss=%s, will do:%s"%(args.loss,loss))
              else:
                  loss=args.loss

              outputs = model(input_ids=input_ids, 
                      token_type_ids=segment_ids, 
                      attention_mask=input_mask, 
                      query_ids=query_ids, 
                      query_input_mask=query_input_mask, 
                      start_positions=start_positions, 
                      end_positions=end_positions, 
                      pretrain_ids=pretrain_ids, 
                      copymethod=args.copymethod, 
                      args=args, 
                      loss=loss, #args.loss,
                      #loss="qa_by_qemb",
                      evaluate=True,
                      encoded_vector=encoded_embedding)

              #outputs = model(input_ids, segment_ids, input_mask, pretrain_ids,  
              #                               start_positions, 
              #                              end_positions,
              #                               copymethod=args.copymethod,
              #                                 args=args, loss=args.loss, evaluate=True)

              tokens += segment_ids.sum().item() - segment_ids.size(0)

              # Remove batch size for not predicting [BOQ] token
              #print("Length of example features %s:"%(len(example_indices)))

              
              #start_logits, end_logits = outputs
              start_logits = outputs[0]
              end_logits = outputs[1]

              #print(len(outputs))
              #input("-")

              if len(outputs) == 3:
                '''
                soft_segments = outputs[2]
                print("soft segments:", soft_segments.view(-1))
                ret=model.print(args, input_ids)
                soft_segments_list = soft_segments.view(-1).tolist()
                length=min(len(ret), len(soft_segments_list))
                print([ ret[i]+"("+str(soft_segments_list[i])+")" for i in range(length) ] )
                input("-")
                '''
                '''
                '''
                pass

              start_val, start_ind = start_logits.max(-1)
              #print("val, start_ind %s, %s"%(val,start_ind))
              end_val, end_ind = end_logits.max(-1)

              start_ind = start_ind.unsqueeze(-1)
              end_ind = end_ind.unsqueeze(-1)

              print_start_positions = torch.cat((start_positions.unsqueeze(1), start_ind),dim=1)  
              print_end_positions = torch.cat((end_positions.unsqueeze(1), end_ind),dim=1)  




              if args.debug:
               if args.do_predict:
                model.print(args, input_ids, phr_start_positions=print_start_positions,phr_end_positions=print_end_positions,prefix="eval qa") 
                print(start_val.item(), end_val.item())
               else:
                model.module.print(args, input_ids, phr_start_positions=print_start_positions,phr_end_positions=print_end_positions,prefix="eval qa") 
                print(start_val.item(), end_val.item())

              #unique_ids.size(0) batch size is 1 normally
              for ui in range(unique_ids.size(0)):
               start_logits = outputs[0][ui]
               end_logits = outputs[1][ui]

               #print("start logits size:",start_logits.size())
               #print("end logits size:",end_logits.size())

               #print("val, end_ind %s, %s"%(val,end_ind))
               #input("-start_end, end_ind-")

               result = SquadResult(unique_ids[ui].item(), start_logits, end_logits)
               all_results.append(result) # result is appended per data-point, not per batch
              

              # From here to 
              '''
              for i, feature_index in enumerate(example_indices):

                # TODO: i and feature_index are the same number! Simplify by removing enumerate?

                

                print("qas id(feature_index) : %s, feature id: %s" %(feature_index.item(), i))
                eval_feature = eval_features[feature_index.item()]
                #unique_id = eval_feature.example_index
                unique_id = eval_feature.unique_id 

                output = [to_list(output[i]) for output in outputs]
                start_logits, end_logits = output

                start = sorted(list(enumerate(start_logits)), key=lambda x:x[1], reverse=True)
                end = sorted(list(enumerate(end_logits)), key=lambda x:x[1], reverse=True)
                best_score = start[0][1] + end[0][1]
                

                result = SquadResult(unique_id, start_logits, end_logits)
                #result = SquadResult(eval_feature, start_logits, end_logits)

                all_results.append(result) # result is appended per data-point, not per batch

                # If the results already contain the unique id and start/end vals,
                #  choose the result that has better start_logits.max(-1), end_logits.max(-1) values 
                
                if unique_id in feature_best_scores:
                    if best_score > feature_best_scores[unique_id]:
                      #print("+++",unique_id)
                      eval_features_by_ids[unique_id] = eval_feature
                      feature_best_scores[unique_id] = best_score
                else:
                  #print("---",unique_id)
                  eval_features_by_ids[unique_id] = eval_feature
                  feature_best_scores[unique_id] = best_score
              # Till here
              '''

        #if args.loss in ["qa","prior", "genqa", "aae"] and not ppl_only:
        if args.loss in ["qa_by_qemb", "qa","prior", "genqa", "aae"] and not ppl_only:
              '''
              eval_comparable_features = []
              for ex in eval_examples:
                eval_comparable_features.append(eval_features_by_ids[ex.qas_id])
              '''

              eval_comparable_features = eval_features
              print("Length of eval features : %s " %(len(eval_features)))
                  
              #print(len(eval_comparable_features), len(eval_examples), len(eval_features))
              #input("lengths of new features/ examples/ old features")

              predictions = compute_predictions_logits(
                              eval_examples,
                              eval_comparable_features,
                              all_results,
                              args.n_best_size,
                              args.max_answer_length,
                              args.do_lower_case,
                              output_prediction_file,
                              output_nbest_file,
                              output_null_log_odds_file,
                              args.verbose_logging,
                              args.version_2_with_negative,
                              args.null_score_diff_threshold,
                              tokenizer
                            )

              results = squad_evaluate(eval_examples, predictions)
              dev_results[devf] = results
              pretty_results = json.dumps(results,indent=4)



              print("results:%s" % (pretty_results))
              loss = results["f1"] 


              #log_file.write("results for %s : %s"%(devf,pretty_results)+"\n")
              #log_file.flush()
              #tokens += segment_ids.sum(-1)
 
        dev_losses[devf] = loss #eval_loss # / (1e-9 + tokens) #no_of_examples)
        pretty_dev_losses = json.dumps(dev_losses[devf],indent=4)
        #log_file.write("\nPPL: %s: %s" % (devf, pretty_dev_losses))
        #log_file.write("\n#tokens: %s\n" % (tokens))
        #log_file.flush()
        print("losses:%s" % (dev_losses[devf]))



        total_dev_loss = 0
        for key in dev_losses:
            total_dev_loss += dev_losses[key]

     print(dev_losses[devf], dev_results[devf])

     #make it pretty
     if ppl_only:
       if args.eval_on == "ppl":
        total_dev_loss = 0
        for key in dev_losses:
            total_dev_loss += dev_losses[key]
        return total_dev_loss, dev_losses
       elif args.loss in ["qa", "prior"] : # not ppl: so, either F1 or EM
        total_dev_loss = 0
        print(dev_results[key][0])
        #input("-check-")
        for key in dev_results:
            total_dev_loss += dev_results[key][0][1] # ordered dictionary, second
            
        return total_dev_loss, dev_losses
     elif args.loss in ["qa","prior"]:
       dev_results = json.dumps(dev_results,indent=4)
       return total_dev_loss, dev_results
     else: # Default case
       dev_losses = json.dumps(dev_losses,indent=4)
       return total_dev_loss, dev_losses

def evaluate_by_loss(model, tokenizer, args, device,no_of_datapoints=-1,train=False,test_examples=None, predict_dir=""):

        predict_files = [f for f in listdir(predict_dir) if re.match("^.*json$|^.*jsonl$", f) ] # isfile(join(mypath, f))]
        print("predict dir:%s"%(predict_dir))
        data_file = predict_dir +"/" + predict_files[0]

        if args.read_enc2dec_file:
         eval_dev_examples  = read_enc2dec_examples(
            input_file=predict_dir, is_training=False, version_2_with_negative=args.version_2_with_negative)
         eval_test_examples = eval_dev_examples
        else:
         if test_examples is None:
           eval_dev_examples, version = read_squad_examples(
            input_file=data_file, is_training=False, version_2_with_negative=args.version_2_with_negative, args=args)
         else:
           eval_dev_examples = test_examples
         '''
         eval_dev_examples  = read_squad_examples(
            input_file=data_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
         '''

        if no_of_datapoints != -1:
            eval_dev_examples = eval_dev_examples[:no_of_datapoints]

        cache_name = "_"
        # if tokenizer
        cached_eval_features_file = data_file+'_{0}_{1}_{2}_{3}'.format(
            list(filter(None, cache_name.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))

        try:
            with open(cached_eval_features_file, "rb") as reader:
                eval_features = pickle.load(reader)
        except:
            eval_features = convert_examples_to_features(
             examples=eval_dev_examples,
             tokenizer=tokenizer,
             max_seq_length=args.max_seq_length,
             doc_stride=args.doc_stride,
             max_query_length=args.max_query_length,
             is_training=True,args=args) #,no_question=True)
            with open(cached_eval_features_file, "wb") as writer:
                    pickle.dump(eval_features, writer)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_dev_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)
        all_pretrain_ids = torch.tensor([f.pretrain_id for f in eval_features], dtype=torch.long)
        all_query_ids = torch.tensor([f.query_ids for f in eval_features], dtype=torch.long)
        all_query_input_mask = torch.tensor([f.query_input_mask for f in eval_features], dtype=torch.long)

        #all_tokens = [f.tokens for f in eval_features]
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_example_index, all_pretrain_ids, all_query_ids, all_query_input_mask)
        # Run prediction for full data
        




        if args.local_rank == -1:
           eval_sampler = SequentialSampler(eval_data)
        else:
           eval_sampler = DistributedSampler(eval_data)

        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        logger.info("Start evaluating")
        eval_loss = 0
        total_tokens = 0
        eval_loss_per_file = {data_file:0} #losses per dev file

        no_of_examples=0
        with torch.no_grad():

          for batch in tqdm(eval_dataloader, desc="Evaluating by loss", disable=args.local_rank not in [-1, 0]):

            input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices, pretrain_ids, query_ids, query_input_mask = batch
            if no_of_examples % 1000 == 0:
                logger.info("Processing example: %d" % (no_of_examples))

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            pretrain_ids = pretrain_ids.to(device)
            query_ids = query_ids.to(device)
            query_input_mask = query_input_mask.to(device)

            '''
            Question: does it make sense to reduced the length of the padded passage to 
               run it fast?
            '''

            #print(pretrain_ids)
            #input("pretrain ids")

            outputs = model(input_ids=input_ids, token_type_ids=segment_ids, attention_mask=input_mask, query_ids=query_ids, query_input_mask=query_input_mask, start_positions=start_positions, end_positions=end_positions, pretrain_ids=pretrain_ids, copymethod=args.copymethod, args=args)

            loss = outputs[0]

            eval_loss += loss.mean().item()
            #total_tokens += (segment_ids == 1).sum()
            no_of_examples += args.predict_batch_size

        eval_loss_per_file[data_file]=eval_loss

        print("Evaluation per loss(qg):",eval_loss_per_file)

        return eval_loss, eval_loss_per_file  #/total_tokens


def print_and_file(*text):
    print(*text)
    #print(text)
    predlog_file.write(text[0]+"\n")
    predlog_file.flush()
    #input("ok?")

def print_input_ids(tokenizer, preds,index=0):

    # For List
    #if type(preds) == list or type(: 
    for j in range(len(preds)):
            return_pred = " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j]))] )) 
            print(index, return_pred)

def print_beam(tokenizer, preds, segments,scores=None,start_indices=None,end_indices=None,secondary_scores=None):
    return_pred=""
    best_score=0

    # For List
    #if type(preds) == list or type(: 
    for j in range(len(preds)):
          if j == 0:

            return_pred = " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])[1:-1])
            best_score = scores[j]

          if scores is None:
            print_and_file("%d %s" % (j," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )

          else:
            if secondary_scores is None:
             if start_indices is not None:
                 print_and_file("%d %f %s | Answer %s" % (j,scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])), " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(start_indices[j].item(), end_indices[j].item()+1)  ]))  ))
             else:
               print_and_file("%d %f %s" % (j,scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )

            else:

             if start_indices is not None:
               print_and_file("%d %f %f %s | Answer %s" % (j,scores[j],secondary_scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])), " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(start_indices[j].item(), end_indices[j].item()+1)  ]))  ))
             else:
               print_and_file("%d %f %f %s" % (j,scores[j],secondary_scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )
    return return_pred, best_score


def print_beam_old(tokenizer, preds, segments,scores=None,start_indices=None,end_indices=None,secondary_scores=None):
    return_pred=""
    best_score=0

    # For List
    if type(preds) == list: 
      for j in range(len(preds)):
          if j == 0:
            return_pred = " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])[1:-1])
            best_score = scores[j]
          if scores is None:
            print("%d %s" % (j," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )
          else:
            if secondary_scores is None:
             if start_indices is not None:
               print("%d %f %s | Answer %s" % (j,scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])), " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(start_indices[j].item(), end_indices[j].item()+1)  ]))  ))
             else:
               print("%d %f %s" % (j,scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )
            else:
             if start_indices is not None:
               print("%d %f %f %s | Answer %s" % (j,scores[j],secondary_scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])), " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(start_indices[j].item(), end_indices[j].item()+1)  ]))  ))
             else:
               print("%d %f %f %s" % (j,scores[j],secondary_scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )

    # For Torch Tensor
    elif type(preds) == torch.Tensor: 
      for j in range(preds.size(0)):

          if j == 0:
             return_pred = " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(preds.size(1)) if segments[j][i] == 1 ])[1:-1])
             best_score = scores[j]

          if scores is None:
            print("%d %s" % (j," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(preds.size(1)) if segments[j][i] == 1 ]))) )
          else:
           if start_indices is not None:
             print("%d %f %s | Answer: %s" % (j,scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(preds.size(1)) if segments[j][i] == 1 ])), " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(start_indices[j].item(), end_indices[j].item()+1)  ])) ) )
             #print("Answer: %s" % (" ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(start_indices[j].item(), end_indices[j].item()+1)  ]))) )
           else:
            print("%d %f %s" % (j,scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(preds.size(1)) if segments[j][i] == 1 ]))) )

    return return_pred, best_score

def multi_gpu_generate(model, tokenizer, args, devices, no_of_datapoints=-1,train=False,qlm_model=None,test_examples=None):

   #split and generate on multi-gpu 
   test_len = len(test_examples)
   split_1 = test_len / len(devices)
   test_examples_s = []
   index=0
   for _ in range(len(devices)): 
     test_examples_s.append(test_examples[index:index+split_1])
     index = index+split_1
   #multi-threading
   for m in range(len(devices)):
     spawn = evaluate_by_generate(model)

def evaluate(args, model, tokenizer, prefix="", encoder_model=None):

    if args.loss in ["qa", "prior"]:
        #def evaluate_qa(model, tokenizer, args, device, step=0, no_of_datapoints=-1,train=False,test_examples=None, ppl_only=False, predict_dir=""):
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        results = evaluate_qa(model, tokenizer, args, device,  predict_dir = args.predict_dir)
              #eval_first call:
              #eval_loss, eval_dev_results = evaluate_qa(model, tokenizer, args,device,
              #                             step=global_step, 
              #                             predict_dir=args.dev_dir)

    elif args.loss in ["qg", "genqa", "qae", "classify_q_n_s"]:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        results = evaluate_by_generate(model, tokenizer, args, device, no_of_datapoints=-1, predict_dir=args.predict_dir, train=True, qlm_model=encoder_model)
        #evaluate(model, tokenizer, args,device,no_of_datapoints=-1,predict_dir=args.predict_dir,train=True)
    else: # qae, ae etc
        # if args.loss == "ae"
        #results = evaluate_genqa(args,model, tokenizer, prefix)
        print("Running evaluate_qa.., loss is %s"%(args.loss))
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        results = evaluate_qa(model, tokenizer, args,  device, predict_dir = args.predict_dir, encoder_model=encoder_model)

    print("---Results:", results, "----")
    #input("results")

    return results

def evaluate_by_generate(model, tokenizer, args, device,no_of_datapoints=-1,predict_dir="",qlm_model=None,test_examples=None, train=False):

      
     predict_files = [f for f in listdir(predict_dir) if re.match("^.*json$", f) ] # isfile(join(mypath, f))]
     print("predict dir:%s"%(predict_dir))

     for fil in predict_files:
      
        try :
          lang = re.sub(".json","",fil).split("_")[-2]
        except:
          lang = "en"

        try :
          boq = re.sub(".json","",fil).split("_")[-1]
        except:
          boq = "question" #question"

        args.lang=lang

        # Question:
        if boq == "question" :
          args.boq_id = tokenizer.convert_tokens_to_ids(["[QUESTION]"])[0] 
          EOP="[QUESTION]"
        else: # boq == "gap":
          #EOP="[GAP]" # we trained once gap with with this eop token, so use it by default 
          args.boq_id = tokenizer.convert_tokens_to_ids(["[GAP]"])[0] 
          EOP="[GAP]"

        # GAP:
        if boq == "gap" :
          #EOP="[GAP]" # we trained once gap with with this eop token, so use it by default 
          args.boq_id = tokenizer.convert_tokens_to_ids(["[GAP]"])[0] 
          EOP="[GAP]"
        else: # boq == "gap":
          args.boq_id = tokenizer.convert_tokens_to_ids(["[QUESTION]"])[0] 
          EOP="[QUESTION]"

        '''
        else:
          args.boq_id = tokenizer.convert_tokens_to_ids([sep_token])[0]
          EOP=sep_token #"[QUESTION]"
        '''
        print("-",args.boq_id)

        #with open(predict_dir+"/"+fil,"rb") as f:
        data_file=predict_dir+"/"+fil
        predict_batch_size = 1

        if args.read_enc2dec_file:
          eval_dev_examples = read_enc2dec_examples(
            input_file=predict_dir, is_training=False, version_2_with_negative=args.version_2_with_negative)
          eval_test_examples = eval_dev_examples
        else:
         if test_examples is None:

           #prev=args.encaps_ans
           #args.encaps_ans = True 
           eval_dev_examples, version = read_squad_examples(
            input_file=data_file, 
            is_training=False, 
            version_2_with_negative=args.version_2_with_negative,
            args=args,
            clm=boq) # boq represents -2 field of the json file (usually question/lm/gap)
           eval_test_examples = eval_dev_examples
           #args.encaps_ans = prev 

         else:

           eval_test_examples = test_examples

        '''
        if args.genset == "dev":
           eval_test_examples = eval_dev_examples
        '''

        if no_of_datapoints != -1:
            eval_test_examples = eval_test_examples[:no_of_datapoints]
        #qamodel = BertForQuestionAnswering.from_pretrained("outputdir_squad2.0_bert-base-cased")
        #qamodel = qamodel.cuda()
        #qamodel.eval()
        qamodel = None

        cache_name = "_gen_" + args.bert_model
        # if tokenizer
        cached_eval_features_file = data_file+'_{0}_{1}_{2}_{3}'.format(
            list(filter(None, cache_name.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))

        try:
            with open(cached_eval_features_file, "rb") as reader:
                eval_features = pickle.load(reader)
        except:
            eval_features = convert_examples_to_features(
             examples=eval_test_examples,
             #examples=eval_dev_examples,
             tokenizer=tokenizer,
             max_seq_length=args.max_seq_length,
             #doc_stride=args.doc_stride,
             doc_stride=1390,
             max_query_length=args.max_query_length,
             is_training=True,
             no_question=True,
             is_generate=True,
             eop_token=EOP,args=args) 
            with open(cached_eval_features_file, "wb") as writer:
                    pickle.dump(eval_features, writer)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_test_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)
        all_pretrain_ids = torch.tensor([f.pretrain_id for f in eval_features], dtype=torch.long)
        all_query_ids = torch.tensor([f.query_ids for f in eval_features], dtype=torch.long)
        all_query_input_mask = torch.tensor([f.query_input_mask for f in eval_features], dtype=torch.long)
        all_tokens = [f.tokens for f in eval_features]
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, 
                                  all_input_mask, 
                                  all_segment_ids, 
                                  all_start_positions, 
                                  all_end_positions, 
                                  all_example_index,
                                  all_pretrain_ids,
                                  all_query_ids,
                                  all_query_input_mask,
                                   )

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        total_ngram_matches = 0
        total_denom = 0
    
        #print_only = [45, 366, 450, 451, 453, 454, 455, 460, 477, 672, 677, 1054, 1267, 1315, 1476, 1590, 1595, 1600, 1922, 2243, 2304, 2360, 2361, 2478, 2491, 2501, 2683, 2862, 2887, 3017, 3019, 3039, 3040, 3117, 3142, 3215, 3218, 3275, 3276, 3427, 3490, 4051, 4168, 4182, 4312, 4786, 4820, 4824, 5121, 5231, 5441, 5462, 5529, 5715, 5919, 5924, 5928, 5932, 5990, 5992, 5994, 5995, 6000, 6155, 6158, 6248, 6296, 6311, 6314, 6316, 6329, 6330, 6354, 6392, 6735, 6769, 6803, 6985, 7096, 7331, 7528, 7529, 7622, 7625, 7626, 7676, 7678, 7922, 7948, 8017, 8021, 8189, 8273, 8735, 8835, 9340, 9741, 9745, 9758, 10000, 10227, 10354, 10408, 10578, 10650, 10653, 10675, 10699, 10700, 10715, 10838, 10866, 10881, 10938, 11031, 11032, 11033, 11034, 11035, 11036, 11037, 11308, 11658, 11780, 11846]

        with torch.no_grad():
         for batch in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):


            input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices, pretrain_ids, query_ids, query_input_mask =  batch
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))

  
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
            pretrain_ids = pretrain_ids.to(device)
            query_ids = query_ids.to(device)
            query_input_mask = query_input_mask.to(device)

            #Batch size = 1, remove the pads
            input_length = (input_mask!=0).sum(-1).item() 
            input_ids = input_ids[:,:input_length]
            input_mask = input_mask[:,:input_length]
            segment_ids = segment_ids[:,:input_length]
            #start_positions = start_positions[:,:input_length]
            #end_positions = end_positions[:,:input_length]
            #print("gen func:%s" % (input_ids))
            #input("input ids evaluate by gen func")

            references = [ re.sub("[     ]*\[SEP\]","",re.sub(".*\[MASK\]",""," ".join(all_tokens[i]))) for i in example_indices ]
            #references = [ re.sub(".*\[MASK\]",""," ".join(all_tokens[i])) for i in example_indices ]
            #ngram_match, denom = generate_questions(model, tokenizer, input_ids, segment_ids,input_mask, start_positions, end_positions, references,args)

            inputs = [ input_ids, segment_ids,input_mask, pretrain_ids, query_ids, query_input_mask, start_positions, end_positions, references ]

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
            #print(inputs)
            #print(model.device)
            #input("###")

            if args.loss in ["qae"]:

              gen_tup = generate_from_embedding(model, tokenizer, inputs, 20, args,qamodel, beam_size=args.beam_size)

              start_positions=torch.zeros_like(start_positions)
              end_positions=torch.zeros_like(start_positions)

            elif args.loss in ["lm"] :
              gen_tup = lm_sampling(model, tokenizer, inputs, 20, args,qamodel, beam_size=args.beam_size)
            else:
              #gen_tup = translate_batch(model, tokenizer, inputs, 20, args,qamodel, beam_size=args.beam_size, qlm_model=qlm_model)
              gen_tup = translate_batch(model, tokenizer, inputs, 20, args,qamodel, beam_size=args.beam_size, encoder_model=qlm_model)


            #print(gen_tup["predictions"])
            #input("")

            preds = gen_tup['predictions']
            segments = gen_tup['segments']
            scores = gen_tup['scores']

            #sort scores
            sorted_indices = [i[0] for i in sorted(enumerate(scores[0]), key=lambda x:x[1].item(),reverse=True)]

            preds = [[ preds[0][ind] for ind in sorted_indices ]]
            scores = [[ scores[0][ind] for ind in sorted_indices ]]
            segments = [[ segments[0][ind] for ind in sorted_indices ]]

            # make all preds the same
            #max_len=max([len(preds[0][j]) for j in range(len(preds[0])) ])


            '''
            for pred in preds[0]:
              while pred.size(0) < max_len:
                pred=torch.cat((pred,torch.zeros(1).to(pred.device)),dim=-1)
            for seg in segments[0]:
              while seg.size(0) < max_len:
                seg=torch.cat((seg,torch.zeros(1).to(seg.device)),dim=-1)
            '''
            #preds = [[ preds[0][ind] for ind in sorted_indices ]]
            

            new_start_indices=None
            new_end_indices=None
            prev_scores = None


            if args.rerank:
              prev_scores = scores[0]
              preds,segments,scores,new_start_indices, new_end_indices = rerank(preds, segments, scores,input_ids,input_mask,qamodel,start_positions,end_positions,top=5) # only rerank top 5 
            else:
              new_start_indices = torch.zeros(len(preds[0])).long().cuda() + start_positions - 1 # -1 because [CLS] token is removed in the preds
              new_end_indices = torch.zeros(len(preds[0])).long().cuda() + end_positions - 1 # -1 because [CLS] token is removed in the preds
            

            print_and_file("--predicted on file--- %s" % (fil))
            print_and_file("context: %s" % ([" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0])) for b in range(input_ids.size(0)) ][0]))
            print_and_file("--N-Best Predictions--")

            #To print beam
            best_pred_text, best_score = print_beam(tokenizer, preds[0], segments[0],scores=scores[0],start_indices = new_start_indices,end_indices=new_end_indices,secondary_scores=prev_scores)
            best_pred_text = re.sub("( \[SEP\])+","",best_pred_text)
            best_pred_text = re.sub("( \[PAD\])+","",best_pred_text)
            best_pred_text = re.sub("( \?)+"," ?",best_pred_text)

            print_and_file("%s: %s"%("#Reference",references[0]))
            print_and_file("#Predicted: %s " %(best_pred_text))
            #print(best_score)
            
            pred_file.write(best_pred_text+"\n")
            ref_file.write(references[0]+"\n")
            pred_file.flush()
            ref_file.flush()

            #input("generation")

            #ngram_match, denom = calculate_ngram_match([references[0]], [n[0]],n=2)
            ngram_match = 0
            denom = 1
            #print("local bleu:",ngram_match/denom)

            total_ngram_matches += ngram_match
            total_denom += denom

        # To avoid divided by zero error
        total_denom = total_denom + 1
        logger.info("BLEU Score on DEV set: %f " % (total_ngram_matches / total_denom) )


# For reproducability
def set_seed(args, n_gpu):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    

def main():

    print("main()")
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                        "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                        "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--qlm_model", default=None, type=str, required=False,
                        help="a language model on questions- used while decoding for pmi")
    parser.add_argument("--encoder_model", default=None, type=str, required=False,
                        help="an encoder model is not optimized further, but used for obtaining pre-trained encoded embeddings")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--genfrom", default=-1, type=int,
                        help="generate from the datapoint (.)")
    parser.add_argument("--gento", default=99999, type=int,
                        help="generate till the datapoint (.)")
    parser.add_argument("--train_file", default=None, type=str, help="SQuAD json for training. E.g., train-v1.1.json")
    parser.add_argument("--train_dir", default=None, type=str, help="training directory where all train*json are present. E.g., train-v1.1.json")
    parser.add_argument("--predict_dir", default=None, type=str, help="predict directory where all dev*json are present. E.g., train-v1.1.json")
    parser.add_argument("--dev_dir", default=None, type=str, help="dev directory where all dev*json are present. E.g., train-v1.1.json")
    parser.add_argument("--eval_on", default="ppl", type=str, help="evaluation criteria: ppl/F1/EM etc")

    parser.add_argument("--predict_file", default=None, type=str,
                        help="SQuAD json for predictions. E.g., dev-v1.1.json or test-v1.1.json")
    parser.add_argument("--predict_file_w_sent_bounds", default=None, type=str,
                        help="SQuAD json with sentence boundaries for QA predictions. E.g., dev-v1.1.json or test-v1.1.json")
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
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--num_iter_on_files", default=3.0, type=float,
                        help="similar to epoch number.")
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
    parser.add_argument( "--config_name", default="", type=str, help="Pretrained config name or path if not the same as model_name")
    parser.add_argument( "--cache_dir", default="", type=str, help="Where do you want to store the pre-trained models downloaded from s3",) 
    parser.add_argument( "--tokenizer_name", default="", type=str, help="Pretrained tokenizer name or path if not the same as model_name",)
    parser.add_argument("--loss", default=None, type=str, required=True,
                        help="genqa/qa/qgen/both")
    parser.add_argument('--eval_first', action='store_true', help="generate few questions during evaluation of training")
    parser.add_argument('--debug', action='store_true', help="debug True")
    parser.add_argument('--lang_finetune', action='store_true', help="language fine-tune")
    parser.add_argument('--lang_freeze', action='store_true', help="to freeze lang_linear layers")
    parser.add_argument('--lang', default="en", type=str, help="language to generate : en/de/es etc")
    parser.add_argument('--boq_id', type=int, default=102, help="id of the begin-of-question token , default : [SEP] (102); other options:  [QUESTION](99 token id)")
    parser.add_argument('--lang_projection_layer', action='store_true', help="to have different language layer for prediction")
    parser.add_argument('--no_mask', action='store_true', help="remove [MASK] in the input paragraph")
    parser.add_argument('--encaps_ans', action='store_true', help="encapsulate answer phrase")
    parser.add_argument('--no_pretrain_embs', action='store_true', help="to not have pretrain_embeddings as input")
    parser.add_argument('--no_enc_position', action='store_true', help="to not have positions for encoder")
    parser.add_argument('--reverse_seq', action='store_false', help="to do qg in reverse order i.e., question || paragraph")

    parser.add_argument('--add_q_emb', action='store_true', help=" to add q emb to the bert output embeddings before generating question")

    parser.add_argument('--add_target_emb', action='store_true', help=" to add target sentence emb to the bert output embeddings before generating it")
    parser.add_argument('--add_target_only', action='store_true', help=" to add target sentence emb based on target sentence only")

    parser.add_argument('--grl', action='store_true', help="Use gradient reverse layer wherever applicable")
    parser.add_argument('--ignore_prev_train', action='store_true', help="Ignore previous learning rate and other parameters for optimization")
    parser.add_argument('--segment_position_independence', action='store_true', help="to have independent numbering for question and paragraph")
    parser.add_argument('--no_absolute_position', action='store_true', help="take position id embeddings seperately")
    parser.add_argument('--qg_with_sim',
                        type=float, default=0,
                        help="qg only:0, qg with sim: 1 sim only: >1 Regularizer to minize the distance between phrase vector and the question vector")
    parser.add_argument("--loss2", default=None, type=str, required=False,
                        help="dummy loss - not used")

    parser.add_argument('--para_output_only', action='store_true', help="to do qg in reverse order i.e., question || paragraph")
    parser.add_argument('--for_all_target', action='store_true', help="to add targetemb/qemb for all the target sentence words not just at [QUESTION] token")
    parser.add_argument('--insert_q_emb_to_cls', action='store_true', help="to add qemb to cls word embedding before BERTQA on paragraph")

    parser.add_argument('--only_grl', action='store_true', help="only grl will exclude other losses")
    parser.add_argument('--evaluate_by_loss', action='store_true', help="only grl will exclude other losses")
    #parser.add_argument('--encoder_loss', action='store_true', help="only grl will exclude other losses")
    parser.add_argument("--encoder_loss", default=None, type=str, help="-loss used for encoder_model : should have the same values as --loss")
    parser.add_argument('--only_eval_first', action='store_true', help="only grl will exclude other losses")
    parser.add_argument('--only_contrastive_loss', action='store_true', help="only contrastive loss")
    parser.add_argument('--contrastive_loss', action='store_true', help="only contrastive loss")
    parser.add_argument('--save_anyway', action='store_true', help="to save models anyway regard less of better eval")
    parser.add_argument('--shallow_discr', action='store_true', help="shallow discr instead of insert_to_cls_emb ")
    parser.add_argument('--subtract_embeddings', action='store_true', help="subtract emeddings")
    parser.add_argument('--asquare_embeddings', action='store_true', help="subtract emeddings")
    parser.add_argument('--remove_emb_at_phr_pos', action='store_true', help="subtract emeddings")
    parser.add_argument('--cloze_id', action='store_true', help="cloze id")
    parser.add_argument('--mse', action='store_true', help="to do mse inaddition to aae")
    parser.add_argument('--lang_loss', action='store_true', help="langloss")
    parser.add_argument("--aae_type", default='pos', type=str, help="type of aae loss")




    args = parser.parse_args()
    print(args)
    print("debug 0")
    print("debug 1")
    print("debug 2")

    # Creat directory if necessary
    '''
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train:
        raise ValueError("Output directory () already exists and is not empty.")
    '''
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

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
       filetag = "_cp_"+str(args.copymethod)+"."+filetag

       filetag = filetag +"."+args.tag

       ref_file = open(args.output_dir+"/references."+filetag+".txt","w")
       pred_file = open(args.output_dir+"/preds."+filetag+".txt","w")
       context_file = open(args.output_dir+"/contexts."+filetag+".txt","w")
       predlog_file = open(args.output_dir+"/qgeneration."+filetag+".txt","w")

    if True: # should allow only when local_rank == -1?
      log_file = open("qg_logs/log."+filetag+".txt","w")
      log_file.write(str(sys.argv[0])+"\n")
      log_file.write(str(args)+"\n")
      log_file.flush()

      if args.gen_during_train:
        ref_file = open(args.output_dir+"/references."+filetag+".txt","w")
        pred_file = open(args.output_dir+"/preds."+filetag+".txt","w")
        context_file = open(args.output_dir +"/contexts."+filetag+".txt","w")
        predlog_file = open(args.output_dir +"/qgeneration."+filetag+".txt","w")


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
        if not args.train_dir:
            raise ValueError(
                "If `do_train` is True, then `train_dir` must be specified.")

    if args.do_predict:
        if not args.predict_dir:
            raise ValueError(
                "If `do_predict` is True, then `predict_dir` must be specified.")


    print("debug 4124")
    #input("tokenizer case")

    global cls_token
    global mask_token
    global sep_token

    config_class, model_class, tokenizer_class, cls_token, mask_token, sep_token = MODEL_CLASSES[args.model_type]

    #tokenizer = tokenizer_class.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    args.model_type = args.model_type.lower()
    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.bert_model,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )


    print(" **** Loading Models ****** ")
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.bert_model,
        do_lower_case=args.do_lower_case,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print("Loaded Tokenizer")

    # Add special tokens
    #special_tokens_dict = {'additional_special_tokens': ['[START]','[END]']}
    special_tokens_dict = {'additional_special_tokens': ['[QUESTION]']} 
    tokenizer.add_special_tokens(special_tokens_dict)

    #model = AutoModelForQuestionGeneration.from_pretrained(
    model = BertForQuestionGeneration.from_pretrained(
        args.bert_model,
        from_tf=bool(".ckpt" in args.bert_model),
        config=config,
        cache_dir=args.cache_dir if args.cache_dir else None,
    )
    print("Loaded Model from : %s"%(args.bert_model)) 

    if args.encoder_model:

          encoder_model = BertForQuestionGeneration.from_pretrained(
            args.encoder_model,
            from_tf=bool(".ckpt" in args.bert_model),
            config=config,
            cache_dir=args.cache_dir if args.cache_dir else None,
          )
          print("Loaded Encoder Model from : %s"%(args.encoder_model))

    else:
          encoder_model = None

     
    prev_global_step = -1 
    if os.path.isfile(args.bert_model+"/training_trajectory.pkl"):
        print(" **** Training Details of the loaded Model ****** ")
        with open(args.bert_model+"/training_trajectory.pkl","rb") as f:
                          prev_train_tup=pickle.load(f)
                          prev_global_step=prev_train_tup[-1][0]
                          print(prev_train_tup)
                          print("trained till",prev_global_step)

    if args.ignore_prev_train:
       prev_global_step = -1 
    print("debug 4188, loaded model")



    '''
    model = BertForQuestionGeneration.from_pretrained(args.bert_model,
                cache_dir=args.cache_dir if args.cache_dir else None)
      
    '''
    
    if args.lang_finetune:

      for parameter in model.parameters():
        parameter.requires_grad = False

      #target_sent_emb = self.question_projection(target_sent_emb)
      #target_sent_emb = target_sent_emb.matmul(self.question_translation_matrix) #target_sent_emb)

      model.question_projection.weight.requires_grad=True
      model.question_projection.bias.requires_grad=True

      model.question_translation_matrix.requires_grad=True

      model.auto_source_linear.weight.requires_grad=True
      model.auto_source_linear.bias.requires_grad=True

      print("auto source linear finetuning only")

    '''
    '''
    '''
      #Final layer
      model.lang_linear_en.weight.requires_grad = True
      model.lang_linear_en.bias.requires_grad = True
      model.lang_linear_de.weight.requires_grad = True
      model.lang_linear_de.bias.requires_grad = True

      #Embedding layer
      model.lang_embeddings.weight.requires_grad = True
      model.lang_matrices.weight.requires_grad = True
    '''

    if args.lang_freeze:
      #Final layer
      model.lang_linear_en.weight.requires_grad = False
      model.lang_linear_en.bias.requires_grad = False
      model.lang_linear_de.weight.requires_grad = False
      model.lang_linear_de.bias.requires_grad = False

      #Embedding layer
      model.lang_embeddings.weight.requires_grad = False
      model.lang_matrices.weight.requires_grad = False



    print("Model edited as per paramters given")

    if args.fp16:
        print("using fp16")
        model.half()
        if args.encoder_model:
            encoder_model.half()
    print("Moving model to ", device)

    model.to(device)
    if args.encoder_model:
            encoder_model.to(device)

    print("Model moved to device")

    if args.local_rank != -1:
        print("distributed model; local_rank=", args.local_rank)
        log_file.write("distributed model; local_rank= %s\n"%(args.local_rank))
        log_file.flush()
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        if args.encoder_model:
          encoder_model = torch.nn.parallel.DistributedDataParallel(
            encoder_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        
    elif n_gpu > 1:

        print("n_gpu>1; dataparallel model; local_rank= %s\n"%(args.local_rank))
        log_file.write("n_gpu>1; dataparallel model; local_rank= %s\n"%(args.local_rank))
        log_file.flush()
        model = torch.nn.DataParallel(model)
        if args.encoder_model:
          encoder_model = torch.nn.DataParallel(encoder_model)


    ### Use freezing after DistrubtedParallel(Model)

    '''
    if args.lang_finetune:
      for parameter in model.parameters():
        parameter.requires_grad = False

      #target_sent_emb = self.question_projection(target_sent_emb)
      #target_sent_emb = target_sent_emb.matmul(self.question_translation_matrix) #target_sent_emb)

      model.module.question_projection.weight.requires_grad=True
      model.module.question_projection.bias.requires_grad=True

      model.module.question_translation_matrix.requires_grad=True

      model.module.auto_source_linear.weight.requires_grad=True
      model.module.auto_source_linear.bias.requires_grad=True
    '''


    print("just before training..")
    # Prepare optimizer
    if args.do_train:

     # This code is redundant but important to get the num_train_optimization steps
     #begin reference
     train_files = [f for f in listdir(args.train_dir) if re.match("^.*json$|^.*jsonl$|^.*txt$", f) ] # isfile(join(mypath, f))]
     train_files = sorted(train_files)
     file_count = len(train_files)
     print("file count %s" %(file_count))


     # Take random files 5 times

     length_of_examples = 1
     no_of_samples = file_count #int(0.1*file_count) + 1 # 10% of the files
     random.shuffle(train_files)
     for i in range(no_of_samples):
      train_file = args.train_dir+"/"+train_files[i]
      print("%s"%(train_files[i]))
      with open(train_file, "r") as f:
        if re.match("^.*json$",train_files[i]):
           data=json.load(f)["data"]
           length=0
           for j in range(len(data)):
             length+=sum([ len(data[j]["paragraphs"][i]["qas"]) for i in range(len(data[j]["paragraphs"])) ])
        elif re.match("^.*txt$",train_files[i]): 
           data=[]
           for line in f:
              line=line.strip()
              data.append(line)
           length=len(data)
        elif re.match("^.*jsonl$",train_files[i]): 
           data=[]
           length=0
           for line in f:
              line=line.strip()
              line_data=json.loads(line)
              data.append(line_data)
           for j in range(len(data)):
             if "header" in data[j].keys():
                   continue
             length+=len(data[j]["qas"]) 

      print("%s %s" %(length, train_file))
      length_of_examples += length
     length_of_examples = length_of_examples / no_of_samples
     print("%s %s" %(length_of_examples, length_of_examples*file_count))
        
     total_epochs = args.num_train_epochs * args.num_iter_on_files


     #args.train_batch_size has become (.) // gradient_accumulation_steps

     num_train_optimization_steps = int(
            length_of_examples*file_count / args.train_batch_size / args.gradient_accumulation_steps) * total_epochs

     if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
     print("num of optimization steps = %s,file_count:%s, last-file length: %s\n" %(num_train_optimization_steps,file_count, length))
     log_file.write("num of optimization steps = %s\n" %(num_train_optimization_steps))
     log_file.write("num of train_files = %s\n" %(file_count))
     log_file.write("total num of examples = %s; %s\n" %(length_of_examples*file_count,length))
     log_file.flush()


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
        '''
        print(optimizer_grouped_parameters,
                             args.learning_rate,
                             args.warmup_proportion,
                             num_train_optimization_steps)
        '''
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)
        #if os.path.isfile(os.path.join(args.bert_model, "optimizer.pt")):
        # Here we ignore previous optimizer states if mentioned in the parameters : args.ignore_prev_train
        if os.path.isfile(os.path.join(args.bert_model, "optimizer.pt")) and (not args.ignore_prev_train):
           optimizer.load_state_dict(torch.load(os.path.join(args.bert_model, "optimizer.pt")))


    # Added here for reproductibility
    set_seed(args,n_gpu)

    if args.do_train:
     path=args.train_dir
     train_files = [f for f in listdir(path) if re.match("^.*json$|^.*jsonl$|^.*txt$", f) ] # isfile(join(mypath, f))]
     train_files = sorted(train_files)
     total_steps = 0
     global_step = 1
     lrs=[]
     lr = 0

     for ep_f in trange(int(args.num_iter_on_files), desc="Epoch"):

      for fil in train_files:
        #lang=re.sub("^.*_([a-z]+)_.*json",\1, fil)
        try :
          lang = re.sub(".json|.txt","",fil).split("_")[-2]
        except:
          lang = 'en'

        if lang in ALLOWED_LANGS:
              args.lang = lang
        else:
              args.lang = "en"
              #print("Skipping %s file for training, %s not allowed" % (train_file,lang))
              print("Using lang as en")

        clm = re.sub(".json|.txt","",fil).split("_")[-1] # The type of Conditional Language Modeling

        if clm == "question":
            EOP="[QUESTION]"
            args.boq_id = tokenizer.convert_tokens_to_ids(["[QUESTION]"])[0] 
        elif clm == "lm":  
            print("clm - %s"%(clm))
            EOP="[QUESTION]"
            args.boq_id = tokenizer.convert_tokens_to_ids(["[QUESTION]"])[0] 
            print("boq - %s"%(args.boq_id))
        elif clm == "gap":
            EOP="[GAP]"
            args.boq_id = tokenizer.convert_tokens_to_ids(["[GAP]"])[0] 
            #EOP="[QUESTION]"
            #args.boq_id = tokenizer.convert_tokens_to_ids(["[QUESTION]"])[0] 
        else:
            EOP=None
            EOP="[QUESTION]"
            args.boq_id = tokenizer.convert_tokens_to_ids(["[QUESTION]"])[0] 
            ''' None is default:'''

            #EOP="[QUESTION]"
            #args.boq_id = tokenizer.convert_tokens_to_ids(["[QUESTION]"])[0] 

        print("boq_id: %s, eop: %s" %(args.boq_id, EOP))
        log_file.write("boq_id:%s, eop: %s ..\n" % (args.boq_id, EOP))


        #for ep in trange(int(args.num_train_epochs), desc="Epoch"):
        train_file = args.train_dir +"/"+fil 
        print("Loading %s file for training.." % (train_file))
        log_file.write("Loading %s file for training..\n" % (train_file))
        log_file.flush()

        if clm == "lm":
          train_examples = read_lm_examples(input_file=train_file,
                                                is_training=True,
                                                version_2_with_negative=args.version_2_with_negative)
        else:
          train_examples, version = read_squad_examples(input_file=train_file,
                                                is_training=True,
                                                version_2_with_negative=args.version_2_with_negative, args=args)

        print("%s datapoints in %s"%(len(train_examples), train_file))


        if args.data_cache_name:
             cache_name = args.data_cache_name
        else:
             cache_name = args.bert_model

        
        cached_train_features_file = train_file+'_{0}_{1}_{2}_{3}'.format(
            list(filter(None, cache_name.split('/'))).pop(), str(args.max_seq_length), str(args.doc_stride), str(args.max_query_length))

        if not args.reverse_seq:
            print("intended: question and paragraph")
            cached_train_features_file = cached_train_features_file + "_rev"
            print("cached train feature file:", cached_train_features_file)
        else:
            print("para and question")

        train_features = None

        try:
            print("cached_train_features_file:", cached_train_features_file)
            
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
            print("loaded features file from cache file")
        except:

          if clm == "lm":
            train_features = convert_lmexamples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True,
                eop_token=EOP)
          else:
            if version == "hotpot":
              print("version hotpot, converting into hotpot examples")
              train_features = convert_hotpotexamples_to_features(examples=train_examples,
                                                tokenizer=tokenizer,
                                                max_seq_length=args.max_seq_length,
                                                doc_stride=args.doc_stride,
                                                max_query_length=args.max_query_length,
                                                is_training=True,
                                                cls_token_segment_id=2 if args.model_type in ['xlnet'] else 0,
                                                pad_token_segment_id=3 if args.model_type in ['xlnet'] else 0,
                                                cls_token_at_end=True if args.model_type in ['xlnet'] else False,
                                                sequence_a_is_doc=True if args.model_type in ['bert'] else False)


            else:
              train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                doc_stride=args.doc_stride,
                max_query_length=args.max_query_length,
                is_training=True,
                eop_token=EOP,args=args)

            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)

        logger.info("***** Running training *****")
        logger.info(" %s Num orig examples = %d", args.local_rank, len(train_examples))
        logger.info(" %s Num split examples = %d", args.local_rank, len(train_features))
        logger.info(" %s Batch size = %d", args.local_rank, args.train_batch_size)
        logger.info(" %s Num steps = %d", args.local_rank, num_train_optimization_steps)

        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in train_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in train_features], dtype=torch.long)
        all_pretrain_ids = torch.tensor([f.pretrain_id for f in train_features], dtype=torch.long)

        
        ''' These attributes are optional depending on the json file '''
        all_phr_start_positions = torch.tensor([f.phr_start_position if "phr_start_position" in f.__dict__.keys() else 0 for f in train_features], dtype=torch.long)
        all_phr_end_positions = torch.tensor([f.phr_end_position if "phr_end_position" in f.__dict__.keys() else 0  for f in train_features], dtype=torch.long)
        #all_phr_end_positions = torch.tensor([f.phr_end_position for f in train_features], dtype=torch.long)

        all_query_ids = torch.tensor([f.query_ids for f in train_features], dtype=torch.long)
        all_query_input_mask = torch.tensor([f.query_input_mask for f in train_features], dtype=torch.long)

        #all_tokens = [f.tokens for f in train_features]

        #all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)

        train_data = TensorDataset(all_input_ids, 
                                   all_input_mask, 
                                   all_segment_ids,
                                   all_start_positions, 
                                   all_end_positions,
                                   all_pretrain_ids,
                                   all_query_ids,
                                   all_query_input_mask,
                                   all_phr_start_positions, 
                                   all_phr_end_positions,
                                   )
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)

        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        model.train()
        best_eval_loss=999999999
        tr_loss = 0
        tracemalloc.start()

        if args.eval_first and args.local_rank in [-1,0]: # only call evaluate once in distributed model evaluation
            if args.gen_during_train:
               evaluate_by_generate(model, tokenizer, args,device,no_of_datapoints=10,predict_dir=args.predict_dir,train=True)

            if args.loss not in ["qg", "qae"]:

              #eval_loss, eval_loss_per_file = evaluate_qa(model,
              #                            tokenizer,
              #                            args, device,
              #                            step=0,
              #                            predict_dir=args.dev_dir,loss="qa")

              '''
              evaluate(args, model, tokenizer, prefix="do_predict devset", encoder_model=encoder_model)
              '''
              eval_loss, eval_dev_results = evaluate_qa(model, tokenizer, args,device,
                                           step=global_step, 
                                           predict_dir=args.dev_dir, encoder_model=encoder_model)

              eval_loss_per_file = eval_dev_results

              #print(eval_loss_per_file)
              #input("-wait-")

                                          #ppl_only=True,

              print("At the beginning, test loss = "+str(eval_loss))
              print("At the beginning, test F1 per file= "+str(eval_loss_per_file))

              log_file.write("Initial test F1 "+str(eval_loss)+"\n")
              log_file.write("At the beginning, test F1 per file= "+str(eval_loss_per_file))
              log_file.flush()
              logger.info("Initial test F1 "+str(eval_loss)+"\n")

        if args.only_eval_first:
            sys.exit(0)

        # For debugging in-place operation in models; It consumes a lot of time if you run with this
        # torch.autograd.set_detect_anomaly(True)

        for ep in trange(int(args.num_train_epochs), desc="Epoch"):
            gc.collect()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])):


                total_steps += 1
                cal_global_step = (total_steps)  // args.gradient_accumulation_steps
                #print("step:", step, total_steps)
                #print("Previosly trained: %s <= %s" %(cal_global_step, prev_global_step))
                if cal_global_step <= prev_global_step:
                    print("step:", step, total_steps)
                    print("Previosly trained: %s <= %s" %(cal_global_step, prev_global_step))
                    continue # Ignore previously trained 

                #if n_gpu == 1:

                batch = tuple(t.to(device) for t in batch) # multi-gpu does scattering it-self
                #all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_pretrain_ids, all_query_ids, all_query_input_mask, all_phr_start_positions, all_phr_end_positions,)
                input_ids, input_mask, segment_ids, start_positions, end_positions, pretrain_ids, query_ids, query_input_mask, phr_start_positions, phr_end_positions  = batch


                # If you are in doubt, print for debugging

                '''
                questions = [" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] != 0][1:])) for b in range(input_ids.size(0)) ]
                contexts =  [" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0])) for b in range(input_ids.size(0)) ]
                answer_phrases = [ " ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0 \
             and  (s >= start_positions.item() and s <= end_positions.item()) ])) for b in range(input_ids.size(0))  ]
                print("answer: %s" %(answer_phrases))
                print("question:%s"%(questions))
                print("context:%s"%(contexts))
                input("\nanswer, question, context")
                '''

                # Get proper input mask for Question Generation


                # For seperate encoder and decoder, we pass the encoded vector as input in the forward function

                if args.encoder_model:


                       
                    with torch.no_grad():

                     '''
                     if args.amalgamated_sentences:
                      #outputs = seperate_encoder(input_ids=input_ids, token_type_ids=
                      encoder_output1 = encoder_model(input_ids=input_ids, 
                        token_type_ids=segment_ids, 
                        attention_mask=input_mask, 
                        start_positions=start_positions, 
                        end_positions=end_positions,
                        query_ids = query_ids,
                        query_input_mask = query_input_mask,
                        copymethod=args.copymethod, 
                        phr_start_positions=phr_start_positions,
                        phr_end_positions=phr_end_positions,
                        args=args, loss='qg', encoding_only=True)
                     '''
                     encoder_output = encoder_model(input_ids=input_ids, 
                        token_type_ids=segment_ids, 
                        attention_mask=input_mask, 
                        start_positions=start_positions, 
                        end_positions=end_positions,
                        query_ids = query_ids, 
                        query_input_mask = query_input_mask,
                        copymethod=args.copymethod, 
                        phr_start_positions=phr_start_positions,
                        phr_end_positions=phr_end_positions,
                        args=args, loss='qg', encoding_only=True)
                else:
                     encoder_output=None

                     #print(encoder_output.size())
                     #encoder_output = encoder_output.unsqueeze(1) # b x 1 x hidden_size
                     


                outputs = model(input_ids=input_ids, 
                        token_type_ids=segment_ids, 
                        attention_mask=input_mask, 
                        pretrain_ids=pretrain_ids, 
                        start_positions=start_positions, 
                        end_positions=end_positions,
                        query_ids = query_ids, 
                        query_input_mask = query_input_mask,
                        copymethod=args.copymethod, 
                        phr_start_positions=phr_start_positions,
                        phr_end_positions=phr_end_positions,
                        args=args, question_emb = encoder_output)

                '''
                outputs2 = model(input_ids=input_ids, 
                        token_type_ids=segment_ids, 
                        attention_mask=input_mask, 
                        pretrain_ids=pretrain_ids, 
                        start_positions=start_positions, 
                        end_positions=end_positions,
                        query_ids = query_ids, 
                        query_input_mask = query_input_mask,
                        copymethod=args.copymethod, 
                        phr_start_positions=phr_start_positions,
                        phr_end_positions=phr_end_positions,
                        args=args, encoded_vector = encoder_output)
                '''

                loss = outputs[0]

                tr_loss += loss.mean().item()

                #print("loss=",loss)

                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles
                        # this automatically

                        lr_this_step = args.learning_rate * warmup_linear.get_lr(global_step/num_train_optimization_steps,
                                                                                 args.warmup_proportion)


                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step
                    lr=optimizer.get_lr()
                    #print("lr=",lr)
                    #input("lr")
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                if (total_steps+1) % args.training_sanity_check_steps  == 0 and args.local_rank in [0,-1]: # only evaluate on the first gpu
                   if args.gen_during_train:
                     pred_file.write("- evaluating at step %s:"%(total_steps))
                     pred_file.flush()
                     predlog_file.write("## Evaluating at step %s ###:"%(total_steps))
                     predlog_file.flush()
                     evaluate_by_generate(model, tokenizer, args,device,no_of_datapoints=10,predict_dir=args.predict_dir,train=True)

                   #eval_loss = evaluate_by_loss(model, tokenizer, args,device)
                   if args.loss in ["qa","prior", "genqa", "qa_by_qemb", "aae"] and not args.evaluate_by_loss: # Is it justified to use this for choosing better model? the training loss is already a good measure for early stopping

                     eval_loss, eval_dev_results = evaluate_qa(model, tokenizer, args,device, 
                                           step=global_step, 
                                           predict_dir=args.dev_dir, encoder_model=encoder_model)


                     print("eval:", eval_loss)
                     print("eval dev results:", eval_dev_results)

                     # Negate the F1 score so that it becomes a kind of loss
                     eval_loss = - eval_loss

                     #input("0")
                     eval_pred_loss = eval_loss
                     eval_pred_results = eval_dev_results

                     #eval_pred_loss, eval_pred_results = evaluate_qa(model, tokenizer, args,device, 
                     #                      step=global_step, 
                     #                      predict_dir=args.predict_dir)

                     eval_loss_per_file = eval_dev_results

                     log_file.write("###### evaluation on files from (predict dir) : %s ##### \n" %(args.predict_dir))
                     log_file.write("predict QA EM/F1: "+eval_pred_results+"\n")
                     log_file.write("predict PPL: "+str(eval_pred_loss) +"\n")
                     log_file.flush()
                   else:

                     #eval_loss, eval_loss_per_file = evaluate(model, tokenizer, args,device, step=global_step, ppl_only=True, predict_dir=args.dev_dir)
                     eval_loss, eval_loss_per_file = evaluate_by_loss(model, tokenizer, args,device, predict_dir=args.dev_dir)
                     #evaluate_by_loss(args, model, tokenizer, prefix="do_predict testset")

                   #not_used_loss, eval_loss_per_file = evaluate_qa(model, tokenizer, args,device, step=global_step, ppl_only=True, predict_dir=args.predict_dir)
                   #eval_loss_per_file = evaluate_qa(model, tokenizer, args,device, step=global_step)
                   lrs.append((global_step,lr, tr_loss, eval_loss_per_file ))

                   current, peak = tracemalloc.get_traced_memory()
                   print(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB")
                   #log_file.write(f"Current memory usage is {current / 10**6}MB; Peak was {peak / 10**6}MB\n")
                   log_file.write("Current memory usage is %s MB; Peak was %s MB" %((current / 10**6) ,(peak / 10**6)) )

                   logger.info("At step %d, training file:%s" % (step, train_file))
                   logger.info("Average TR loss %f" % (tr_loss/global_step))
                   logger.info("Dev PPL %s" % (eval_loss))

                   log_file.write("\n###### evaluation on files from (dev dir) : %s ##### \n" %(args.dev_dir))
                   log_file.write("#DEV PPL "+str(eval_loss)+", best eval loss :" + str(best_eval_loss) +"\n")
                   log_file.write("#Average Training loss for  "+str(global_step)+" global optimization steps="+str(tr_loss/global_step)+"\n")
                   log_file.flush()
                   better_eval=False

                   if best_eval_loss > eval_loss:
                      best_eval_loss = eval_loss
                      better_eval=True

                   if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and (better_eval or args.save_anyway):
                       # Save a trained model, configuration and tokenizer
                       logger.info("Better eval found")
                       log_file.write("Better eval found\n")
                       log_file.flush()
                       model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

                       # If we save using the predefined names, we can load using `from_pretrained`
                       output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
                       output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                       torch.save(model_to_save.state_dict(), output_model_file)
                       torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
                       with open(args.output_dir+"/training_trajectory.pkl","wb") as f:
                          pickle.dump(lrs,f)

                       logger.info("Saving model checkpoint to %s", args.output_dir)

                       torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))

                       model_to_save.config.to_json_file(output_config_file)
                       tokenizer.save_vocabulary(args.output_dir)

            gc.collect()

            # At the end of an epoch, evaluate and save
            #eval_loss = evaluate_by_loss(model, tokenizer, args,device)
            #logger.info("At the end of epoch %d" % (ep))
            #logger.info("Average TR loss %f" % (tr_loss/step))
            #logger.info("Dev PPL %f" % (eval_loss))

            #log_file.write("Dev PPL "+str(eval_loss)+"\n")
            #log_file.write("Average Training loss for "+str(step)+" steps="+str(tr_loss/step)+"\n")
            #log_file.flush()
            #better_eval=False
            #if best_eval_loss > eval_loss:
            #  best_eval_loss = eval_loss
            #  better_eval=True

            '''
            if args.do_train and (args.local_rank == -1 or torch.distributed.get_rank() == 0) and better_eval:
               # Save a trained model, configuration and tokenizer
               logger.info("Better eval found")
               log_file.write("Better eval found\n")
               model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

               # If we save using the predefined names, we can load using `from_pretrained`
               output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
               output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

               torch.save(model_to_save.state_dict(), output_model_file)
               torch.save(args, os.path.join(args.output_dir, "training_args.bin"))
               logger.info("Saving model checkpoint to %s", args.output_dir)

               torch.save(optimizer.state_dict(), os.path.join(args.output_dir, "optimizer.pt"))

               model_to_save.config.to_json_file(output_config_file)
               tokenizer.save_vocabulary(args.output_dir)
            '''

                    
        tracemalloc.stop() 

        #for _ in trange(int(args.num_train_epochs), desc="Epoch"):
    if args.do_train: # and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Save a trained model, configuration and tokenizer

        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)
        output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(args.output_dir)

        
        # Load a trained model and vocabulary that you have fine-tuned
        print("Loading a trained model and vocabulary that you have fine tuned..")
        model = model_class.from_pretrained(args.output_dir)

        print(model)

        tokenizer = tokenizer_class.from_pretrained(args.output_dir, do_lower_case=args.do_lower_case) #, do_basic_tokenize=False)
    else:

        print("Loading a trained model from args.bert_model", args.bert_model)
        model = model_class.from_pretrained(args.bert_model) #(,do_lower_case=args.do_lower_case)
        tokenizer = tokenizer_class.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)#, do_basic_tokenize=False)

    # Add special tokens
    special_tokens_dict = {'additional_special_tokens': ['[START]','[END]']}

    #special_tokens_dict = {'additional_special_tokens': ['[SID]', '[SPT]', '[tok]', '[arg1-1]', '[pred-1]', '[arg2-1]', '[arg3-1]', '[no_rel]', '[arg1-2]', '[pred-2]', '[arg1-3]', '[pred-3]', '[arg2-2]', '[arg2-3]', '[arg1-4]', '[pred-4]', '[arg2-4]', '[arg5-1]', '[arg4-1]', '[arg3-2]', '[arg1-6]', '[arg1-5]', '[pred-5]', '[pred-6]', '[arg1-7]', '[pred-7]', '[arg3-3]', '[arg3-4]', '[arg2-5]', '[arg2-6]', '[arg2-7]', '[arg2-12]', '[pred-8]', '[pred-12]', '[arg1-8]', '[arg2-9]', '[arg2-10]', '[arg1-12]', '[pred-9]', '[arg1-9]', '[arg2-11]', '[pred-10]', '[pred-11]', '[arg1-10]', '[arg1-11]', '[arg3-5]', '[arg3-6]', '[arg3-7]', '[arg2-8]', '[arg3-9]', '[arg3-8]', '[arg4-2]', '[arg4-3]', '[arg1-13]', '[arg1-14]', '[pred-13]', '[pred-14]', '[arg2-13]', '[arg2-14]', '[arg3-10]', '[arg3-11]', '[arg3-12]', '[arg3-13]', '[arg4-8]', '[arg5-3]', '[arg6-3]', '[arg5-2]', '[arg4-4]', '[arg4-17]', '[arg4-18]', '[arg1-18]', '[arg1-25]', '[arg2-20]', '[arg1-17]', '[arg2-19]', '[pred-20]', '[arg1-24]', '[pred-19]', '[arg1-19]', '[arg1-20]', '[arg1-21]', '[arg1-23]', '[pred-21]', '[arg2-17]', '[arg2-18]', '[arg2-21]', '[pred-17]', '[pred-18]', '[arg3-17]', '[arg3-18]', '[arg1-15]', '[arg1-16]', '[pred-15]', '[pred-16]', '[arg2-15]', '[arg2-16]', '[arg4-22]', '[arg4-23]', '[arg4-24]', '[arg4-25]', '[arg1-22]', '[pred-22]', '[pred-23]', '[pred-24]', '[pred-25]', '[arg2-22]', '[arg2-23]', '[arg2-24]', '[arg2-25]', '[arg3-14]', '[arg3-15]', '[arg3-16]', '[arg3-22]', '[arg3-23]', '[arg3-24]', '[arg3-25]', '[arg6-1]', '[arg4-5]', '[arg4-6]', '[arg4-7]']}
    #tokenizer.add_special_tokens(special_tokens_dict)

    #print(model.bert.embeddings.word_embeddings.weight.size())
    #input("model word embeddings size")
    #model.resize_token_embeddings(len(tokenizer))
    #model.config.vocab_size = len(tokenizer) #.resize_token_embeddings(len(tokenizer))
    #print("total vocab size:", len(tokenizer), model.config.vocab_size)

    model.to(device)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):

      if args.qlm_model:
        qlm_model = BertForQuestionLM.from_pretrained(args.qlm_model,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))
        #qlm_model = BertForQuestionGeneration.from_pretrained(args.qlm_model,
        #        cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(args.local_rank)))
        qlm_model.cuda()

      else:
        qlm_model=None


      if args.testset:
        train_examples, version = read_squad_examples(
            input_file=args.train_file, is_training=True, version_2_with_negative=args.version_2_with_negative, args=args)
        #test_examples = train_examples[:11877]
        #evaluate_by_generate(model, tokenizer, args,device,no_of_datapoints=-1,predict_dir=args.predict_dir,train=True)
        evaluate(args, model, tokenizer, prefix="do_predict testset")

      else:

        #print("encoder_model:", encoder_model)
        print("args . do predict")
        #evaluate_by_generate(model, tokenizer, args,device,no_of_datapoints=-1,predict_dir=args.predict_dir,train=True)
        evaluate(args, model, tokenizer, prefix="do_predict devset", encoder_model=encoder_model)
        #evaluate_qa(model, tokenizer, args,device)
        #eval_loss, eval_loss_per_file = evaluate_qa(model, tokenizer, args, device, step=0, ppl_only=True, predict_dir=args.predict_dir)
    

      '''
        for i, example_index in enumerate(example_indices):
                start_logits = batch_start_logits[i].detach().cpu().tolist()
                end_logits = batch_end_logits[i].detach().cpu().tolist()
                eval_feature = eval_features[example_index.item()]
                unique_id = int(eval_feature.unique_id)
                all_results.append(RawResult(unique_id=unique_id,
                                             start_logits=start_logits,
                                             end_logits=end_logits))

        output_prediction_file = os.path.join(args.output_dir, "predictions.json")
        output_nbest_file = os.path.join(args.output_dir, "nbest_predictions.json")
        output_null_log_odds_file = os.path.join(args.output_dir, "null_odds.json")

        write_predictions(eval_examples, eval_features, all_results,
                          args.n_best_size, args.max_answer_length,
                          args.do_lower_case, output_prediction_file,
                          output_nbest_file, output_null_log_odds_file, args.verbose_logging,
                          args.version_2_with_negative, args.null_score_diff_threshold)

      '''

if __name__ == "__main__":
    main()

