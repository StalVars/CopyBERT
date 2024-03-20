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
from tokenMap import AlignTokenizations as Align 

import argparse
import collections
import json
import logging
import math
import os
import random
import sys
from io import open
import sys
import re

sys.path.append("./pytorch-pretrained-BERT_older/examples")

import numpy as np
import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import torch.nn.functional as F
sys.path.append('./')


from pytorch_pretrained_bert_local.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert_local.modeling import BertForQuestionGeneration, BertConfig
from pytorch_pretrained_bert_local.modeling import BertForQuestionAnswering
from pytorch_pretrained_bert_local.modeling import BertForQuestionLM

from pytorch_pretrained_bert_local.optimization import BertAdam, WarmupLinearSchedule
from pytorch_pretrained_bert_local.tokenization import (BasicTokenizer,
                                                  BertTokenizer,
                                                  whitespace_tokenize)
from pytorch_pretrained_bert_local.beam_search import BeamSearch
#from pytorch_pretrained_bert_local.beam import Beam, GNMTGlobalScorer
from nltk.translate.bleu_score import corpus_bleu

import datetime


now = datetime.datetime.now()
filetag = str(now.hour)+"."+str(now.minute)+"."+str(now.second)

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

logger = logging.getLogger(__name__)
#logging.basicConfig(filename='logs/logfile.txt', filemode='w')


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


def read_squad_examples(input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""
    with open(input_file, "r", encoding='utf-8') as reader:
        input_data = json.load(reader)["data"]

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    # If there exists a dumped examples file for dev and test separately, load them and ignore further processing:

    
    examples = []
    for entry in input_data:
        for paragraph in entry["paragraphs"]:
            paragraph_text = paragraph["context"]
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

            for qa in paragraph["qas"]:
                qas_id = qa["id"]
                question_text = qa["question"]
                start_position = None
                end_position = None
                orig_answer_text = None
                is_impossible = False
                if True: #is_training:
                    if version_2_with_negative:
                        is_impossible = qa["is_impossible"]

                    '''
                    if (len(qa["answers"]) != 1) and (not is_impossible):
                        raise ValueError(
                            "For training, each question should have exactly 1 answer.")
                    '''
                    if not is_impossible:
                        answer = qa["answers"][0]
                        orig_answer_text = answer["text"]
                        answer_offset = answer["answer_start"]
                        answer_length = len(orig_answer_text)
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
                    is_impossible=is_impossible)
                examples.append(example)
    #Shuffle 
    if not is_training:
        if not os.path.exists("dev_examples.pt"): 
          random.shuffle(examples)
          dev_length=len(examples)//2
          dev_examples=examples[:dev_length]
          test_examples=examples[dev_length:]
          with open("dev_examples.pt","wb") as writer:
              pickle.dump((dev_examples,test_examples),writer)
        else:
          with open("dev_examples.pt", "rb") as reader:
                examples = pickle.load(reader)
          dev_examples, test_examples = examples
        print("Dev dataset length:",len(dev_examples))
        print("Test dataset length:",len(test_examples))
        log_file.write("Dev dataset length:"+str(len(dev_examples))+"\n")
        log_file.write("Test dataset length:"+str(len(test_examples))+"\n")
        return dev_examples, test_examples

    # Take half and save 2 files dev and test
    # return dev

    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,no_question=False,is_generate=False):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000

    features = []
    for (example_index, example) in enumerate(examples):
        query_tokens = tokenizer.tokenize(example.question_text)
        #print(query_tokens)
        #input("query tokens")

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

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

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        if is_generate:
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
        else:
         while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []
            tokens.append("[CLS]")
            segment_ids.append(0)

            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(0)
            tokens.append("[MASK]")
            segment_ids.append(1) 

            # Question to come after the context
            if not no_question:
             for token in query_tokens:
                tokens.append(token)
                segment_ids.append(1)
             tokens.append("[SEP]")
             segment_ids.append(1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)
            if no_question:
             for token in query_tokens:
                tokens.append(token)
             tokens.append("[SEP]")

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

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
                if out_of_span:
                    start_position = 0
                    end_position = 0
                else:
                    #doc_offset = len(query_tokens) + 2
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

            features.append(
                InputFeatures(
                    unique_id=unique_id,
                    example_index=example_index,
                    doc_span_index=doc_span_index,
                    tokens=tokens,
                    token_to_orig_map=token_to_orig_map,
                    token_is_max_context=token_is_max_context,
                    input_ids=input_ids,
                    input_mask=input_mask,
                    segment_ids=segment_ids,
                    start_position=start_position,
                    end_position=end_position,
                    is_impossible=example.is_impossible))
            unique_id += 1

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
            return_attention=False):

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

        # (0) pt 2, prep the beam object
        beam = BeamSearch(
            beam_size,
            tokenizer,
            batch_size=batch_size,
            n_best=n_best,
            min_length=min_length,
            max_length=max_length,
            block_ngram_repeat=True)

        model.eval()

        #print(start_positions, end_positions)
        #input("start positions and end positions")
        #print([" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) ])) for b in range(input_ids.size(0)) ][0])
        #input("input_ids")

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


def evaluate_by_loss(model, tokenizer, args, device,no_of_datapoints=-1,train=False):

        if train:
            data_file = args.train_file
        else:
            data_file = args.predict_file
        eval_dev_examples, eval_test_examples = read_squad_examples(
            input_file=data_file, is_training=False, version_2_with_negative=args.version_2_with_negative)
        if no_of_datapoints != -1:
            eval_dev_examples = eval_dev_examples[:no_of_datapoints]

        eval_features = convert_examples_to_features(
            examples=eval_dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            doc_stride=args.doc_stride,
            max_query_length=args.max_query_length,
            is_training=True) #,no_question=True)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_dev_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_start_positions = torch.tensor([f.start_position for f in eval_features], dtype=torch.long)
        all_end_positions = torch.tensor([f.end_position for f in eval_features], dtype=torch.long)
        #all_tokens = [f.tokens for f in eval_features]
        all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_start_positions, all_end_positions, all_example_index)
        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        logger.info("Start evaluating")
        eval_loss = 0
        total_tokens = 0

        no_of_examples=0
        with torch.no_grad():
          for input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
            if no_of_examples % 1000 == 0:
                logger.info("Processing example: %d" % (no_of_examples))

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

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
            loss = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=args.copymethod)
            eval_loss += loss 
            #total_tokens += (segment_ids == 1).sum()
            no_of_examples += args.predict_batch_size

        return eval_loss #/total_tokens


def print_and_file(*text):
    print(*text)
    #print(text)
    #predlog_file.write(text[0]+"\n")
    #predlog_file.flush()
    #input("ok?")

def print_beam(tokenizer, preds, segments,scores=None,start_indices=None,end_indices=None,secondary_scores=None):
    return_pred=""
    best_score=0

    preds_in_text=[]
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
                preds_in_text.append(" ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])))
              else:
                print_and_file("%d %f %s" % (j,scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )
            else:

             if start_indices is not None:
               print_and_file("%d %f %f %s | Answer %s" % (j,scores[j],secondary_scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])), " ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(start_indices[j].item(), end_indices[j].item()+1)  ]))  ))
               preds_in_text.append(" ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ])))
             else:
               print_and_file("%d %f %f %s" % (j,scores[j],secondary_scores[j]," ".join(tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1 ]))) )

    #return return_pred, best_score
    #return preds, scores
    return preds_in_text , scores


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


def generate_per_text(model, tokenizer, args, device, input_text, start_position, end_position, qlm_model=None):
        #input_mask

        with torch.no_grad():

           #Get following tensors from input
           #input_ids
           #segment_ids
           #start_positions
           #end_positions

            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            #Batch size = 1, remove the pads
            input_length = (input_ids!=0).sum(-1).item()
            input_ids = input_ids[:,:input_length]
            input_mask = input_mask[:,:input_length]
            segment_ids = segment_ids[:,:input_length]
            #start_positions = start_positions[:,:input_length]
            #end_positions = end_positions[:,:input_length]

            #ngram_match, denom = generate_questions(model, tokenizer, input_ids, segment_ids,input_mask, start_positions, end_positions, references,args)

            inputs = [ input_ids, segment_ids,input_mask, start_positions, end_positions, references ]

            #predlog_file.write("---"+" ".join([str(i) for i in example_indices ])+"\n")
            #gen_tup  = sample_sequence(model, tokenizer, inputs, 20, args)

            gen_tup = translate_batch(model, tokenizer, inputs, 20, args,qamodel, beam_size=args.beam_size, qlm_model=qlm_model)
            preds = gen_tup['predictions']
            segments = gen_tup['segments']
            scores = gen_tup['scores']

            new_start_indices=None
            new_end_indices=None
            prev_scores = None


            if args.rerank:
              prev_scores = scores[0]
              preds,segments,scores,new_start_indices, new_end_indices = rerank(preds, segments, scores,input_ids,input_mask,qamodel,start_positions,end_positions,top=10)
            else:
              new_start_indices = torch.zeros(len(preds[0])).long().cuda() + start_positions - 1 # -1 because [CLS] token is removed in the preds
              new_end_indices = torch.zeros(len(preds[0])).long().cuda() + end_positions - 1 # -1 because [CLS] token is removed in the preds

            print_and_file("##### Context ###")
            print_and_file([" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0])) for b in range(input_ids.size(0)) ][0])
            print_and_file("##### N-Best Predictions ###")
            #To print beam
            best_pred_text, best_score = print_beam(tokenizer, preds[0], segments[0],scores=scores[0],start_indices = new_start_indices,end_indices=new_end_indices,secondary_scores=prev_scores)
            best_pred_text = re.sub("( \[SEP\])+","",best_pred_text)
            best_pred_text = re.sub("( \[PAD\])+","",best_pred_text)
            best_pred_text = re.sub("( \?)+"," ?",best_pred_text)
            print_and_file("#Predicted: %s " %(best_pred_text))

            ngram_match = 0
            denom = 1

def evaluate_by_generate(model, tokenizer, args, device,no_of_datapoints=-1,train=False,qlm_model=None):

        if train:
            data_file = args.train_file
        else:
            data_file = args.predict_file

        eval_dev_examples,eval_test_examples = read_squad_examples(
            input_file=data_file, is_training=False, version_2_with_negative=args.version_2_with_negative)

        if args.genset == "dev":
           eval_test_examples = eval_dev_examples

        if no_of_datapoints != -1:
            eval_test_examples = eval_test_examples[:no_of_datapoints]
        qamodel = BertForQuestionAnswering.from_pretrained("outputdir_squad2.0_bert-base-cased")
        qamodel = qamodel.cuda()
        qamodel.eval()

        eval_features = convert_examples_to_features(
            examples=eval_test_examples,
            #examples=eval_dev_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            #doc_stride=args.doc_stride,
            doc_stride=1390,
            max_query_length=args.max_query_length,
            is_training=True,no_question=True,is_generate=True)

        logger.info("***** Running predictions *****")
        logger.info("  Num orig examples = %d", len(eval_test_examples))
        logger.info("  Num split examples = %d", len(eval_features))
        logger.info("  Batch size = %d", args.predict_batch_size)

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
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)

        model.eval()
        all_results = []
        logger.info("Start evaluating")
        total_ngram_matches = 0
        total_denom = 0
        with torch.no_grad():
         for input_ids, input_mask, segment_ids, start_positions, end_positions, example_indices in tqdm(eval_dataloader, desc="Evaluating", disable=args.local_rank not in [-1, 0]):
            if len(all_results) % 1000 == 0:
                logger.info("Processing example: %d" % (len(all_results)))

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            #Batch size = 1, remove the pads
            input_length = (input_ids!=0).sum(-1).item() 
            input_ids = input_ids[:,:input_length]
            input_mask = input_mask[:,:input_length]
            segment_ids = segment_ids[:,:input_length]
            #start_positions = start_positions[:,:input_length]
            #end_positions = end_positions[:,:input_length]

            references = [ re.sub("[     ]*\[SEP\]","",re.sub(".*\[MASK\]",""," ".join(all_tokens[i]))) for i in example_indices ]
            #references = [ re.sub(".*\[MASK\]",""," ".join(all_tokens[i])) for i in example_indices ]
            #ngram_match, denom = generate_questions(model, tokenizer, input_ids, segment_ids,input_mask, start_positions, end_positions, references,args)
            
            inputs = [ input_ids, segment_ids,input_mask, start_positions, end_positions, references ]

            #predlog_file.write("---"+" ".join([str(i) for i in example_indices ])+"\n")
            #gen_tup  = sample_sequence(model, tokenizer, inputs, 20, args)
            gen_tup = translate_batch(model, tokenizer, inputs, 20, args,qamodel, beam_size=args.beam_size, qlm_model=qlm_model)
            preds = gen_tup['predictions']
            segments = gen_tup['segments']
            scores = gen_tup['scores']

            new_start_indices=None
            new_end_indices=None
            prev_scores = None


            if args.rerank:
              prev_scores = scores[0]
              preds,segments,scores,new_start_indices, new_end_indices = rerank(preds, segments, scores,input_ids,input_mask,qamodel,start_positions,end_positions,top=10) 
            else:
              new_start_indices = torch.zeros(len(preds[0])).long().cuda() + start_positions - 1 # -1 because [CLS] token is removed in the preds
              new_end_indices = torch.zeros(len(preds[0])).long().cuda() + end_positions - 1 # -1 because [CLS] token is removed in the preds
            

            print_and_file("##### Context ###")
            print_and_file([" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0])) for b in range(input_ids.size(0)) ][0])
            print_and_file("##### N-Best Predictions ###")
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
        logger.info("BLEU Score on DEV set: %f " % (total_ngram_matches / total_denom) )

class BertQGEN(object):
   """
   A single training/test example for the Squad dataset.
   For examples without an answer, the start and end position are -1.
   """

   def __init__(self):

        bert_model="pytorch-pretrained-BERT_older/qgen_model/"
        local_rank=-1
        self.model = BertForQuestionGeneration.from_pretrained(bert_model,
                cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed_{}'.format(local_rank)))
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=False)
        self.qamodel = BertForQuestionAnswering.from_pretrained("pytorch-pretrained-BERT_older/outputdir_squad2.0_bert-base-cased")
        self.align = Align()
        self.qlm_model = None
        self.model.cuda()
        self.qamodel.cuda()

   def __str__(self):
        return self.__repr__()

   def qa_scores(self,preds, segments, scores, input_ids, input_mask,qamodel,start_positions,end_positions,top=10):
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
    
    '''
    val, ind = torch.sort(new_scores,dim=0,descending=True)
    ind = ind.squeeze(-1)
    new_preds = new_preds.index_select(0,ind) 
    new_scores = new_scores.index_select(0,ind) 
    new_segments = new_segments.index_select(0,ind) 
    new_start_indices = new_start_indices.index_select(0,ind) 
    new_end_indices = new_end_indices.index_select(0,ind) 
    '''
    

    #return [preds], [segments], [scores]
    return [new_preds], [1-new_segments], [new_scores],new_start_indices, new_end_indices

   def rerank(self,preds, segments, scores, input_ids, input_mask,qamodel,start_positions,end_positions,top=10):
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
    
   def translate_batch(
            self,
            model,
            tokenizer,
            batch,
            max_length,
            qamodel,
            qlm_model=None,
            min_length=0,
            ratio=0.,
            n_best=1,
            beam_size=10,
            return_attention=False):

        # TODO: support these blacklisted features.

        # (0) Prep the components of the search.
        #beam_size = 5 #20
        batch_size = 1
        n_best = 10 #args.n_best_size = 20,normally

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

        # (0) pt 2, prep the beam object
        beam = BeamSearch(
            beam_size,
            tokenizer,
            batch_size=batch_size,
            n_best=n_best,
            min_length=min_length,
            max_length=max_length,
            block_ngram_repeat=True)

        model.eval()

        #print(start_positions, end_positions)
        #input("start positions and end positions")
        #print([" ".join(tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) ])) for b in range(input_ids.size(0)) ][0])
        #input("input_ids")

        log_probs = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=1,evaluate=True)

        if qlm_model is not None:
            lm_logits = qlm_model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=1,evaluate=True)
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

            log_probs = model(input_ids, segment_ids, input_mask, start_positions, end_positions,copymethod=1,evaluate=True)
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
        results["scores"] = [[ beam.scores[0][i]/beam.segments[0][i].float().sum() for i in range(len(beam.scores[0])) ]]
        results["predictions"] = beam.predictions
        results["segments"] = beam.segments
        #results["attention"] = beam.attention
        return results


   def generate_per_text(self, input_text, start_position, end_position,rerank=False,get_qa_scores=False ):

        #input_mask
        device=0
        preds_in_text=[]
        with torch.no_grad():

            #Get following tensors from input
            toks = input_text.split(" ")
            tokenized_input = self.tokenizer.tokenize(input_text)

            print("length of input", len(toks))
            #input("tokenized input")

            map_src_toks2berttoks=self.align.map(toks,tokenized_input)
            print("qgen for answer phrase:",start_position,end_position)
            print("length of map_src_toks2berttoks=",len(map_src_toks2berttoks))
            bert_start = map_src_toks2berttoks[start_position][0] + 1 #+1 for [CLS]
            bert_end = map_src_toks2berttoks[end_position][0] + 1 

            #print(bert_start, bert_end)
            #print(tokenized_input[bert_start:bert_end+1])
            #input("answer phrase")
            tokenized_input = ["[CLS]"]+tokenized_input+["[MASK]"]

            #input_ids
            input_ids = torch.tensor([self.tokenizer.convert_tokens_to_ids(tokenized_input)])
            print(input_ids)
            print(tokenized_input)
            print("****************** --- DEBUG --- ******************")

            #segment_ids
            segment_ids = torch.zeros_like(input_ids)
            segment_ids[:,-1] = 1

            input_mask = torch.ones_like(input_ids)

            #start_positions
            start_positions = torch.tensor([bert_start])
            #end_positions
            end_positions = torch.tensor([bert_end])

            input_ids = input_ids.to(device)
            input_mask = input_mask.to(device)
            segment_ids = segment_ids.to(device)
            start_positions = start_positions.to(device)
            end_positions = end_positions.to(device)

            #Batch size = 1, remove the pads
            '''
            input_length = (input_ids!=0).sum(-1).item()
            input_ids = input_ids[:,:input_length]
            input_mask = input_mask[:,:input_length]
            segment_ids = segment_ids[:,:input_length]
            #start_positions = start_positions[:,:input_length]
            #end_positions = end_positions[:,:input_length]
            '''

            #ngram_match, denom = generate_questions(model, tokenizer, input_ids, segment_ids,input_mask, start_positions, end_positions, references,args)
            references=[""]

            inputs = [ input_ids, segment_ids,input_mask, start_positions, end_positions, references ]

            #predlog_file.write("---"+" ".join([str(i) for i in example_indices ])+"\n")
            #gen_tup  = sample_sequence(self.model, self.tokenizer, inputs, 20, args)

            gen_tup = self.translate_batch(self.model, self.tokenizer, inputs, 20, self.qamodel, beam_size=10, qlm_model=self.qlm_model)

            preds = gen_tup['predictions']
            segments = gen_tup['segments']
            scores = gen_tup['scores']
            #sort scores
            sorted_indices = [i[0] for i in sorted(enumerate(scores[0]), key=lambda x:x[1].item(),reverse=True)]
            preds = [[ preds[0][ind] for ind in sorted_indices ]]
            scores = [[ scores[0][ind] for ind in sorted_indices ]]
            segments = [[ segments[0][ind] for ind in sorted_indices ]]
            '''
            '''
            new_start_indices=None
            new_end_indices=None
            prev_scores = None


            if rerank:
              prev_scores = scores[0]
              preds,segments,scores,new_start_indices, new_end_indices = self.rerank(preds, segments, scores,input_ids,input_mask,self.qamodel,start_positions,end_positions,top=10)
            elif get_qa_scores:
              prev_scores = scores[0]
              preds,segments,scores,new_start_indices, new_end_indices = self.qa_scores(preds, segments, scores,input_ids,input_mask,self.qamodel,start_positions,end_positions,top=10)
            else:
              new_start_indices = torch.zeros(len(preds[0])).long().cuda() + start_positions - 1 # -1 because [CLS] token is removed in the preds
              new_end_indices = torch.zeros(len(preds[0])).long().cuda() + end_positions - 1 # -1 because [CLS] token is removed in the preds

            print_and_file("##### Context ###")
            print_and_file([" ".join(self.tokenizer.convert_ids_to_tokens([input_ids[b][s].item() for s in range(input_ids.size(1)) if segment_ids[b][s] == 0])) for b in range(input_ids.size(0)) ][0])
            print_and_file("##### N-Best Predictions ###")
            #To print beam
            #best_pred_text, best_score = print_beam(self.tokenizer, preds[0], segments[0],scores=scores[0],start_indices = new_start_indices,end_indices=new_end_indices,secondary_scores=prev_scores)
            preds, scores = print_beam(self.tokenizer, preds[0], segments[0],scores=scores[0],start_indices = new_start_indices,end_indices=new_end_indices,secondary_scores=prev_scores)

            '''
            for j in range(len(preds)):
              preds_in_text.append(" ".join(self.tokenizer.convert_ids_to_tokens([preds[j][i].item() for i in range(len(preds[j])) if segments[j][i] == 1  ]))) 
            '''
            #print(preds, scores)
            #input("preds scores")
            best_pred_text = preds[0]
            best_score = preds[0]
            best_pred_text = re.sub("( \[SEP\])+","",best_pred_text)
            best_pred_text = re.sub("( \[PAD\])+","",best_pred_text)
            best_pred_text = re.sub("( \?)+"," ?",best_pred_text)
            best_pred_text = re.sub(" ##","",best_pred_text)
            print_and_file("#Predicted: %s " %(best_pred_text))
            #return best_pred_text
            untok_preds=[re.sub(" \[SEP\]|\[MASK\] ","",re.sub(" ##","",pred)) for pred in preds ]
            untok_preds=[re.sub(" \[PAD\]","",pred) for pred in untok_preds]
            untok_preds=[re.sub("\[CLS\] ","",pred) for pred in untok_preds]
            #if get_qa_scores:
            #  return untok_preds, prev_scores, scores 
            return untok_preds, scores 


