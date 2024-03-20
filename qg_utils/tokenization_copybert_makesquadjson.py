from typing import Dict, Optional, List
import torch.nn as nn
import os
from torch.utils.data import DataLoader
#from transformers import LayoutLMv3TokenizerFast, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor, LayoutLMv3Processor
from transformers import LayoutLMv3FeatureExtractor
from src.tokenization_layoutlmv3_fast import LayoutLMv3TokenizerFast 
from dataclasses import dataclass
from PIL import Image
import torch
import cv2
from src.utils import bbox_string
import json

def get_subword_start_end(word_start, word_end, subword_idx2word_idx, sequence_ids):
    ## find the separator between the questions and the text
    start_of_context = -1
    for i in range(len(sequence_ids)):
        if sequence_ids[i] == 1:
            start_of_context = i
            break
    num_question_tokens = start_of_context
    assert start_of_context != -1, "Could not find the start of the context"
    subword_start = -1
    subword_end = -1
    for i in range(start_of_context, len(subword_idx2word_idx)):
        if word_start == subword_idx2word_idx[i] and subword_start == -1:
            subword_start = i
        if word_end == subword_idx2word_idx[i]:
            subword_end = i
    return subword_start, subword_end, num_question_tokens

"""
handling the out of maximum length reference:
https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
"""


def make_squad_json():
    data=dict()
    data["data"]=[ {"title":"docvqa", "paragraphs":[] }]
    data["version"]=0.1
    return data

def tokenize_docvqa(examples):

    #tokenizer: LayoutLMv3TokenizerFast,
    #img_dir: Dict[str, str],
    #add_metadata: bool = True,
    #use_msr_ocr: bool = False,
    #use_generation: bool = False,
    #doc_stride: int = 128,
    #ignore_unmatched_answer_span_during_train: bool = True): 

    ## doc stride for sliding window, if 0, means no sliding window.

    #features = {"input_ids": [], "image":[], "bbox":[], "start_positions": [], "end_positions":[],  "metadata": []}
    squad_data=make_squad_json()

    #for idx, (question, image_path, words, layout) in enumerate(zip(examples["question"], examples["image"], examples["words"], examples["layout"])):
    for idx, ex in enumerate(examples): 
        question=ex["question"]
        words=ex["words"]
        answer_list=ex["processed_answers"]

        #answer_list = examples["processed_answers"][idx] if "processed_answers" in examples else []
        #original_answer = examples["original_answer"][idx] if "original_answer" in examples else []

        #print(question, words, layout, answer_list, original_answer)
        context=" ".join(words)
        answers=[]
        for ans in answer_list:
          #ext_ans= ans["extracted_answer"]
          gold_ans= ans["gold_answer"]
          ext_ans = " ".join(words[ans["start_word_position"]:ans["end_word_position"]+1])

          if ext_ans == "":
              continue
              print("wait")
              print(ans["start_word_position"])
              print(ans["end_word_position"])
              print(len(words))
              input("")
          if ans["start_word_position"] == 0:
            start_char_position=len(" ".join(words[:ans["start_word_position"]]) ) 
          else:
            start_char_position=len(" ".join(words[:ans["start_word_position"]]) ) + 1

          if start_char_position >= len(context):
              print("char out of context:")
              print(ans["start_word_position"])
              print(ans["end_word_position"])
              print(len(words))

              input("")
          ans_len=len(ext_ans)
          #print("ans check:", context[start_char_position:start_char_position+ans_len], "//", gold_ans, "//", ext_ans)

          print(context[start_char_position:start_char_position+ans_len],ext_ans)
          assert context[start_char_position:start_char_position+ans_len] == ext_ans

          answers.append({"text":ext_ans, 
              "answer_start":start_char_position})

          qas=[{"question":question, "answers":answers, "id":str(idx)}]
        squad_data["data"][0]["paragraphs"].append({"context":context, 
                                                     "qas": qas
                                                     } )

    return squad_data


@dataclass
class DocVQACollator:
    tokenizer: LayoutLMv3TokenizerFast
    feature_extractor: LayoutLMv3FeatureExtractor
    pretrained_model_name: str
    padding: bool = True
    model: Optional[nn.Module] = None

    def __call__(self, batch: List):

        labels = [feature["labels"] for feature in batch] if "labels" in batch[0].keys() else None
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            for feature in batch:
                remainder = [self.tokenizer.pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["label_ids"] = feature["labels"] + remainder
                # print(feature["labels"])
                feature.pop("labels")

        for feature in batch:
            #print(feature["image"])
            image = Image.open(feature["image"]).convert("RGB")
            vis_features = self.feature_extractor(images=image, return_tensors='np')["pixel_values"][0]
            if "layoutlmv2" in self.pretrained_model_name:
                feature["image"] = vis_features.tolist()
            else:
                feature['pixel_values'] = vis_features.tolist()
                if 'image' in feature:
                    feature.pop('image')

        batch = self.tokenizer.pad(
            batch,
            padding=self.padding,
            pad_to_multiple_of=None,
            return_tensors="pt",
            return_attention_mask=True
        )

         # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            #print("preparing decoder input ids from labels..")
            batch.pop("start_positions")
            batch.pop("end_positions")
            #print("batch keys:", batch.keys())
            if "label_ids" in batch:
                decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["label_ids"])
                batch["decoder_input_ids"] = decoder_input_ids
                #print("batch[decoder_input_ids]:", batch["decoder_input_ids"])
                ## for validation, we don't have "labels
        if "label_ids" in batch:
            ## layoutlmv3 tokenizer issue, they don't allow "labels" key as a list..so we did a small trick
            batch["labels"] = batch.pop("label_ids")
        return batch




if __name__ == '__main__':
    from datasets import load_from_disk, DatasetDict
    from tqdm import tqdm
    #dataset_file = './data/docvqg_cached_all_lowercase_True_msr_False_extraction_v3' 
    dataset_file = 'data/docvqg_cached_copybert'
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained('bert-base-uncased')
    dataset = load_from_disk(dataset_file)

    # dataset = DatasetDict({"train": dataset["train"], "val": dataset['val']})
    # image_dir = {"train": "data/docvqa/train", "val": "data/docvqa/val", "test": "data/docvqa/test"}
    # use_msr = "msr_True" in dataset_file
    image_dir="./data/docvqg/train/documents"
    use_msr=False


    squad_data_train = tokenize_docvqa(dataset["train"])
    print("train:", len(squad_data_train["data"][0]["paragraphs"]))

    with open("docvqg.train.json","w") as f:
        json.dump(squad_data_train, f, indent=4)

    squad_data_val = tokenize_docvqa(dataset["val"])
    print("val:", len(squad_data_val["data"][0]["paragraphs"]))
    with open("docvqg.val.json","w") as f:
        json.dump(squad_data_val, f, indent=4)

    squad_data_test = tokenize_docvqa(dataset["test"])
    print("test:", len(squad_data_test["data"][0]["paragraphs"]))
    with open("docvqg.test.json","w") as f:
        json.dump(squad_data_test, f, indent=4)

    '''
    '''
