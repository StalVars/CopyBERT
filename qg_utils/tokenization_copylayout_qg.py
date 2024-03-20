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
import re

def get_subword_start_end(word_start, word_end, subword_idx2word_idx, sequence_ids):
    ## find the separator between the questions and the text

    start_of_context = -1
    num_question_tokens = 0
    for i in range(len(sequence_ids)):
        if sequence_ids[i] == 1:
            #start_of_context = i
            num_question_tokens += 1
            break

    start_of_question = 0
    for i in range(len(sequence_ids)):
        if sequence_ids[i] == 1:
            start_of_question = i
            break

    #num_question_tokens = start_of_context
    start_of_context  = 1 # 2nd token for copylayout QG


    assert start_of_context != -1, "Could not find the start of the context"
    #print("#start_of_context", start_of_context, "#q", num_question_tokens)

    subword_start = -1
    subword_end = -1
    #for i in range(start_of_context, len(subword_idx2word_idx)):
    for i in range(start_of_context, start_of_question): #len(subword_idx2word_idx)):
        if word_start == subword_idx2word_idx[i] and subword_start == -1:
            subword_start = i
        if word_end == subword_idx2word_idx[i]:
            subword_end = i
    return subword_start, subword_end, num_question_tokens

"""
handling the out of maximum length reference:
https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
"""

def tokenize_docvqa(examples,
                    tokenizer: LayoutLMv3TokenizerFast,
                    img_dir: Dict[str, str],
                    add_metadata: bool = True,
                    use_msr_ocr: bool = False,
                    use_generation: bool = False,
                    doc_stride: int = 128,
                    ignore_unmatched_answer_span_during_train: bool = True): ## doc stride for sliding window, if 0, means no sliding window.

    features = {"input_ids": [], "image":[], "bbox":[], "start_positions": [], "end_positions":[],  "metadata": []}
    current_split = examples["data_split"][0]
    #print("Use_generation:", use_generation)

    if use_generation:
        features["labels"] = []
    max_bbox=0
    #print("in func tokenize_docvqa..")
    #input("")

    missed_features = 0
    for idx, (question, image_path, words, layout) in enumerate(zip(examples["question"], examples["image"], examples["words"], examples["layout"])):
        current_metadata = {}
        added=False
        #print("-", idx)

        file = os.path.join(img_dir[examples["data_split"][idx]], image_path)
        # img = Image.open(file).convert("RGB")
        answer_list = examples["processed_answers"][idx] if "processed_answers" in examples else []
        original_answer = examples["original_answer"][idx] if "original_answer" in examples else []
        # image_id = f"{examples['ucsf_document_id'][idx]}_{examples['ucsf_document_page_no'][idx]}"

        if len(words) == 0 and current_split == "train":
            continue

        return_overflowing_tokens = doc_stride>0
        ocr_text=" ".join(words)

        if current_split != 'train':
            original_answer = original_answer[0:1]

        question = re.sub("(.*)(<.s>)(.*)",r'\1 \2 \3',question) 
        #print("modified question:", question)

        #Length questions should be removed

        '''
        if len(question.split(" ")) > 100:
            print("Too big of a question")
            continue

        if len(words) < 2:
            print("Too small of an ocr text")
            continue
        '''


        for ans in original_answer:

          #print("question:", question)
          #print("Answer:", ans)

          # Trim down too long answers
          if len(ans.split(" ")) > 100:
              ans=" ".join(ans.split(" ")[:100])

          ans = ans + " " + tokenizer.eos_token + " " + question
          #ans = ans + " " + tokenizer.mask_token  + " " + question

          ans_words = ans.split(" ")
          #dummy_words = ["dummy" for i in range(100)]
          #ans_words += dummy_words
          #print("tokenization.. only first")
          #truncation_string="do_not_truncate"
          #print("####words(text):", ocr_text) 
          #print("####ans(text_pair) comes first:", ans_words)

          #print("###")
          # The following are for reverse order
          #print("####words(text_pair):", ocr_text) 
          #print("####ans(text):", ans_words)
          #truncation_string="only_second"
          #tokenized_res = tokenizer.encode_plus(text=ocr_text, text_pair=ans_words, boxes=layout, add_special_tokens=True,
          
          #print("####words(text_pair)(comes first):", words)
          #print("####ans+q(text):", ans)

          #Truncate words if the list is too long:
          #if len(words) > 512:
          #    words=words[:512]


          truncation_string="only_first" 

          #print("text(ans):", ans)
          #print("text_pairs(words):", words)

          if current_split != 'train':
              max_length = 512 - 10
          else:
              max_length = 512 

          tokenized_res = tokenizer.encode_plus(text=ans, text_pair=words, boxes=layout, add_special_tokens=True,
                                              max_length=512, truncation=truncation_string, 
                                              return_offsets_mapping=True, stride=doc_stride,
                                              return_overflowing_tokens=return_overflowing_tokens)

          '''
          print("input_ids:", 
                  len(tokenizer.decode(tokenized_res["input_ids"]).split(" ")), 
                  type(tokenized_res["input_ids"]), idx) 
          '''

          if len(tokenized_res["input_ids"]) < 10:
              print("Too small, continuing..", idx)
              continue

          # sample_mapping = tokenized_res.pop("overflow_to_sample_mapping")
          offset_mapping = tokenized_res.pop("offset_mapping")

           #use_msr_ocr for norm bbox
           #print("use_msr_ocr", use_msr_ocr)
          #use_msr_ocr = False #True
          #documents/

          if not return_overflowing_tokens:
             offset_mapping = [offset_mapping]

          if use_generation:
             #dummy_boxes = [[[0,0,0,0]] for _ in range(len(original_answer))]
             dummy_boxes = [[0,0,0,0]]

             #answer_ids = tokenizer.batch_encode_plus([[ans] for ans in original_answer], boxes = dummy_boxes,
             #                                         add_special_tokens=True, max_length=100, truncation="longest_first")["input_ids"]

             #print("longest_first")

             q_truncation_string="do_not_truncate"


             question_ids = tokenizer.batch_encode_plus([[question]], boxes = [dummy_boxes],
                                                     add_special_tokens=True, max_length=100, truncation=q_truncation_string)["input_ids"]
             question_ids = question_ids[0]

             #print("question ids:", len(question_ids), question_ids)

          else:
            question_ids = [[0] for _ in range(len(original_answer))]

          if not use_msr_ocr:
            img = cv2.imread(file)
            height, width = img.shape[:2]


          span_found_atleast_once=False

          for stride_idx, offsets in enumerate(offset_mapping):

            input_ids = tokenized_res["input_ids"][stride_idx] if return_overflowing_tokens else tokenized_res["input_ids"]
            bboxes = tokenized_res["bbox"][stride_idx] if return_overflowing_tokens else tokenized_res["bbox"]
            subword_idx2word_idx = tokenized_res.encodings[stride_idx].word_ids
            sequence_ids = tokenized_res.encodings[stride_idx].sequence_ids

            #if current_split != 'train' and span_found_atleast_once:
            if current_split != 'train' and added:
                break

            if current_split == "train" or True:
                # for training, we treat instances with multiple answers as multiple instances
                #for answer, label_ids in zip(answer_list, answer_ids):

                answer=answer_list[0]

                #label_ids=answer_ids
                #print(stride_idx)

                if True:
                    #print("Use_generation:", use_generation)
                    #if True: #not use_generation:

                    if True: #not use_generation:
                        #print("start pos ")
                        if answer["start_word_position"] == -1:
                            #print("start_word_position == -1")
                            subword_start = 0 ## just use the CLS
                            subword_end = 0
                            num_question_tokens = 0
                            if ignore_unmatched_answer_span_during_train and current_split == 'train':
                                continue
                        else:
                            #print("Get answer start end positions of tokens")

                            subword_start, subword_end, num_question_tokens = get_subword_start_end(answer["start_word_position"], answer["end_word_position"], subword_idx2word_idx, sequence_ids)

                            #print(answer["start_word_position"], answer["end_word_position"], subword_start, subword_end, subword_idx2word_idx) 
                            #print(answer["start_word_position"], answer["end_word_position"], subword_start, subword_end) #, sequence_ids)


                            if subword_start == -1:
                                subword_start = 0  ## just use the CLS
                                subword_end = 0

                                #Ignore unmatched span during testing used when  'consider_ans_pos'

                                #if current_split != 'train' and not stride_idx != len(offset_mapping)-1:
                                #   continue


                                if ignore_unmatched_answer_span_during_train and current_split == 'train':
                                    continue

                            else:
                                span_found_atleast_once=True

                            if subword_end == -1:
                                ## that means the end position is out of maximum boundary
                                ## last is </s>, second last
                                subword_end = 511 - 1
                    else:
                        #print("label ids:" , label_ids)
                        #input("-")
                        #features["labels"].append(question_ids)

                        subword_start = -1  ## useless as in generation
                        subword_end = -1
                        num_question_tokens = 0

                    features["image"].append(file)
                    features["input_ids"].append(input_ids)
                    features["labels"].append(question_ids)

                    boxes_norms = []
                    for box in bboxes:

                        box_norm = box if use_msr_ocr else bbox_string([box[0], box[1], box[2], box[3]], width, height)

                        if max(box_norm) > 1000:
                            print(box_norm, ">1k", 
                                    "use_msr_ocr:", use_msr_ocr)
                            #input("")

                        boxes_norms.append(box_norm)

                    features["bbox"].append(boxes_norms)
                    features["start_positions"].append(subword_start)
                    features["end_positions"].append(subword_end)
                    current_metadata["original_answer"] = original_answer
                    current_metadata["question"] = question
                    current_metadata["num_question_tokens"] = num_question_tokens ## only used in testing.
                    current_metadata["words"] = words
                    current_metadata["subword_idx2word_idx"] = subword_idx2word_idx
                    current_metadata["questionId"] = examples["questionId"][idx]
                    current_metadata["data_split"] = examples["data_split"][idx]
                    features["metadata"].append(current_metadata)
                    #print( "length of labels/start_positions:", 
                    #        len(features["labels"]), 
                    #        len(features["start_positions"]) )
                    assert len(features["labels"]) == len(features["start_positions"])

                    if not add_metadata:
                        features.pop("metadata")
            else:
                # for validation and test, we treat instances with multiple answers as one instance
                # we just use the first one, and put all the others in the "metadata" field
                print("Validation/Test")
                subword_start, subword_end = -1, -1
                #print("Use_generation:", use_generation)
                if use_generation:
                  features["labels"].append(question_ids)
                for i in range(len(sequence_ids)):
                    if sequence_ids[i] == 1:
                        num_question_tokens = i
                        break
                features["image"].append(file)
                features["input_ids"].append(input_ids)
                boxes_norms = []
                for box in bboxes:
                    box_norm = box if use_msr_ocr else bbox_string([box[0], box[1], box[2], box[3]], width, height)
                    boxes_norms.append(box_norm)
                features["bbox"].append(boxes_norms)
                features["start_positions"].append(subword_start)
                features["end_positions"].append(subword_end)
                current_metadata["original_answer"] = original_answer
                current_metadata["question"] = question
                current_metadata["num_question_tokens"] = num_question_tokens
                current_metadata["words"] = words
                current_metadata["subword_idx2word_idx"] = subword_idx2word_idx
                current_metadata["questionId"] = examples["questionId"][idx]
                current_metadata["data_split"] = examples["data_split"][idx]
                features["metadata"].append(current_metadata)

                if not add_metadata:
                    features.pop("metadata")

            # End of the offset_mapping loop means features are added
            added=True


            local_max_bbox=0
            for x in features["bbox"]:
                for y in x:
                    if max(y) > local_max_bbox:
                         local_max_bbox = max(y)
            if max_bbox < local_max_bbox:
              max_bbox = local_max_bbox

        if not added:
            missed_features += 1 #print(idx, words,  "-missing-features-")


    print(current_split,  "missed features:", missed_features)
    print(max_bbox)

    return features


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
            #print(feature["image"][:10])
            if 'documents' not in feature["image"]: 

              dirname=os.path.dirname(feature["image"])
              basename=os.path.basename(feature["image"])
              imagef=dirname + "/documents/"+basename 

            else:
              imagef=feature["image"]

            #image = Image.open(feature["image"]).convert("RGB")
            image = Image.open(imagef).convert("RGB")

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
            #batch.pop("start_positions")
            #batch.pop("end_positions")
            #print("batch keys:", batch.keys())
            if "label_ids" in batch:
                #decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=batch["label_ids"])
                decoder_input_ids = batch["label_ids"]
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
    tokenizer = LayoutLMv3TokenizerFast.from_pretrained('microsoft/layoutlmv3-base')
    dataset = load_from_disk(dataset_file)

    # dataset = DatasetDict({"train": dataset["train"], "val": dataset['val']})
    # image_dir = {"train": "data/docvqa/train", "val": "data/docvqa/val", "test": "data/docvqa/test"}
    # use_msr = "msr_True" in dataset_file
    image_dir="./data/docvqg/train/documents"
    
    tokenized = dataset.map(tokenize_docvqa,
                             fn_kwargs={"tokenizer": tokenizer,
                                        "img_dir": image_dir,
                                        "use_msr_ocr": use_msr,
                                        "use_generation": False,
                                        "doc_stride": 128},
                            batched=True, num_proc=8,
                             load_from_cache_file=True)

    max_answer_length = -1
    max_input_length = -1
    from collections import Counter
    length_counter = Counter()
    split = "train"
    for obj in tqdm(dataset[split], total=len(dataset[split]), desc=split):
        for ans in obj["original_answer"]:
            if len(ans.split()) > max_answer_length:
                max_answer_length = len(ans)
        if len(obj["words"]) > 510:
            length_counter["500"] += 1
        if len(obj["words"]) > 600:
            length_counter["600"] += 1
        if len(obj["words"]) > 700:
            length_counter["700"] += 1
        if len(obj["words"]) > 800:
            length_counter["800"] += 1
        max_input_length = max(max_input_length, len(obj["words"]))

    print(f"maximum number of answer words: {max_answer_length}, maximum number of input words: {max_input_length}")
    print(length_counter)
    feature_extractor = LayoutLMv3FeatureExtractor.from_pretrained('microsoft/layoutlmv3-base', apply_ocr=False)
    collator = DocVQACollator(tokenizer, feature_extractor, 'microsoft/layoutlmv3-base')
    #loader = DataLoader(dataset.remove_columns("metadata"), batch_size=3, collate_fn=collator, num_workers=1)
    loader = DataLoader(dataset["train"], batch_size=3, collate_fn=collator, num_workers=1)
    for batch in loader:
        print(batch.input_ids)
