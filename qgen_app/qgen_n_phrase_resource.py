# -*- coding; utf-8 -*-

import json
import sys
import torch
import copy

sys.path.append("pytorch-pretrained-BERT_older/examples/") 
sys.path.append("pytorch-pretrained-BERT_older/") 
sys.path.append("./qgen_app/")

import identifyphrases

from examples.run_squad_qgen_generate import BertQGEN


class QGenResource:
    
    def __init__(self, model_name):
        self.bertqgen = BertQGEN(model_name)
    
    def on_post(self, req, resp):
        body = req.stream.read()
        body = json.loads(body)
        #data = dict(words=body["text"]) 
        data, starts, ends = identifyphrases.getphrases(body["text"], 10)
        new_data={"words":data["words"], "phrases":[]}
        new_starts=[]
        new_ends=[]
        for pi in range(len(data["phrases"])) :
              phr=data["phrases"][pi]
              print(phr)
              if phr["text"] not in [p["text"] for p in new_data["phrases"]]:
                  new_data["phrases"].append(phr)
                  new_starts.append(starts[pi])
                  new_ends.append(ends[pi])

        phrases = {
            "words": data["words"], "phrases": new_data["phrases"], 
            "starts": new_starts, "ends": new_ends
        }

        #starts = body["starts"]
        #ends = body["ends"]
        text = " ".join(data["words"])
        #data = dict(words=body["words"], phrases=body["phrases"])
        print("identified %s phrases.."%(len(phrases)))
        prediction = self.generate_qa_given_paragraph(text, data, starts, ends)
        resp.body = json.dumps(prediction, indent=2, ensure_ascii=False)
    
    def generate_qa_given_paragraph(self, text, data, starts, ends):
        posdata = dict()
        posdata["start"]=starts
        posdata["end"]=ends
        jsondata=json.dumps(data)
        posdata=json.dumps(posdata)
        qs_per_phrases=dict()
        qs_per_phrases["context"] = text
        qs_per_phrases["phrases"]=[]
        qs_per_phrases["questions"]=[]
        qs_per_phrases["scores"]=[]
          
        all_qs_per_phrases = copy.deepcopy(qs_per_phrases)
          
        for phrid in range(len(starts)):
            print("length of the paragraph=", len(text.split(" ")))
            phrase_adj=" ".join(data["words"][starts[phrid]:ends[phrid]+1])
            phrase=data["phrases"][phrid]
            print(phrase, phrase_adj)
            all_gens,all_scores = self.bertqgen.generate_per_text(text, 
                                        starts[phrid], 
                                        ends[phrid],
                                        get_qa_scores=False) 
            predict = all_gens[0]
            score = torch.exp(all_scores[0]).item()*100

            qs_per_phrases["questions"].append(predict)
            qs_per_phrases["phrases"].append(phrase)
            qs_per_phrases["scores"].append(score)
    
            
            '''
            for pi in range(len(all_gens)):
                predict=all_gens[pi]
                score = torch.exp(all_scores[pi]).item()*100
                all_qs_per_phrases["questions"].append(predict)
                all_qs_per_phrases["phrases"].append(phrase)
                all_qs_per_phrases["scores"].append(score)
            '''
          
        #return all_qs_per_phrases
        return qs_per_phrases
