#!/bin/python


import sys
import json

fil=sys.argv[1]


with open(fil,"r") as f:
       data=json.load(f)



def get_data(data,typ="question"):


  data=data["data"]
  dat_list=[]
  for dat in data:
    for d in dat["paragraphs"]:
       for qa in d["qas"]:
            if typ=="question":
              print(qa["question"])
              dat_list.append(qa["question"])
            elif typ=="answer":
              dat_list.append(qa["answers"][0]["text"])
              print(qa["answers"][0]["text"])
            elif typ=="context":
              print(d["context"])
              dat_list.append(d["context"])
            elif typ=="qa":
               print(qa["answers"][0]["text"],":", qa["question"])
               dat_list.append((qa["answers"][0]["text"], qa["question"]))
     
  return dat_list



if len(sys.argv) < 3:
    typ="answer"
else:
    typ=sys.argv[2]
    if typ not in ["answer", "context", "question","qa"]:
        print("you can only fetch answer/context/question/qa from squad json file")
        sys.exit(1)

# get quesiton/answer/context from squad json

get_data(data, typ=typ)
