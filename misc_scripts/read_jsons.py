#!/bin/python

import spacy

import json
import re
import sys
nlp = spacy.load("en_core_web_sm")

if len(sys.argv) !=2:
    print("Usage:",sys.argv[0],"<squad-like-json-file>")
    sys.exit(1)
squad_json = sys.argv[1]
squad_json_target = sys.argv[1]+".targ"

with open(squad_json,"r") as f:
     jdata=json.load(f)


def get_answer_starts(c,p,train=False):
        p_re=re.sub(r"([()+.?*\$\[\]\{\}])",r"\\\1",p) # make phrase searchable
        starts=[m.start(0) for m in re.finditer(p_re,c)]
        if train:
          starts=starts[0:1]
        return [{"text":p,"answer_start":start} for start in starts ]

newjdata={"data":[],"version":"sentence wise dev"}
for d in jdata["data"]:
    new_d=dict()
    title=d["title"]
    paras=d["paragraphs"]
    idx=0
    new_paras=[]

    for para in paras:

        idx += 1
        context=para["context"]
        #doc = nlp(context)
        qas=para["qas"]
        for qa in qas:
            if not qa['is_impossible']:
             print(qa['is_impossible'])
             input("is impossible")
    '''
    print(paras,len(paras))
    input("--paras--")
    print(new_paras,len(new_paras))
    input("--new paras--")
    '''
    new_d["title"]=title
    new_d["paragraphs"]=new_paras
    newjdata["data"].append(new_d)

#with open("squad_jsons/dev-v1.1.sentence_wise.json","w") as f:
with open(squad_json_target,"w") as f:
    json.dump(newjdata,f)
'''
'''

