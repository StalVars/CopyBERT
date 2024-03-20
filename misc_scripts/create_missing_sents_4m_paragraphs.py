import json

import sys
import spacy
import re
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

if len(sys.argv) != 2:
    print("Usage:",sys.argv[0], "<textfile with paragraphs>")
    sys.exit(1)



def get_answer_starts(c,p,train=False):
        p_re=re.sub(r"([()+.?*\$\[\]\{\}])",r"\\\1",p) # make phrase searchable
        starts=[m.start(0) for m in re.finditer(p_re,c)]
        if train:
           starts=starts[0:1]

        return [{"text":p,"answer_start":start} for start in starts ]

def create_missing_sents(paragraphs):
    q_i=0
    new_paragraphs=[]
    questions=[]
    positions=[]
    for p_i in tqdm(range(len(paragraphs))):
            paragraph_text=paragraphs[p_i]
            doc=nlp(paragraph_text)
            sents=[ x.string.strip() for  x in doc.sents]
            if len(sents) <= 2:
               continue
            for s_i in range(len(sents)):
                new_sents = sents[:s_i] + ["[MASK]"] + sents[s_i+1:]
                answer_start = len(" ".join(sents[:s_i])) +  1 # +1 for the "space" after sents[:s_i]
                if s_i == 0:
                   answer_start=0 # no space when it is the first sentence
                answer_end = answer_start + 6 - 1 # [MASK] has 6 chars
                new_paragraph = " ".join(new_sents)
                question = sents[s_i]
                assert new_paragraph[answer_start:answer_end+1] == "[MASK]"
                new_paragraphs.append(new_paragraph)
                questions.append(question)
                positions.append((answer_start,answer_end))
    filetag="_"
    paragraphf=open("data/seq2seq/"+filetag+".source.txt","w")
    questionf=open("data/seq2seq/"+filetag+".target.txt","w")
    positionf=open("data/seq2seq/"+filetag+".positions.txt","w")
    assert len(new_paragraphs) == len(questions)
    assert len(new_paragraphs) == len(positions)
    print("Total datapoints:",len(positions))
    for pi in range(len(new_paragraphs)):
       paragraphf.write(new_paragraphs[pi]+"\n")
       questionf.write(questions[pi]+"\n")
       positionf.write(" ".join([str(el) for el in positions[pi]])+"\n")
       paragraphf.flush()
       questionf.flush()
       positionf.flush()
    paragraphf.close()
    questionf.close()
    positionf.close()
      
if __name__ == "__main__":
  paragraphs=[]
  with open(sys.argv[1],"r") as f:
    for line in f:
        line=line.strip()
        paragraphs.append(line)
  create_missing_sents(paragraphs)

