import json

import sys
import spacy
import re
nlp = spacy.load("en_core_web_sm")

if len(sys.argv) != 2:
    print("Usage:",sys.argv[0], "<jsonfile>")
    sys.exit(1)

'''
questions=[]
with open(sys.argv[1],"r") as f:
    for line in f:
        line=line.strip()
        questions.append(line)
'''

json_file = sys.argv[1]

def get_answer_starts(c,p,train=False):
        p_re=re.sub(r"([()+.?*\$\[\]\{\}])",r"\\\1",p) # make phrase searchable
        starts=[m.start(0) for m in re.finditer(p_re,c)]
        if train:
           starts=starts[0:1]

        return [{"text":p,"answer_start":start} for start in starts ]

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
    json_questions=[]
    q_i=0
    new_data=[]
    paragraphs=[]
    questions=[]
    positions=[]
    for e_i in range(len(input_data)):
        entry = input_data[e_i]
        new_entry={"paragraphs":[]}
        new_paragraphs=[]
        for p_i in range(len(entry["paragraphs"])):
            paragraph = entry["paragraphs"][p_i]
            paragraph_text = paragraph["context"]
            #doc=nlp(paragraph_text)
            for qa_i in range(len(paragraph["qas"])):
                qa = paragraph["qas"][qa_i]
                qas_id = qa["id"]
                question_text = qa["question"]
                answer_text = qa["answers"][0]["text"]
                answer_start = qa["answers"][0]["answer_start"]
                paragraphs.append(paragraph_text)
                questions.append(question_text)
                positions.append((answer_start,answer_start+len(answer_text)-1))

                '''
                for sent in doc.sents:
                    sent_text = sent.string.strip()
                    if answer_text in sent_text:
                        new_paragraph = dict()
                        #input("given answers")
                        para_answers=get_answer_starts(paragraph_text, answer_text) # adjust start position
                        #print(para_answers)
                        #input("searched para answers")
                        answers=get_answer_starts(sent_text, answer_text) # adjust start position
                        #print(answers)
                        #print(len(answer_text), answers, len(sent_text))
                        if qa["answers"][0]["answer_start"] == 0:
                            print(sent_text,answer_text)
                            #input("wait-2")
                        assert len(answer_text)+int(answers[0]["answer_start"]) <= len(sent_text)
                        qa["answers"]=answers
                        qa["question"] = question_text 
                        new_paragraph["qas"]=[qa]
                        new_paragraph["context"] = sent_text
                        new_paragraphs.append(new_paragraph)
                '''
                q_i += 1
    filetag=re.sub(".json$","",re.sub("^.*/","",input_file))
    paragraphf=open("data/seq2seq/"+filetag+".source.txt","w")
    questionf=open("data/seq2seq/"+filetag+".target.txt","w")
    positionf=open("data/seq2seq/"+filetag+".positions.txt","w")
    assert len(paragraphs) == len(questions)
    assert len(paragraphs) == len(positions)
    print("Total datapoints:",len(positions))
    for pi in range(len(paragraphs)):
       paragraphf.write(paragraphs[pi]+"\n")
       questionf.write(questions[pi]+"\n")
       positionf.write(" ".join([str(el) for el in positions[pi]])+"\n")
       paragraphf.flush()
       questionf.flush()
       positionf.flush()
    paragraphf.close()
    questionf.close()
    positionf.close()
      

eval_test_examples = read_squad_examples(
            input_file=json_file, is_training=False, version_2_with_negative=True)

