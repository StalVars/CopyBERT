#!/bin/python


import json
import re
import spacy
import nltk
import random
from argparse import ArgumentParser
nlp = spacy.load("en_core_web_sm")
import sys

from flair.data import Sentence
from flair.models import SequenceTagger
import os
from sutime import SUTime
import re

ner_tagger = SequenceTagger.load('ner')
pos_tagger = SequenceTagger.load('pos-multi')


def get_answer_starts(c,p,train=True):
        #p_re=re.sub(r"([()+.?*\$\[\]\{\}])",r"\\\1",p) # make phrase searchable
        p_re=re.escape(p) #
        starts=[m.start(0) for m in re.finditer(p_re,c)]
        if train:
          starts=starts[0:1]
        return [{"text":p,"answer_start":start} for start in starts ]


# Change the below location based on the sutime - jars' location
jar_files = os.path.join(os.path.dirname("/raid/data/stva02/aqeg_d3_d1/misc_scripts/pretrain_eqa/sutime/python-sutime/jars"), 'jars')
#sutime = SUTime(jars=jar_files, mark_time_ranges=True)


def wordscores(textsplit, starts,ends): 
   scores=[0 for i in range(len(textsplit))]
   for i in range(len(starts)): 
      #print(starts[i],ends[i],end_probs[i])
      for el in range(starts[i],ends[i]+1):
        scores[el]=i+1
   return scores

def wordscores_old(textsplit, starts,ends,start_probs,end_probs):
   scores=[0 for i in range(len(textsplit))]
   for i in range(len(starts)): 
      print(starts[i],ends[i],end_probs[i])
      for el in range(starts[i],ends[i]+1):
        scores[el]=np.exp(end_probs[i])
      scores[starts[i]]=np.exp(start_probs[i])
   return scores

def getphrases(paragraph,phrasecount=5):

   paragraph=paragraph.strip()
   sentsplit=paragraph.split()

   #phrasewords, attentions,start_poses,end_poses,start_probs, end_probs = findphrases( embedd_enc, embedd_dec, encoder1, encoder2, attn_decoder1, concat1, paragraph,100,phrasecount)
   #sentsplit, start_poses,end_poses,start_probs, end_probs = bert_findphrases( berttokenizer, bertmodel, paragraph,100,phrasecount)
   #sentsplit, start_poses,end_poses,start_probs, end_probs = spacyflair_findphrases( paragraph,phrasecount)
   context, phrases = spacyflair_findphrases( paragraph,phrasecount)

   print("Context: [length:%s] %s " % (len(context), context) )

   #print(phrases)

   #coloured_text = colourtext(sentsplit,start_poses,end_poses)
   #print(sentsplit)

   ''' get sentsplit, start_poses, end_poses'''
   sentsplit = []
   re.sub("  *", " ", context) # convert extra spaces to 1 space
   char2word = [ci for ci in range(len(context))]
   text = ""

   for ci in range(len(context)):

      char2word[ci] = len(sentsplit) # length starts from 1, so character index ci is mapped with next word (assuming word list starts from index 0)
      c = context[ci]
      
      if c == " ":
         sentsplit.append(text)
         text = ""
      else:
         text += c

   # Add the last word
   if text != "":
         sentsplit.append(text)


   print("list of words", sentsplit)


   start_poses=[]
   end_poses=[]
   unique_phrases=[]
   unique_squad_like_phrases=[]
   for phrase in phrases:

      phrase_start_c = phrase["answer_start"]
      phrase_end_c = phrase_start_c + len(phrase["text"])

      if phrase_end_c < len(context):

        phrase_start_w = char2word[phrase_start_c]
        phrase_end_w = char2word[phrase_end_c]

        if phrase_end_c < len(context) : #and (phrase_start_w, phrase_end_w) not in unique_phrases:
          print("phrase = ", sentsplit[phrase_start_w:phrase_end_w+1], phrase["text"])
          start_poses.append(phrase_start_w)
          end_poses.append(phrase_end_w)
          unique_phrases.append( (phrase_start_w, phrase_end_w) )
          unique_squad_like_phrases.append(phrase)

      else:
        continue

   ## Limit by phrasecount here if necessary:
   #start_poses = start_poses[:phrasecount]
   #end_poses = end_poses[:phrasecount]

   ''' '''

   scores = wordscores(sentsplit,start_poses,end_poses) 

   #print(scores)
   print("sent split", sentsplit)
   data=dict()
   data["words"]=sentsplit
   data["scores"]=scores
   data["phrases"]=unique_squad_like_phrases

   #return coloured_text+" ---- "+" ".join(phrasewords)+" ---- "+" ".join([str(p) for p in start_poses])+" ---- "+" ".join([str(p) for p in end_poses]) 
   return data,start_poses,end_poses 


def spacyflair_findphrases(context, phrasecount=5):

      ''' sentence wise '''
      '''
      doc = nlp(context)
      sentences=[]
      for sent in doc.sents:
          sentences.append(sent.string.strip())
      '''

      sentences=[context]
      tagged_sentences = [Sentence(s) for s in sentences ]
      # Named Entities Identification
      for sent_id, sentence in enumerate(tagged_sentences):  # first sentence is the paragraph title
            pos_tagger.predict(sentence)
            labels=[]
            texts=[]
            adj_phrases=[]
            t_pos=[]
            for token in sentence.tokens:
                #print(token)
                l=token.get_labels()[0]
                t=token.text
                labels.append(l.value)

                '''
                if l.value == "ADJ" or l.value == "NUM":
                    phr=dict()
                    phr["text"]=t
                    #adict = get_answer_starts(sentence_text, t)
                    #start_pos = adict[0]["answer_start"]
                    phr["answer_start"]=t.start_pos
                    adj_phrases.append(phr)
                '''

            

            ner_tagger.predict(sentence)
            ne_phrases=[]
            for entity in sentence.get_spans('ner'):
                    phr=dict()
                    text=entity.text
                    if entity.text.endswith(('.','?','!',',',':')): # counter tagging errors
                            text = entity.text[:-1]
                    phr["text"]=text
                    phr["answer_start"]=entity.start_pos
                    ne_phrases.append(phr)
            total_phrases=ne_phrases
            start_positions=[phr["answer_start"] for phr in total_phrases]
            



            ###  Add ADJ PHRASES ? ###
            '''
            for phr in adj_phrases:
                if phr["answer_start"] not in start_positions:
                    total_phrases.append(phr)
            '''

            #time_json=sutime.parse(sentence.to_plain_string())
            time_json=[]

            start_positions=[phr["answer_start"] for phr in total_phrases]
            ## TIME PHRASES ##
            time_phrases=[]
            for tphr in time_json:
                phr=dict()
                phr["text"] = tphr["text"]
                phr["answer_start"] = tphr["start"]
                time_phrases.append(phr)

            for phr in time_phrases:
                if phr["answer_start"] not in start_positions:
                    total_phrases.append(phr)

            #print("Just NUM and ADJ", adj_phrases)
            print("NE phrases", ne_phrases)
            print("time phrases", time_phrases)

      return context, total_phrases


if __name__ == '__main__':
   phrases=getphrases("The Bombing of Yawata on the night of 15/16 June 1944 was the first air raid on the Japanese home islands conducted by United States Army Air Forces (USAAF) strategic bombers during World War II. The attack was undertaken by 75 B-29 Superfortress heavy bombers (examples pictured) staging from bases in China. Only 47 of these aircraft dropped bombs near the raid's primary target, the Imperial Iron and Steel Works at Yawata, and little damage was caused.")

   print(phrases)
