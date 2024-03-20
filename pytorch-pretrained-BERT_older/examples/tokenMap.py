#!/bin/python

''' 
 Class for tokenizing / mapping tokenizations

'''

import re
from difflib import SequenceMatcher
from nltk import metrics
from pytorch_pretrained_bert import BertTokenizer
#berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
berttokenizer = BertTokenizer.from_pretrained('/raid/data/stva02/saved_bert_models/saved-bert-base-cased')


class AlignTokenizations:

 def get_word_boundaries(self,words):
   #remove hashes(for BERT)
   words=[re.sub("^##","",word) for word in words]
   words=[re.sub(u"\s","",word) for word in words]

   #without_spaces=re.sub("[	 ]",""," ".join(words))
   #print(" ".join(words))

   without_spaces=re.sub(u"\s",""," ".join(words))

   #print(without_spaces)
   #input("without spaces")
   #chars_nospace=[ c for c in " ".join(without_spaces)]

   chars_nospace=[ c for c in without_spaces ]
   chars_withspaces=[ c for c in " ".join(words)]
   char2word=[-1 for i in chars_withspaces]

   word_no=0
   for ci in range(len(chars_withspaces)):
     if chars_withspaces[ci] == " ":
        word_no += 1
        continue
     char2word[ci]=word_no

   char2word = [  i for i in char2word if i != -1 ] # Remove -1's to get len(char2word) = len(chars_nospace)
   wordboundaries = [-1*(char2word[i] - char2word[i+1]) for i in range(len(char2word)-1)] +[1] # Assuming,last character is not space - it will be a word boundary
  
   return chars_nospace, char2word,wordboundaries
    

 def map_only_seen(self, words1, words2, interms_of=1):
   char2word1,wordboundaries_1=self.get_word_boundaries(words1)
   char2word2,wordboundaries_2=self.get_word_boundaries(words2)

   if len(char2word1) != len(char2word2):
      print("The number of characters in the two sentences didn't match")
      return False
   print(wordboundaries_1)
   print(wordboundaries_2)
   input("word boundaries")

   matched_boundaries = [ wordboundaries_1[i]*wordboundaries_2[i] for i in range(len(wordboundaries_1)) ]

 def sim(self,word1, word2):
     strictness = 0.8
     word1 = word1.lower()
     word2 = word2.lower()
     if len(word1) <= len(word2):
        similarity = SequenceMatcher(None, word1, word2)
        #similarity = metrics.edit_distance(word, search_key)
        #print(similarity.ratio())
        if similarity.ratio() > strictness:
            return True
     return False

 def map_alternative(self,words1, words2,interms_of=1):
     #words1 is a list
     #words2 is a list

     words1 = [ re.sub("^##","",word) for word in words1 ]
     words2 = [ re.sub("^##","",word) for word in words2 ]
     bert_words1 = [ berttokenizer.tokenize(word) for word in words1 ]
     #print(len(words1))

     words1 = [ "".join([ re.sub("^##","",word) for word in words ]) for words in bert_words1 ]

     #print(len(words1))
     #input("words1")

     bert_words2 = [ berttokenizer.tokenize(word) for word in words2 ]
     words2 = [ "".join([ re.sub("^##","",word) for word in words ]) for words in bert_words2 ]
     map1 = self.map(words1,words2)
     #print(map1)
     return map1


 def map_alternative2(self,words1, words2,interms_of=1):

     mapping1=[[] for i in words1 ]
     word2_index=0
     word1_index=0
     while word1_index <= len(words1)-1:
         word1 = words1[word1_index]
         word2 = words2[word2_index]

         if self.sim(word1,word2) or re.search(re.escape(word1.lower()),re.escape(word2.lower())):
            mapping1[word1_index] == [word2_index]
            if re.search(re.escape(word1.lower())+"$",re.escape(word2.lower())):
               print(word1,word2)
               word2_index += 1
            word1_index += 1
         elif self.sim(word2,word1) or re.search(re.escape(word2.lower()),re.escape(word1.lower())):
            mapping1[word1_index].append(word2_index)
            word2_index += 1
            if re.search(re.escape(word2.lower())+"$",re.escape(word1).lower()):
            #if len(re.sub(word2,"",word1)) > 0:
                word1_index += 1
            print(word1,"-->",word2)
         else:
            print("else")
            if len(mapping1[word1_index]) == 0: 
                  if word1_index != 0:
                   mapping1[word1_index] = mapping1[word1_index-1]
                  else:
                   mapping1[word1_index] = [0]
            word1_index += 1
            word2_index += 1
            #if len(word1) <= len(word2):
            #else:
            print(word1,word2,word1_index, word2_index)
            input("ok")

     print(mapping1)
     for i in range(len(mapping1)):
         if len(mapping1[i]) == 0 and i !=0:
             mapping1[i] = mapping1[i-1]
         elif i == 0:
             mapping1[i] = [0]
     print(mapping1)
     print(words1)
     print(words2)
     return mapping1
       


 def map(self,words1, words2,interms_of=1):

   '''
   print(words1)
   print(words2)
   print("Before")
   # convert the characters into byte?
   #words1 = [ re.sub("[\"']$","",re.sub("^b[\"']","",str(word.encode("utf-8")))) for word in words1 ]
   #words2 = [ re.sub("[\"']$","",re.sub("^b[\"']","",str(word.encode("utf-8")))) for word in words2 ]
   print(words1)
   print(words2)
   input("After")
   '''

   nospace_chars1, char2word1,wordboundaries_1=self.get_word_boundaries(words1)
   nospace_chars2, char2word2,wordboundaries_2=self.get_word_boundaries(words2)

   if len(char2word1) != len(char2word2):
      #print("unequal lengths")
      #map1 = self.map(words1, basic_words1) 
      #map2 = self.map(basic_words1, words2)

      #print("The number of characters in the two sentences didn't match")
      #print(words1)
      #print(words2)
      #print(nospace_chars1)
      #print(nospace_chars2)
      #print(char2word1)
      #print(char2word2)
      #print("checking for mismatches")
      #basic_words1 = berttokenizer.basic_tokenizer.tokenize(" ".join(words1))
      '''
      '''
      map1 = self.map_alternative(words1, words2)

      return map1

   matched_boundaries = [ wordboundaries_1[i]*wordboundaries_2[i] for i in range(len(wordboundaries_1)) ]

   word_segments_1=[]
   word_segments_2=[]
   mapping1 = [[] for i in words1]
   mapping2 = [[] for i in words2]

   for i in range(len(matched_boundaries)):
     if char2word1[i] not in word_segments_1:
        word_segments_1.append(char2word1[i])
     if char2word2[i] not in word_segments_2: 
        word_segments_2.append(char2word2[i])
     if matched_boundaries[i] == 1:
       for x in word_segments_1:
        mapping1[x] = word_segments_2
       for x in word_segments_2:
        mapping2[x] = word_segments_1
       word_segments_1=[]
       word_segments_2=[]
   if interms_of == 1:
      return mapping1
   else:
      return mapping2

def main():
   from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
   berttokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
   sample_text = "This is a sample text to check if BERTtokenization is rightlymatched"
   bert_tokens = berttokenizer.tokenize(sample_text)
   normal_tokens = sample_text.split(" ")
   align=AlignTokenizations()
   map_interms_of_seq1 = align.map(normal_tokens,bert_tokens) 
   print(normal_tokens)
   print(bert_tokens)
   print(map_interms_of_seq1)


if __name__ == "__main__":
   main()
   
