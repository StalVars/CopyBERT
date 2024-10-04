

This is the code for the paper: 
https://aclanthology.org/2020.nlp4convai-1.3/

#Abstract:
Contextualized word embeddings provide better initialization for neural networks that deal with various natural language understanding (NLU) tasks including Question Answering (QA) and more recently, Question Generation(QG). Apart from providing meaningful word representations, pre-trained transformer models (Vaswani et al., 2017), such as BERT (Devlin et al., 2019) also provide self-attentions which encode syntactic information that can be probed for dependency parsing (Hewitt and Manning, 2019) and POStagging (Coenen et al., 2019). In this paper, we show that the information from selfattentions of BERT are useful for language modeling of questions conditioned on paragraph and answer phrases. To control the attention span, we use semi-diagonal mask and utilize a shared model for encoding and decoding, unlike sequence-to-sequence. We further employ copy-mechanism over self-attentions to acheive state-of-the-art results for Question Generation on SQuAD v1.1 (Rajpurkar et al., 2016)


# Env 
```sh
conda create -n copybert python=3.9
conda activate copybert
pip -r requirements.txt
python -m spacy download en_core_web_sm
```



#Train
```sh
bash qg_scripts/run_qg.sh models/baseline bert-large-cased 10000 0 1 3e-5 -e --train_file data/squad_unilm/squad.unilm.train.json --do_train --predict_file ./data/squad_unilm/squad.unilm.test.json
```

#Generate
```sh
bash qg_scripts/run_qg.sh models/baseline bert-large-cased 10000 0 1 3e-5 -e --train_file data/squad_unilm/squad.unilm.train.json --do_predict --predict_file ./data/squad_unilm/squad.unilm.test.json
```


#Demo Server  

The following falcon server takes a passage and generates question and answers, returns it to the client process '''

```sh
python qgen_app/app.py
```

curl command to invoke the server:
```sh
curl --header "Content-Type: application/json" --request POST --data '{"text": "Cars came into global use during the 20th century, and developed economies depend on them. The year 1886 is regarded as the birth year of the modern car when German inventor Karl Benz patented his Benz Patent-Motorwagen. Cars became widely available in the early 20th century. One of the first cars accessible to the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced animal-drawn carriages and carts, but took much longer to be accepted in Western Europe and other parts of the world."}' http://localhost:1223/qgennphr
```

# Checkpoints:
#
Download checkpoints from this location: https://drive.google.com/drive/folders/14rzRhhlQNMY_iqjxJzBzkhGbmZAhxhiY?usp=drive_link
