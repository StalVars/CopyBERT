
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

The following falcon server takes a passage and generates question, returns it to the client process '''

```sh
bash qg_scripts/generate_q.sh models/yahoo-nonfactoid bert-large-cased 10000 0 1 3e-5 -e --train_file data/yahoo-non-factoid/nfL6.train.squad.json --do_predict --predict_file ./data/yahoo-non-factoid/nfL6.dev.squad.json --bert_model models/yahoo-nonfactoid-bert-large-cased/ 
```
