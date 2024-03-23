
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
url --header "Content-Type: application/json" --request POST --data '{"question": "When was the car invented?","text": "Cars came into global use during the 20th century, and developed economies depend on them. The year 1886 is regarded as the birth year of the modern car when German inventor Karl Benz patented his Benz Patent-Motorwagen. Cars became widely available in the early 20th century. One of the first cars accessible to the masses was the 1908 Model T, an American car manufactured by the Ford Motor Company. Cars were rapidly adopted in the US, where they replaced animal-drawn carriages and carts, but took much longer to be accepted in Western Europe and other parts of the world."}' http://localhost:1223/qgennphr
```
