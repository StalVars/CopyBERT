#


# To train
bash qg_scripts/run_qg.sh models/baseline bert-large-cased 10000 0 1 3e-5 -e --train_file data/squad_unilm/squad.unilm.train.json --do_train --predict_file ./data/squad_unilm/squad.unilm.test.json



# To generate questions for squad test json
bash qg_scripts/run_qg.sh models/baseline bert-large-cased 10000 0 1 3e-5 -e --train_file data/squad_unilm/squad.unilm.train.json --do_predict --predict_file ./data/squad_unilm/squad.unilm.test.json


# Evaluate generatation


# Demo

# Start the falcon server that when submitted a context, generates question and return it to the client process

bash ~/run_docker.sh -p V100-32GB bash qg_scripts/generate_q.sh models/yahoo-nonfactoid bert-large-cased 10000 0 1 3e-5 -e --train_file data/yahoo-non-factoid/nfL6.train.squad.json --do_predict --predict_file ./data/yahoo-non-factoid/nfL6.dev.squad.json --bert_model models/yahoo-nonfactoid-bert-large-cased/ 

# Call the falcon client



# To run CopyLayout QG:
bash qg_scripts/run_qg_viz.sh models/dovqg_w_viz bert-large-cased 10000 0 1 3e-5 -e --train_file data/docvqa_format/docvqg.train.json --do_train --predict_file ./data/docvqa_format/docvqg.val.json
