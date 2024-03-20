bert_models=/raid/svaranasi/saved_bert_models


if [ "$1" == "" ]
then

	echo "first argument should be a file"
	exit
fi


output_dir=$1

python qg_examples/run_qg_2ndtry.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad/dev-v1.1.json --predict_batch_size=1 --train_batch_size=6 --gradient_accumulation_steps 6 --learning_rate=3e-5 --num_train_epochs=3.0 --max_seq_length=384 --doc_stride=128 --bert_model ${bert_models}/saved-bert-base-cased --copymethod=1 --training_sanity_check_steps 1000 --do_lower_case --do_train --data_cache_name saved-bert-base-cased --testset 
 
