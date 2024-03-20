bert_models=/raid/svaranasi/saved_bert_models


if [ "$1" == "" ]
then

	echo "first argument should be a file"
	exit
fi

output_dir=$1

python qg_examples/run_qg_bert.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad/dev-v1.1.json --predict_batch_size=1 --train_batch_size=1 --gradient_accumulation_steps 1 --learning_rate=3e-5 --num_train_epochs=3.0 --max_seq_length=384 --doc_stride=128 --model_name_or_path ${bert_models}/saved-span-bert-case-large --copymethod=1 --training_sanity_check_steps 1000 --do_train --data_cache_name saved-span-bert-case-large --testset

#python qg_examples/run_qg_bert.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad/dev-v1.1.json --predict_batch_size=1 --train_batch_size=1 --gradient_accumulation_steps 1 --learning_rate=3e-5 --num_train_epochs=3.0 --max_seq_length=384 --doc_stride=128 --model_name_or_path ${bert_models}/saved-span-bert-case-large --copymethod=1 --training_sanity_check_steps 10000 --do_lower_case --do_train --data_cache_name saved-span-bert-case-large --testset
