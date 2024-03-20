bert_models=/raid/svaranasi/saved_bert_models


if [ "$4" == "" ]
then

	echo "Usage: <script> output_dir model eval_check_steps gen_during_training"
	exit
fi


output_dir=$1"-"$2
model=$2 #"saved-roberta-large"
modeltype=$(echo $model | cut -d "-" -f 1)
savedmodel="saved-"$model
evalsteps=$3
genduring=""

if [ "$4" != "0" ]
then
  genduring="--gen_during_train"
fi

python qg_examples/check_qg_mem.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad/dev-v1.1.json --predict_batch_size=6 --train_batch_size=6 --gradient_accumulation_steps 6 --learning_rate=3e-5 --num_train_epochs=3.0 --max_seq_length=384 --doc_stride=128 --bert_model ${bert_models}/$savedmodel --copymethod=1 --training_sanity_check_steps $evalsteps --do_train --data_cache_name $model --testset --model_type $modeltype $genduring
 
