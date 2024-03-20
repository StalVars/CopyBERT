bert_models=/raid/svaranasi/saved_bert_models


if [ "$5" == "" ]
then

	echo "Usage: <script> output_dir model eval_check_steps gen_during_training predict_batch_size"
	exit
fi


output_dir=$1"-"$2
model=$2 #"saved-roberta-large"
modeltype=$(echo $model | cut -d "-" -f 1)
savedmodel="saved-"$model
evalsteps=$3
genduring=""
predict_batch_size=$5
learning_rate=$6

if [ "$4" != "0" ]
then
  genduring="--gen_during_train"
fi

python qg_examples/run_qg.py  --output_dir $output_dir  --train_file data/squad_unilm/squad.unilm.train.json --predict_file data/squad_unilm/squad.unilm.dev.json --predict_batch_size=${predict_batch_size} --train_batch_size=24 --gradient_accumulation_steps 24 --learning_rate=$learning_rate --num_train_epochs=5.0 --max_seq_length=384 --doc_stride=128 --bert_model ${bert_models}/$savedmodel --copymethod=0 --training_sanity_check_steps $evalsteps --do_train --data_cache_name $model --testset --model_type $modeltype $genduring $extra
 
