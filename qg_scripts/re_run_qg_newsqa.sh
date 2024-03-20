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

if [ "$4" != "0" ]
then
  genduring="--gen_during_train"
fi
output_dir2=$1"-1-"$2

python qg_examples/run_qg.py  --output_dir $output_dir2  --train_file data/newsqa/newsqa-v1-train.json --predict_file data/newsqa/newsqa-v1-dev.json --predict_batch_size=${predict_batch_size} --train_batch_size=6 --gradient_accumulation_steps 6 --learning_rate=3e-5 --num_train_epochs=3.0 --max_seq_length=384 --doc_stride=128 --bert_model $output_dir --copymethod=1 --training_sanity_check_steps $evalsteps --do_train --data_cache_name $model --testset --model_type $modeltype $genduring --version_2_with_negative

 
