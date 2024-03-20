bert_models=../saved_bert_models/


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

bert_model=${bert_models}/$savedmodel
#bert_model=$model 
python=/netscratch/svaranasi/miniconda3/envs/latesthf/bin/python

# 7th argument to train/predict 
if [ "$7" == "re" ]
then
tag=$8
bert_model=$output_dir
#predictfile=data/xquad/xquad.en.json #data/squad_unilm/squad.unilm.test.json
#do="--do_train" #predict"
output_dir=$1"-o-"$tag"-o-"$2
else
echo
#predictfile=data/xquad/xquad.en.json #data/squad_unilm/squad.unilm.test.json
#do="--do_train"
fi

predictfile_w_sent=data/dev-v1.1.json.noq.hotpotifiedWithPhrases
procpernode=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
echo $CUDA_VISIBLE_DEVICES
echo "Using $procpernode gpu(s).."

if [ "$4" != "0" ]
then
  genduring="--gen_during_train"
fi

# Shift till you see -e 
while [[ "$1" != "-e" ]]; do
	shift
done
shift

random=`$python -c "import random;print(random.randint(1000,9999))"`
master_arg="--master_port $random"
echo "MASTER PORT:$master_arg"

extra_opts=$*
echo "extra_opts:" $*
echo "python:" $python

#predict_dir="./data/xquad_sample/"
#predict_dir="./data/xquad/"

#python qg_examples/run_qg_new.py  --output_dir $output_dir  --train_file data/squad_unilm/squad.unilm.train.json --predict_file data/squad_unilm/squad.unilm.dev.json --predict_batch_size=${predict_batch_size} --train_batch_size=24 --gradient_accumulation_steps 24 --learning_rate=$learning_rate --num_train_epochs=5.0 --max_seq_length=384 --doc_stride=128 --bert_model ${bert_models}/$savedmodel --copymethod=1 --training_sanity_check_steps $evalsteps --do_train --data_cache_name $model --testset --model_type $modeltype  $extra

		#--train_dir data/train_dir_squad_unilm/ \
		#--train_dir data/german_pretrain \
		#--predict_dir ${predict_dir} \
		#--predict_file $predictfile \
		#$do \
		#$genduring \

$python qg_examples/run_qg_decoder.py  --output_dir $output_dir \
		--predict_batch_size=${predict_batch_size} \
		--learning_rate=$learning_rate \
		--num_train_epochs=7 \
		--max_seq_length=384 \
		--doc_stride=128 \
		--bert_model $bert_model \
		--copymethod=0 \
		--training_sanity_check_steps $evalsteps \
		--data_cache_name $model \
		--model_type $modeltype \
		--train_batch_size=24 \
		--gradient_accumulation_steps 24 \
		$extra_opts
 
