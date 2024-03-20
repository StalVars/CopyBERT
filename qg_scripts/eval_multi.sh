#bert_models=/raid/data/stva02/saved_bert_models


if [[ "$4" == "" ]]
then

	echo "Usage: <script> output_dir model 0 1 2 <cuda devices(atleast 2)>"
	exit
fi


export output_dir=$1"-"$2
export model=$2 #"saved-roberta-large"
export modeltype=$(echo $model | cut -d "-" -f 1)
export savedmodel="saved-"$model

echo $*
# consume 1,2 arguments
shift
shift

export last_cuda=$1
export splits=$#
export splitlen=$((11877/$splits))

echo "last cuda:" $last_cuda
echo "split len :" $splitlen
shift
export rest_cudas=$*

export genfrom=0
export gento=$splitlen
for cuda in $rest_cudas
do
export cuda
echo $genfrom $gento $cuda

screen_name=$cuda"-"$model"-test-"$cuda"-"$dash
screen -dmS ${screen_name}
screen -S ${screen_name} -X stuff 'export\n'
screen -S ${screen_name} -X stuff 'source activate clone\n'
screen -S ${screen_name} -X stuff 'CUDA_VISIBLE_DEVICES=$cuda python qg_examples/run_qg.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad_unilm/squad.unilm.test.json --predict_batch_size=1 --train_batch_size=6 --gradient_accumulation_steps 6 --max_seq_length=384 --doc_stride=128 --bert_model $output_dir --copymethod=1 --do_predict --data_cache_name $model --model_type $modeltype  --genfrom $genfrom --gento $gento --tag $genfrom"-"$gento\n'


#screen -S $model"-test-"$cuda"-"$dash -X stuff CUDA_VISIBLE_DEVICES=$cuda python qg_examples/run_qg_eval.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad_unilm/squad.unilm.test.json --predict_batch_size=1 --train_batch_size=6 --gradient_accumulation_steps 6 --learning_rate=3e-5 --num_train_epochs=3.0 --max_seq_length=384 --doc_stride=128 --bert_model $output_dir --copymethod=1  --do_predict --data_cache_name $model --model_type $modeltype --tok_model ${bert_models}/$savedmodel $extra --genfrom $genfrom --gento $gento --tag $cuda"-"$dash\n
export genfrom=$((genfrom+splitlen))
export gento=$((gento+splitlen))

read
done


screen_name=${last_cuda}"-"$model"-test-"${last_cuda}"-1"
screen -dmS ${screen_name}
screen -S ${screen_name} -X stuff 'export\n' 
screen -S ${screen_name} -X stuff 'source activate clone\n'
screen -S ${screen_name} -X stuff 'CUDA_VISIBLE_DEVICES=$last_cuda python qg_examples/run_qg.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad_unilm/squad.unilm.test.json --predict_batch_size=1 --train_batch_size=6 --gradient_accumulation_steps 6 --max_seq_length=384 --doc_stride=128 --bert_model $output_dir --copymethod=1 --do_predict --data_cache_name $model --model_type $modeltype  --genfrom $genfrom --tag $genfrom"-"\n'

