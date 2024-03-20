bert_models=/raid/data/stva02/saved_bert_models


if [[ "$5" == "" ]]
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
export splitlen=$((11877/$splits/2))

echo "last cuda:" $last_cuda
echo "split len :" $splitlen
shift
export rest_cudas=$*

export genfrom=0
export gento=$splitlen
for dash in 1 2
do
for cuda in $rest_cudas
do
echo $genfrom $gento $cuda
screen -dmS $model"-test-"$cuda"-"$dash
screen -S $model"-test-"$cuda"-"$dash -X stuff 'export\n'
screen -S $model"-test-"$cuda"-"$dash -X stuff 'CUDA_VISIBLE_DEVICES=$cuda python qg_examples/run_qg_eval.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad_unilm/squad.unilm.test.json --predict_batch_size=1 --train_batch_size=6 --gradient_accumulation_steps 6 --learning_rate=3e-5 --num_train_epochs=3.0 --max_seq_length=384 --doc_stride=128 --bert_model $output_dir --copymethod=1  --do_predict --data_cache_name $model --model_type $modeltype --tok_model ${bert_models}/$savedmodel $extra --genfrom $genfrom --gento $gento --tag $cuda"-"$dash\n'
export genfrom=$((genfrom+splitlen))
export gento=$((gento+splitlen))

read
done
done

echo $genfrom $gento $last_cuda
screen -dmS $model"-test-"${last_cuda}"-0"
screen -S $model"-test-"${last_cuda}"-0" -X stuff 'export\n' 
screen -S $model"-test-"${last_cuda}"-0" -X stuff 'CUDA_VISIBLE_DEVICES=$last_cuda python qg_examples/run_qg_eval.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad_unilm/squad.unilm.test.json --predict_batch_size=1 --train_batch_size=6 --gradient_accumulation_steps 6 --learning_rate=3e-5 --num_train_epochs=3.0 --max_seq_length=384 --doc_stride=128 --bert_model $output_dir --copymethod=1  --do_predict --data_cache_name $model --model_type $modeltype --tok_model ${bert_models}/$savedmodel $extra --genfrom $genfrom --gento $gento --tag $last_cuda"-0"\n'
 
export genfrom=$gento
export gento=$((gento+splitlen))
echo $genfrom $gento $last_cuda

screen -dmS $model"-test-"${last_cuda}"-1"
screen -S $model"-test-"${last_cuda}"-1" -X stuff 'export\n' 
screen -S $model"-test-"${last_cuda}"-1" -X stuff 'CUDA_VISIBLE_DEVICES=$last_cuda python qg_examples/run_qg_eval.py  --output_dir $output_dir  --train_file data/squad/train-v1.1.json --predict_file data/squad_unilm/squad.unilm.test.json --predict_batch_size=1 --train_batch_size=6 --gradient_accumulation_steps 6 --learning_rate=3e-5 --num_train_epochs=3.0 --max_seq_length=384 --doc_stride=128 --bert_model $output_dir --copymethod=1  --do_predict --data_cache_name $model --model_type $modeltype --tok_model ${bert_models}/$savedmodel $extra --genfrom $genfrom --tag $last_cuda"-1"\n'

