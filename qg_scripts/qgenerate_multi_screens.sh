if [[ "$1" == "" ]]
then

	echo "Usage: <script> <num_of_shards>"
	exit
fi

export num_of_shards=$1
export model=$2



shard_id=0
genfrom=0
gento=100
total=5000
split=$((total/num_of_shards))


echo "Use -e for script to not hang"
while [[ "$1" != "-e" ]]; do
	shift
done
shift


rest=$*




modelname=${##*/model}

echo "num_of_shards:$num_of_shards, model:$model, csv:$csv, output:$outputdir"
while [ $shard_id -lt $num_of_shards ]
do

	export genfrom=$((shard_id*split)) 
	export gento=$(((shard_id+1)*split)) 

	if [[ $shard_id -eq  $((num_of_shards-1)) ]]
	then
		echo "last shard"
		export gento=99999
	fi

	echo "####"
	echo "genfrom:"$genfrom
	echo "gento:"$gento
	echo "####"




 echo "shard_id:" $shard_id
 export shard_id num_of_shards modelname rest
 screen_name="qgeneration_shard-"$shard_id"-"$modelname
 screen -dmS ${screen_name}
 echo 'Running: bash ~/run_docker.sh bash generate_dense_embeddings.sh '$shard_id $num_of_shards $model $csv $outputdir"\n"
 #screen -S ${screen_name} -X stuff 'bash ~/run_docker.sh -p V100-32GB --mem 40G bash scripts/generate_dense_embeddings_flexi.sh $shard_id $num_of_shards $model $csv $outputdir -e \n'

 #bash ~/run_docker.sh  -p RTXA6000 --mem 100G bash qg_scripts/run_qg_actualviz.sh models/docvqg_w_actualviz bert-base-uncased 10000 0 1 3e-5 -e --train_file data/docvqa_format/docvqg.train.json --do_predict --predict_file ./data/docvqa_format/docvqg.test.json --do_lower_case --bert_model models/docvqg_w_actualviz-bert-base-uncased/ 

 screen -S ${screen_name} -X stuff 'cd /netscratch/svaranasi/raid-9206/copybert/ \n bash ~/run_docker.sh -p V100-32GB --mem 40GB \
	bash qg_scripts/run_qg_actualviz_w_images.sh models/docvqg_w_actualviz bert-base-uncased 10000 0 1 3e-5 -e --train_file data/docvqa_format/docvqg.train.json --do_predict --predict_file ./data/docvqa_format/docvqg.test.json --bert_model $model --output_dir $model  --genfrom $genfrom --gento $gento $rest \n'

 shard_id=$((shard_id+1))

 read
done

