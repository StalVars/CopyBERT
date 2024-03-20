


if [ "$1" == "" ]
then
  echo "first argument should be the generation file"
  exit
fi



src_file="/raid/svaranasi/seq2compress/pytorch-pretrained-BERT_older/unilm/src/qg_data/test/test.pa.txt"
tgt_file="generations/tgt-test.txt"
output_file=$1
output_file_tok=${output_file%%.txt}.tok
output_file_low=${output_file%%.txt}.low

#sed 's/ ##//g' $output_file > $output_file_tok 
#cat $output_file_tok  | tr '[A-Z]' '[a-z]' > $output_file_low 

#python qg_scripts/nlg_eval.py $output_file_tok $tgt_file
nlg-eval --hypothesis=$output_file --references=$tgt_file

