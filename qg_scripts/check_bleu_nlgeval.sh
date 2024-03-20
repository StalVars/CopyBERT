


if [ "$1" == "" ]
then
  echo "first argument should be the generation file"
  exit
fi



src_file="/raid/svaranasi/seq2compress/pytorch-pretrained-BERT_older/unilm/src/qg_data/test/test.pa.txt"
tgt_file="/raid/svaranasi/seq2compress/pytorch-pretrained-BERT_older/unilm/src/qg_data/nqg_processed_data/tgt-test.txt"
output_file=$1
output_file_tok=${output_file%%.txt}.tok

sed 's/ ##//g' $output_file > $output_file_tok 

python2 /raid/svaranasi/unilm/unilm-v1/src/qg/eval.py --out_file $output_file_tok --tgt_file $tgt_file --src_file $src_file
