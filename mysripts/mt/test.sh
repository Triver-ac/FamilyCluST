# model_dir=/home2/chu/fairseq_sign/sign/mysripts/mt/checkpoints
# CUDA_VISIBLE_DEVICES=1 \
# fairseq-generate /home2/chu/fairseq_sign/sign/dataset/mt/en-de/origin_mbart_index \
#   --path $model_dir/checkpoint_best.pt \
#   --task translation_from_pretrained_bart \
#   --gen-subset valid \
#   -t en_XX -s de_DE \
#   --bpe 'sentencepiece' --sentencepiece-model /home2/chu/fairseq_sign/sign/pretrained_models/mbart.cc25.v2/sentence.bpe.model \
#   --sacrebleu --remove-bpe 'sentencepiece' \
#   --batch-size 32 --langs $langs > de_en
# cat de_en | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[de_DE\]//g' |$TOKENIZER de > de_en.hyp
# cat de_en | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[de_DE\]//g' |$TOKENIZER de > de_en.ref
# sacrebleu -tok 'none' -s 'none' de_en.ref < de_en.hyp
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
CUDA_VISIBLE_DEVICES=2 python /home2/chu/fairseq_sign/sign/generate.py /home2/chu/fairseq_sign/sign/dataset/sp-10/index/de-en/add_short_mt_mbart_index \
    -t en_XX -s de_DE \
    --task translation_from_pretrained_bart \
    --post-process \
    --gen-subset valid \
    --remove-bpe \
    --batch-size 32 \
    --beam 5 \
    --lenpen 1 \
    --langs $langs \
    --path /home2/chu/fairseq_sign/sign/experiments/mt/de-en/sp-10/checkpoint_best.pt \
    --scoring sacrebleu |tee valid.log
cat valid.log | grep -P "^H" |sort -V |cut -f 3- | sed 's/\[de_DE\]//g' > de_en.hyp
cat valid.log | grep -P "^T" |sort -V |cut -f 2- | sed 's/\[de_DE\]//g' > de_en.ref
spm_decode --model=/home2/chu/fairseq_sign/sign/pretrained_models/mbart.cc25.v2/sentence.bpe.model --input_format=piece < de_en.hyp > output.hyp
spm_decode --model=/home2/chu/fairseq_sign/sign/pretrained_models/mbart.cc25.v2/sentence.bpe.model --input_format=piece < de_en.ref > output.ref
sacrebleu -tok 'none' -s 'none' output.ref < output.hyp