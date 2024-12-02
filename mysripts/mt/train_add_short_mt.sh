PRETRAIN=/home2/chu/fairseq_sign/sign/pretrained_models/mbart.cc25.v2/model.pt
langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
ST_SAVE_DIR=/home2/chu/fairseq_sign/sign/experiments/mt/de-en/add_short_sp-10
rm -r $ST_SAVE_DIR
mkdir $ST_SAVE_DIR -p
CUDA_VISIBLE_DEVICES=2 \
nohup fairseq-train /home2/chu/fairseq_sign/sign/dataset/sp-10/index/de-en/add_short_mt_mbart_index \
  --distributed-world-size 1 \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_large --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang de_DE --target-lang en_XX \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 \
  --total-num-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 1024 --update-freq 2 \
  --keep-best-checkpoints 5 --no-epoch-checkpoints \
  --save-dir  $ST_SAVE_DIR \
  --seed 222 --log-format simple --log-interval 100 \
  --restore-file $PRETRAIN \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --langs $langs \
  --ddp-backend legacy_ddp \
  --validate-after-updates 5000 \
  > train_add_short_mt.log 2>&1 &
  # --eval-bleu --eval-bleu-detok space --eval-bleu-remove-bpe sentencepiece --eval-bleu-print-samples \
  # --best-checkpoint-metric bleu \
  # --scoring sacrebleu \
  # --maximize-best-checkpoint-metric \
