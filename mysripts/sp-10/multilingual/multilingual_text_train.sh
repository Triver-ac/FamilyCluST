# ST_SAVE_DIR=/home2/chu/fairseq_sign/sign/experiments/sp-10/en-en
INPUT_TYPE=text
SRC_LANGS=zh,uk,ru,bg,is,de,it,sv,lt,en
TGT_Langs=zh,uk,ru,bg,is,de,it,sv,lt,en
CONFIG_YAML=config-wiki-all.yaml
ST_SAVE_DIR=/home2/chu/fairseq_sign/sign/experiments/sp-10/multilingual/many2one/${INPUT_TYPE}-${SRC_LANGS}-${TGT_Langs}
DATASET_DIR=/home2/chu/fairseq_sign/sign/dataset/sp-10/256x256
TENSORBOARD_DIR=/home2/chu/fairseq_sign/sign/experiments/sp-10/tensorboard/multilingual/many2one/${INPUT_TYPE}-${SRC_LANGS}-${TGT_Langs}
rm -r $ST_SAVE_DIR
mkdir $ST_SAVE_DIR -p
rm -r $TENSORBOARD_DIR
mkdir $TENSORBOARD_DIR -p

# langs=ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN
langs=zh,uk,ru,bg,is,de,it,sv,lt,en
CUDA_VISIBLE_DEVICES=3 \
nohup python /home2/chu/fairseq_sign/sign/train.py  $DATASET_DIR \
    --batch-size-valid 1 \
    --ddp-backend legacy_ddp \
    --distributed-world-size 1 \
    --use-src-lang-id \
    --config-yaml $CONFIG_YAML \
    --langs $langs \
    --input-type $INPUT_TYPE \
    --share-decoder-input-output-embed \
    --encoder-embed-dim 512 \
    --encoder-ffn-embed-dim 2048 \
    --sign-encoder-layers 3 \
    --text-encoder-layers 3 \
    --decoder-layers 3 \
    --layernorm-embedding \
    --encoder-normalize-before --decoder-normalize-before \
    --tensorboard-logdir $TENSORBOARD_DIR \
    --task sign_text_joint_to_text \
    --src-langs $SRC_LANGS \
    --tgt-langs $TGT_Langs \
    --train-subset train_joint --valid-subset dev_joint \
    --save-dir  $ST_SAVE_DIR \
    --criterion label_smoothed_cross_entropy_with_hidden_mapping_sign \
    --arch sign2text_jmt_transformer \
    --batch-size 16 --max-epoch 100 \
    --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
    --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9,0.98)' \
    --lr-scheduler inverse_sqrt --lr 1e-4 --warmup-init-lr 1e-7 --warmup-updates 5000 \
    --weight-decay 0.0001 \
    --label-smoothing 0.2 \
    --dropout 0.2 \
    --attention-dropout 0.2 \
    --activation-dropout 0.2 \
    --subsampling-norm none \
    --subsampling-activation glu \
    --activation-fn relu \
    --no-epoch-checkpoints \
    --keep-best-checkpoints 3 \
    --num-workers 10 \
    --update-freq 1 \
    --seed 222 \
    --report-accuracy \
    --subsampling-type layernorm \
    --eval-mt-bleu --eval-bleu-detok space --eval-bleu-remove-bpe sentencepiece \
    --best-checkpoint-metric mt_bleu \
    --scoring sacrebleu \
    --maximize-best-checkpoint-metric \
    --validate-after-updates 200000 \
    --hidden-mapping-loss-type kl \
    --similarity-loss-type l2 \
    --hidden-mapping-task-flag \
    > ${ST_SAVE_DIR}/${INPUT_TYPE}-${SRC_LANGS}-${TGT_Langs}.log 2>&1 &
    # --encoder-learned-pos \
    # --decoder-learned-pos \
#  --eval-bleu-print-samples \





