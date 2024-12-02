for SRC_LANGS in zh uk ru bg is de it sv lt en
do
    for TGT_LANGS in zh uk ru bg is de it sv lt en
    do
        CUDA_VISIBLE_DEVICES=1 python /home2/chu/fairseq_sign/sign/generate.py /home2/chu/fairseq_sign/sign/dataset/sp-10/256x256 \
            --eval-mt-bleu \
            --use-src-lang-id \
            --src-langs ${SRC_LANGS} \
            --tgt-langs ${TGT_LANGS} \
            --config-yaml config-wiki-all.yaml \
            --task sign_text_joint_to_text \
            --post-process \
            --gen-subset test_joint \
            --remove-bpe \
            --batch-size 1 \
            --beam 5 \
            --lenpen 1 \
            --langs zh,uk,ru,bg,is,de,it,sv,lt,en \
            --path  /home2/chu/fairseq_sign/sign/experiments/sp-10/multilingual/many2one/text-zh,uk,ru,bg,is,de,it,sv,lt,en-zh,uk,ru,bg,is,de,it,sv,lt,en/checkpoint_best.pt \
            --scoring sacrebleu >  /home2/chu/fairseq_sign/sign/experiments/sp-10/multilingual/many2one/text-zh,uk,ru,bg,is,de,it,sv,lt,en-zh,uk,ru,bg,is,de,it,sv,lt,en/mt-test/${SRC_LANGS}-${TGT_LANGS}.log 2>&1
    done
done

# /home2/chu/fairseq_sign/sign/experiments/sp-10/bilingual/sign2en/sign-${SRC_LANGS}-en/checkpoint_best.pt
#