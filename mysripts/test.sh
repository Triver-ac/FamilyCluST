
for lang in ru #en sv it zh uk ru bg is de lt # ru bg it is #en sv it zh uk ru bg is de it sv lt
do
    for bs in 8
    do
        for beam in 5
        do
            for lp in 1
            do
                CUDA_VISIBLE_DEVICES=1 python /home2/chu/fairseq_sign/sign/generate.py /home2/chu/fairseq_sign/sign/dataset/sp-10/256x256 \
                    --use-src-lang-id \
                    --src-langs $lang \
                    --tgt-langs en \
                    --config-yaml config-wiki-all.yaml \
                    --task sign_text_joint_to_text \
                    --post-process \
                    --gen-subset dev_joint \
                    --remove-bpe \
                    --batch-size ${bs} \
                    --beam ${beam} \
                    --lenpen ${lp} \
                    --langs zh,uk,ru,bg,is,de,it,sv,lt,en \
                    --path /home2/chu/fairseq_sign/sign/experiments/sp-10/multilingual/many2one/sign-is,en,zh,it-en/checkpoint_best.pt \
                    --scoring sacrebleu
            done
        done
    done
done
# > ${lang}-en.log 2>&1 &
# CUDA_VISIBLE_DEVICES=3 python /home2/chu/fairseq_sign/sign/generate.py /home2/chu/fairseq_sign/sign/dataset/sp-10/256x256 \
#     --eval-mt-bleu \
#     --use-src-lang-id \
#     --src-langs zh,uk,ru,bg,is,de,it,sv,lt,en \
#     --tgt-langs en \
#     --config-yaml config-wiki-all.yaml \
#     --task sign_text_joint_to_text \
#     --post-process \
#     --gen-subset train_joint \
#     --remove-bpe \
#     --batch-size 1 \
#     --beam 5 \
#     --lenpen 1 \
#     --langs zh,uk,ru,bg,is,de,it,sv,lt,en \
#     --path /home2/chu/fairseq_sign/sign/experiments/sp-10/multilingual/256x256-multi-zh,uk,ru,bg,is,de,it,sv,lt,en-en/checkpoint_best.pt \
#     --encoder-states-save-path /home2/chu/fairseq_sign/sign/experiments/encoder-states-save/256x256-sign-zh,uk,ru,bg,is,de,it,sv,lt,en-en-train-text.pkl \
#     --scoring sacrebleu 

