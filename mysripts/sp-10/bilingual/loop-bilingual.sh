# for src_lang in bg
# do
#     for input_type in multi text sign
#     do
#         if [ "$input_type" = "multi" ] || [ "$input_type" = "sign" ]; then
#             metric="bleu"
#         else
#             metric="mt_bleu"
#         fi
#         bash bilingual.sh "$input_type" "$src_lang" "$metric" "1"
#     done
# done
# zh uk ru bg is de it sv lt en
# 1358636

for src_lang in is de
do
    for input_type in sign multi
    do
        if [ "$input_type" = "multi" ] || [ "$input_type" = "sign" ]; then
            metric="bleu"
        else
            metric="mt_bleu"
        fi
        bash bilingual.sh "$input_type" "$src_lang" "$metric" "1"
    done
done