import argparse
import os
from typing import List
import pdb
import torch

from fairseq.data import Dictionary


def load_dict(langs: List[str], path: str) -> Dictionary:
    d = Dictionary.load(path)
    for l in langs:
        d.add_symbol(f"[{l}]")
    d.add_symbol("<mask>")
    return d


def main() -> None:
    parser = argparse.ArgumentParser(description="Trims pre-trained mBART model for fine-tuning.")
    parser.add_argument("--pre-train-dir", type=str, default="/home2/chu/fairseq_sign/sign/pretrained_models/mbart.cc25.v2", help="The pre-trained mBART model directory.")
    parser.add_argument("--ft-dict", type=str, default="/home2/chu/fairseq_sign/sign/dataset/mt/en-de/add_sp_10/short_add_sp_10_index/dict.de.txt", help="The fine-tuning model dictionary.")
    parser.add_argument("--langs", type=str, default="ar_AR,cs_CZ,de_DE,en_XX,es_XX,et_EE,fi_FI,fr_XX,gu_IN,hi_IN,it_IT,ja_XX,kk_KZ,ko_KR,lt_LT,lv_LV,my_MM,ne_NP,nl_XX,ro_RO,ru_RU,si_LK,tr_TR,vi_VN,zh_CN",help="The pre-trained model languages.")
    parser.add_argument("--output", type=str, default="/home2/chu/fairseq_sign/sign/pretrained_models/mbart.cc25.v2/europarl-short-en-de-prune-mbart.pt", help="The trimmed mBART model.")
    args = parser.parse_args()

    langs = args.langs.split(",")

    pre_dict = load_dict(langs, os.path.join(args.pre_train_dir, "dict.txt"))
    ft_dict = load_dict(langs, args.ft_dict)
    data = torch.load(os.path.join(args.pre_train_dir, "model.pt"))
    model = data["model"]

    mapping: List[int] = []
    for i in range(len(ft_dict)):
        word = ft_dict[i]
        mapping.append(pre_dict.index(word))

    for name in ["encoder.embed_tokens.weight", "decoder.embed_tokens.weight"]:
        pre_tensor: torch.Tensor = model[name]
        ft_tensor = torch.zeros(
            [len(ft_dict), 1024], dtype=pre_tensor.dtype, layout=pre_tensor.layout, device=pre_tensor.device,
        )
        for ft_i, pre_i in enumerate(mapping):
            ft_tensor[ft_i] = pre_tensor[pre_i]
        model[name] = ft_tensor

    torch.save(data, args.output)


if __name__ == "__main__":
    main()