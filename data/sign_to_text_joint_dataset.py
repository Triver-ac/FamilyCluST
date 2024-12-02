# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional,Tuple

import torch
import pickle
import gzip
import numpy as np
from fairseq.data import ConcatDataset, Dictionary, ResamplingDataset
from fairseq.data import data_utils as fairseq_data_utils
from .data_cfg import (
    S2TJointDataConfig,
    S2TDataConfig,
)
from .sign_dataset import (
    SignToTextDataset,
    SignToTextDatasetCreator,
)
import pdb
logger = logging.getLogger(__name__)



class SignToTextJointDatasetItem(NamedTuple):
    index: int
    source_sign: torch.Tensor
    source_text: Optional[torch.Tensor] = None
    target: Optional[torch.Tensor] = None
    tgt_lang_tag: Optional[int] = None
    src_lang_tag: Optional[int] = None



class SignToTextJointDataset(SignToTextDataset):
    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TDataConfig,
        n_frames: List[int],
        src_texts_lens: List[int],
        tgt_texts_lens: List[int],
        signs: Optional[List[torch.Tensor]] = None,
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        signers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        src_dict: Optional[Dictionary] = None,
        src_bpe_tokenizer=None,
        tgt_bpe_tokenizer=None,
        joint_bpe_tokenizer=None,
        append_eos: Optional[bool] = True,
        use_src_lang_id: Optional[bool] = False,
    ):
        super().__init__(
            split,
            is_train_split,
            data_cfg,
            n_frames,
            signs,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            signers=signers,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_dict=src_dict,
            src_bpe_tokenizer=src_bpe_tokenizer,
            tgt_bpe_tokenizer=tgt_bpe_tokenizer,
            joint_bpe_tokenizer=joint_bpe_tokenizer,
        )
        self.src_texts_lens = src_texts_lens
        self.tgt_texts_lens = tgt_texts_lens
        self.src_dict = src_dict
        self.src_bpe_tokenizer = src_bpe_tokenizer
        self.use_src_lang_id = use_src_lang_id
        self.append_eos = append_eos
        self.cfg = data_cfg
        self.src_lang = src_langs[0]
        self.tgt_lang = tgt_langs[0]
        
        logger.info(self.__repr__())

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len = len(tokenized.split(" "))
        s_len = 0
        if self.src_texts is not None:
            tokenized = self.tokenize_text(self.src_texts[index],True)
            s_len = len(tokenized.split(" "))
        return self.n_frames[index], s_len, t_len

    # @property
    # def sizes(self):
    #     return np.array(self.n_frames)

    # @property
    # def src_sizes(self):
    #     return np.array(self.src_texts_lens)
    # @property
    # def tgt_sizes(self):
    #     return np.array(self.tgt_texts_lens)
    # def ordered_indices(self):
    #     if self.shuffle:
    #         order = [np.random.permutation(len(self))]
    #     else:
    #         order = [np.arange(len(self))]
    #     # first by descending order of # of frames then by original/random order
    #     order.append([-n for n in self.n_frames])
    #     return np.lexsort(order)

    # def num_tokens(self, index):
    #     """Return the number of tokens in a sample. This value is used to
    #     enforce ``--max-tokens`` during batching."""
    #     return max(
    #         self.src_sizes[index],
    #         self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
    #     )
    # def num_tokens_vec(self, indices):
    #     """Return the number of tokens for a set of positions defined by indices.
    #     This value is used to enforce ``--max-tokens`` during batching."""
    #     sizes = self.src_sizes[indices]
    #     if self.tgt_sizes is not None:
    #         sizes = np.maximum(sizes, self.tgt_sizes[indices])
    #     return sizes

    @classmethod
    def get_lang_tag_idx(cls, lang: str, dictionary: Dictionary):
        lang_tag_idx = dictionary.index(cls.LANG_TAG_TEMPLATE.format(lang))
        assert lang_tag_idx != dictionary.unk()
        return lang_tag_idx

    def __getitem__(self, index: int) -> SignToTextJointDatasetItem:
        s2t_dataset_item = super().__getitem__(index)
        src_lang_tag = None
        tgt_lang_tag = None
        add_lang_id_src_text = None
        add_lang_id_target_text = None
        if self.use_src_lang_id:
            src_lang_tag = self.get_lang_tag_idx(self.src_langs[index], self.src_dict)
            add_lang_id_src_text = torch.cat([s2t_dataset_item.source_text, s2t_dataset_item.source_text.new([src_lang_tag])])


        # prepend_tgt_lang_tag: modify prev_output_tokens instead
        tgt_lang_tag = self.get_lang_tag_idx(self.tgt_langs[index], self.tgt_dict)
        if self.cfg.prepend_tgt_lang_tag:
            add_lang_id_target_text = torch.cat([s2t_dataset_item.target, s2t_dataset_item.target.new([tgt_lang_tag])])
    
        return SignToTextJointDatasetItem(
            index=index,
            source_sign=s2t_dataset_item.source_sign,
            source_text=add_lang_id_src_text if self.use_src_lang_id else s2t_dataset_item.source_text,
            target=add_lang_id_target_text if self.cfg.prepend_tgt_lang_tag else s2t_dataset_item.target,
            tgt_lang_tag=tgt_lang_tag,
            src_lang_tag=src_lang_tag,
        )

    def __len__(self):
        return self.n_samples

    def __repr__(self):
        return (
                self.__class__.__name__
                + f'(src_lang="{self.src_lang}", tgt_lang={self.tgt_lang}, '
                  f'(split="{self.split}", n_samples={self.n_samples}, '
                  f"prepend_tgt_lang_tag={self.data_cfg.prepend_tgt_lang_tag}, "
                  f"shuffle={self.shuffle})"
        )
    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = self.collate_frames([x.source_sign for x in samples])
        n_frames = torch.tensor([x.source_sign.size(0) for x in samples], dtype=torch.long)

        # TODO, 需要完善代码
        # sort samples by descending number of frames
        # 用手语的长度排序
        n_frames, order = n_frames.sort(descending=True)

        src_texts = None
        if self.src_texts is not None:
            encode_dict = self.src_dict if self.src_dict is not None else self.tgt_dict
            src_texts = fairseq_data_utils.collate_tokens(
                [x.source_text for x in samples],
                encode_dict.pad(),
                encode_dict.eos(),
                left_pad=True,
                move_eos_to_beginning=False,
            )
        src_texts = src_texts.index_select(0, order)
        src_lengths = torch.tensor(
            [x.source_text.size(0) for x in samples], dtype=torch.long
        ).index_select(0, order)

        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

        # 用text的长度排序
        # src_texts = None
        # if self.src_texts is not None:
        #     encode_dict = self.src_dict if self.src_dict is not None else self.tgt_dict
        #     src_texts = fairseq_data_utils.collate_tokens(
        #         [x.source_text for x in samples],
        #         encode_dict.pad(),
        #         encode_dict.eos(),
        #         left_pad=True,
        #         move_eos_to_beginning=False,
        #     )
        # src_lengths = torch.tensor(
        #     [x.source_text.size(0) for x in samples], dtype=torch.long
        # )
        # src_lengths, order = src_lengths.sort(descending=True)
        # src_texts = src_texts.index_select(0, order)
        # indices = indices.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                None, #self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(x.target.size(0) for x in samples)

        tgt_langs = torch.tensor([x.tgt_lang_tag for x in samples], dtype=torch.long)
        tgt_langs = tgt_langs.index_select(0, order)

        src_langs = torch.tensor([x.src_lang_tag for x in samples], dtype=torch.long)
        src_langs = src_langs.index_select(0, order)
    
        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "src_langs": src_langs,
            },
            "net_text_input": {
                "src_tokens": src_texts,
                "src_lengths": src_lengths,
                "prev_output_tokens": prev_output_tokens,
            },
            "target": target,
            "target_lengths": target_lengths,
            "ntokens": ntokens,
            "nsentences": len(samples),
            "order": order,
            "tgt_langs": tgt_langs,
            "src_langs": src_langs,
        }
        return out

class SignToTextJointDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SIGN, KEY_N_FRAMES = "id", "sign", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SIGNER, KEY_SRC_TEXT = "signer", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    KEY_SRC_TEXT_LEN, KEY_TGT_TEXT_LEN = "src_text_mbart_index_length","tgt_text_mbart_index_length"
    # default values
    DEFAULT_SIGNER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""
    DEFAULT_TEXT_LEN = 0

    @classmethod
    def _from_list(
            cls,
            split_name: str,
            is_train_split,
            samples: List[Dict],
            data_cfg: S2TDataConfig,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_dict,
            src_bpe_tokenizer,
            tgt_bpe_tokenizer,
            joint_bpe_tokenizer,
            append_eos,
            use_src_lang_id,
    ) -> SignToTextJointDataset:
        signs, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        signers, src_langs, tgt_langs = [], [], []
        src_texts_lens, tgt_texts_lens = [],[]
        s = samples

        ids.extend([ss[cls.KEY_ID] for ss in s])
        signs = [ss[cls.KEY_SIGN] for ss in s]
        n_frames.extend([int(ss[cls.KEY_N_FRAMES]) for ss in s])
        tgt_texts.extend([ss[cls.KEY_TGT_TEXT] for ss in s])
        src_texts.extend(
            [ss.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for ss in s]
        )
        signers.extend([ss.get(cls.KEY_SIGNER, cls.DEFAULT_SIGNER) for ss in s])
        src_langs.extend([ss.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for ss in s])
        tgt_langs.extend([ss.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for ss in s])
        src_texts_lens.extend([ss.get(cls.KEY_SRC_TEXT_LEN, cls.DEFAULT_TEXT_LEN) for ss in s])
        tgt_texts_lens.extend([ss.get(cls.KEY_TGT_TEXT_LEN, cls.DEFAULT_TEXT_LEN) for ss in s])
        return SignToTextJointDataset(
            split_name,
            is_train_split,
            data_cfg,
            n_frames,
            src_texts_lens,
            tgt_texts_lens,
            signs,
            src_texts,
            tgt_texts,
            signers,
            src_langs,
            tgt_langs,
            ids,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_dict,
            src_bpe_tokenizer,
            tgt_bpe_tokenizer,
            joint_bpe_tokenizer,
            append_eos,
            use_src_lang_id,
        )



    @classmethod
    def _load_dataset_from_pickle(
        cls,
        split_name,
        root,
        src_langs,
        tgt_langs,
        is_train_split,
        data_cfg,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_dict,
        src_bpe_tokenizer,
        tgt_bpe_tokenizer,
        joint_bpe_tokenizer,
        append_eos,
        use_src_lang_id,
    ):
        pickle_path = Path(root) / f"{split_name}.pkl"
        if not pickle_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {pickle_path}")
        
        lang_samples = {}
        for src_lang in src_langs:
            for tgt_lang in tgt_langs:
                lang_samples[f"{src_lang}-{tgt_lang}"] = []
        samples = []
        if data_cfg.data_name == "sp-10":
            with open(pickle_path, "rb") as f:
                tmp = pickle.load(f)
            for s in tmp:
                seq_id = str(s["id"])
                if len(s.items()) != 11: #test集中有的seq中个别语言有缺失
                    continue
                for src_lang in src_langs:
                    for tgt_lang in tgt_langs:
                        sample = {
                            "id" : seq_id+src_lang,
                            "sign" : s[src_lang]['video_feature'],
                            "n_frames": len(s[src_lang]['video_feature']),
                            "tgt_text" : s[tgt_lang]['text'],
                            "src_text" : s[src_lang]['text'],
                            "src_text_mbart_index_length":  s[src_lang]['mbart_index_length'] if "mbart_index_length" in s[src_lang] else 0,
                            "tgt_text_mbart_index_length":  s[tgt_lang]['mbart_index_length'] if "mbart_index_length" in s[tgt_lang] else 0,
                            "src_lang" : src_lang,
                            "tgt_lang" : tgt_lang,
                        }
                        lang_samples[f"{src_lang}-{tgt_lang}"].append(sample)    
                        samples.append(sample)
        if data_cfg.data_name == "sp-10+":
            with open(pickle_path, "rb") as f:
                tmp = pickle.load(f)
            for s in tmp:
                seq_id = str(s["id"])
                if len(s.items()) != 11: #test集中有的seq中个别语言有缺失
                    continue
                for src_lang in src_langs:
                    for tgt_lang in tgt_langs:
                        sample = {
                            "id" : seq_id+src_lang,
                            "sign" : s[src_lang]['video_feature'],
                            "n_frames": len(s[src_lang]['video_feature']),
                            "tgt_text" : s[tgt_lang]['text'],
                            "src_text" : s[src_lang]['text'],
                            "src_text_mbart_index_length":  s[src_lang]['mbart_index_length'] if "mbart_index_length" in s[src_lang] else 0,
                            "tgt_text_mbart_index_length":  s[tgt_lang]['mbart_index_length'] if "mbart_index_length" in s[tgt_lang] else 0,
                            "src_lang" : src_lang,
                            "tgt_lang" : tgt_lang,
                        }
                        lang_samples[f"{src_lang}-{tgt_lang}"].append(sample)    
                        samples.append(sample)
            if "train" in split_name:
                with open(data_cfg.extra_sign_path, "rb") as f:
                    tmp = pickle.load(f)
                for s in tmp:
                    seq_id = s["name"]
                    if "en_text" not in s:
                        continue
                    sample = {
                        "id" : seq_id,
                        "sign" : s['sign'],
                        "n_frames": len(s['sign']),
                        "tgt_text" : s['en_text'],
                        "src_text" : s['text'],
                        "src_text_mbart_index_length":  0,
                        "tgt_text_mbart_index_length":  0,
                        "src_lang" : "de",
                        "tgt_lang" : "en",
                    }
                    lang_samples[f"de-en"].append(sample)    
                    samples.append(sample)

        if data_cfg.data_name == "sp-10" and len(samples) == 0:
            raise ValueError(f"Empty manifest: {pickle_path}")
        if data_cfg.combine_lang_datasets:
            datasets = []
            for key in lang_samples:
                datasets.append(
                    cls._from_list(
                    split_name,
                    is_train_split,
                    lang_samples[key],
                    data_cfg,
                    tgt_dict,
                    pre_tokenizer,
                    bpe_tokenizer,
                    src_dict,
                    src_bpe_tokenizer,
                    tgt_bpe_tokenizer,
                    joint_bpe_tokenizer,
                    append_eos,
                    use_src_lang_id,
                )
                )
            return datasets
        else:
            return cls._from_list(
                    split_name,
                    is_train_split,
                    samples,
                    data_cfg,
                    tgt_dict,
                    pre_tokenizer,
                    bpe_tokenizer,
                    src_dict,
                    src_bpe_tokenizer,
                    tgt_bpe_tokenizer,
                    joint_bpe_tokenizer,
                    append_eos,
                    use_src_lang_id,
                )
    @classmethod
    def _get_size_ratios(cls, ids: List[str], sizes: List[int], alpha: float = 1.0):
        """Size ratios for temperature-based sampling
        (https://arxiv.org/abs/1907.05019)"""
        _sizes = np.array(sizes)
        prob = _sizes / _sizes.sum()
        smoothed_prob = prob ** alpha
        smoothed_prob = smoothed_prob / smoothed_prob.sum()
        size_ratio = (smoothed_prob * _sizes.sum()) / _sizes

        o_str = str({_i: f"{prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"original sampling probability: {o_str}")
        p_str = str({_i: f"{smoothed_prob[i]:.3f}" for i, _i in enumerate(ids)})
        logger.info(f"balanced sampling probability: {p_str}")
        sr_str = str({_id: f"{size_ratio[i]:.3f}" for i, _id in enumerate(ids)})
        logger.info(f"balanced sampling size ratio: {sr_str}")
        return size_ratio.tolist()
    
    @classmethod
    def from_pickle(
            cls,
            root: str,
            src_langs: List[str],
            tgt_langs: List[str],
            data_cfg: S2TDataConfig,
            split: str,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split: bool,
            epoch: int,
            seed: int,
            src_dict,
            src_bpe_tokenizer,
            tgt_bpe_tokenizer,
            joint_bpe_tokenizer,
            append_eos: bool,
            use_src_lang_id: bool,
    ) -> SignToTextJointDataset:
        datasets = cls._load_dataset_from_pickle(
                split,
                root,
                src_langs,
                tgt_langs,
                is_train_split,
                data_cfg,
                tgt_dict,
                pre_tokenizer,
                bpe_tokenizer,
                src_dict,
                src_bpe_tokenizer,
                tgt_bpe_tokenizer,
                joint_bpe_tokenizer,
                append_eos,
                use_src_lang_id,
            )

        # if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
        #     # temperature-based sampling
        #     size_ratios = cls._get_size_ratios(
        #         _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
        #     )
        #     datasets = [
        #         ResamplingDataset(
        #             d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
        #         )
        #         for d, r in zip(datasets, size_ratios)
        #     ]
        if len(datasets) == 1:
            return datasets[0]
        else:
            return ConcatDataset(datasets)
