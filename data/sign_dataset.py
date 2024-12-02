#!/usr/bin/env/python

import csv
import io
import logging
import os.path as op
import re
from typing import Dict, List, Optional, Tuple, NamedTuple
from pathlib import Path
import pickle
import gzip
import numpy as np
import torch
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    FairseqDataset,
    ResamplingDataset,
    data_utils as fairseq_data_utils,
)
from data.data_cfg import S2TDataConfig
import pdb
logger = logging.getLogger(__name__)

class SignToTextDatasetItem(NamedTuple):
    index: int
    source_sign: torch.Tensor
    source_text: torch.Tensor
    target: Optional[torch.Tensor]

class SignToTextDataset(FairseqDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        data_cfg: S2TDataConfig,
        n_frames: List[int],
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

    ):
        self.split, self.is_train_split = split, is_train_split
        self.data_cfg = data_cfg
        self.signs = signs
        self.n_frames = n_frames
        self.n_samples = len(signs)
        assert len(n_frames) == self.n_samples > 0
        assert src_texts is None or len(src_texts) == self.n_samples
        assert tgt_texts is None or len(tgt_texts) == self.n_samples
        assert signers is None or len(signers) == self.n_samples
        assert src_langs is None or len(src_langs) == self.n_samples
        assert tgt_langs is None or len(tgt_langs) == self.n_samples
        assert ids is None or len(ids) == self.n_samples
        assert (tgt_dict is None and tgt_texts is None) or (
                tgt_dict is not None and tgt_texts is not None
        )
        self.src_texts, self.tgt_texts = src_texts, tgt_texts
        self.src_langs, self.tgt_langs = src_langs, tgt_langs
        self.src_dict, self.tgt_dict = src_dict, tgt_dict
        self.check_tgt_lang_tag()
        self.ids = ids
        self.shuffle = data_cfg.shuffle if is_train_split else False

        self.pre_tokenizer = pre_tokenizer
        self.bpe_tokenizer = bpe_tokenizer
        self.joint_bpe_tokenizer = joint_bpe_tokenizer
        self.src_bpe_tokenizer = src_bpe_tokenizer
        self.tgt_bpe_tokenizer = tgt_bpe_tokenizer



    def __repr__(self):
        return (
                self.__class__.__name__
                + f'(split="{self.split}", n_samples={self.n_samples}, '
                  f"prepend_tgt_lang_tag={self.data_cfg.prepend_tgt_lang_tag}, "
                  f"shuffle={self.shuffle})"
        )

    @classmethod
    def is_lang_tag(cls, token):
        pattern = cls.LANG_TAG_TEMPLATE.replace("{}", "(.*)")
        return re.match(pattern, token)

    def check_tgt_lang_tag(self):
        if self.data_cfg.prepend_tgt_lang_tag:
            assert self.tgt_langs is not None and self.tgt_dict is not None
            tgt_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.tgt_langs)
            ]
            assert all(t in self.tgt_dict for t in tgt_lang_tags)

    def tokenize_text(self, text: str, is_src=False):
        # if self.pre_tokenizer is not None:
        #     text = self.pre_tokenizer.encode(text)
        # if self.bpe_tokenizer is not None:
        #     text = self.bpe_tokenizer.encode(text)
        # return text
        # TODO 需要针对不同的tokenizer在做适配
        if self.pre_tokenizer is not None:
            text = self.pre_tokenizer.encode(text)
        if self.joint_bpe_tokenizer is not None:
            text = self.joint_bpe_tokenizer.encode(text)
        elif self.src_bpe_tokenizer is not None and self.tgt_bpe_tokenizer is not None:
            text = self.tgt_bpe_tokenizer.encode(text) if not is_src else self.src_bpe_tokenizer.encode(text)
        elif self.bpe_tokenizer is not None:
            text = self.bpe_tokenizer.encode(text)
        return text

    def __getitem__(
            self, index: int
    ) -> Tuple[int, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        source_sign = self.signs[index]

        target = None
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
        source_text = None
        if self.src_texts is not None:
            tokenized = self.tokenize_text(self.src_texts[index], is_src=True)
            # 如果没有源端词典，则都是用tgt_dict，此时它为共享词典
            encode_dict = self.src_dict if self.src_dict is not None else self.tgt_dict
            source_text = encode_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=True
            ).long()
        return SignToTextDatasetItem(
            index=index, source_sign=source_sign, source_text=source_text, target=target
        )


    def __len__(self):
        return self.n_samples
    
    def collate_frames(
        self, frames: List[torch.Tensor], is_sign_input: bool = False
    ) -> torch.Tensor:
        """
        Convert a list of 2D frames into a padded 3D tensor
        Args:
            frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
                length of i-th frame and f_dim is static dimension of features
        Returns:
            3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
        """
        max_len = max(frame.size(0) for frame in frames)
        if is_sign_input:
            out = frames[0].new_zeros((len(frames), max_len))
        else:
            out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
        for i, v in enumerate(frames):
            out[i, : v.size(0)] = v
        return out
    def collater(self, samples: List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]) -> Dict:
        if len(samples) == 0:
            return {}
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)
        frames = self.collate_frames([x.source_sign for x in samples])

        # sort samples by descending number of frames
        n_frames = torch.tensor([x.source_sign.size(0) for x in samples], dtype=torch.long)
        n_frames, order = n_frames.sort(descending=True)
        indices = indices.index_select(0, order)
        frames = frames.index_select(0, order)

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
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(x.target.size(0) for x in samples)

        src_texts = None
        if self.src_texts is not None:
            encode_dict = self.src_dict if self.src_dict is not None else self.tgt_dict
            src_texts = fairseq_data_utils.collate_tokens(
                [x.source_text for x in samples],
                encode_dict.pad(),
                encode_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
        src_texts = src_texts.index_select(0, order)
        src_lengths = torch.tensor(
            [x.source_text.size(0) for x in samples], dtype=torch.long
        ).index_select(0, order)


        out = {
            "id": indices,
            "net_input": {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
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
        }
        return out

    def num_tokens(self, index):
        return self.n_frames[index]

    def size(self, index):
        t_len = 0
        if self.tgt_texts is not None:
            tokenized = self.tokenize_text(self.tgt_texts[index])
            t_len = len(tokenized.split(" "))
        return self.n_frames[index], t_len

    @property
    def sizes(self):
        return np.array(self.n_frames)


    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        order.append([-n for n in self.n_frames])
        return np.lexsort(order)

    @property
    def can_reuse_epoch_itr_across_epochs(self):
        return True

class SignToTextDatasetCreator(object):
    # mandatory columns
    KEY_ID, KEY_SIGN, KEY_N_FRAMES = "id", "sign", "n_frames"
    KEY_TGT_TEXT = "tgt_text"
    # optional columns
    KEY_SIGNER, KEY_SRC_TEXT = "signer", "src_text"
    KEY_SRC_LANG, KEY_TGT_LANG = "src_lang", "tgt_lang"
    # default values
    DEFAULT_SIGNER = DEFAULT_SRC_TEXT = DEFAULT_LANG = ""

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
    ) -> SignToTextDataset:
        signs, n_frames, src_texts, tgt_texts, ids = [], [], [], [], []
        signers, src_langs, tgt_langs = [], [], []
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
        return SignToTextDataset(
            split_name,
            is_train_split,
            data_cfg,
            n_frames,
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
    ):
        pickle_path = Path(root) / f"{split_name}.pkl"
        if not pickle_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {pickle_path}")
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
                            "id" : seq_id,
                            "sign" : s[src_lang]['video_feature'],
                            "n_frames": len(s[src_lang]['video_feature']),
                            "tgt_text" : s[tgt_lang]['text'],
                            "src_text" : s[src_lang]['text'],
                            "src_lang" : src_lang,
                            "tgt_lang" : tgt_lang,
                        }
                        samples.append(sample)
        elif data_cfg.data_name == "csl":
            with gzip.open(pickle_path, "rb") as f:
                tmp = pickle.load(f)
            for s in tmp:
                sample = {
                    "id" : s['name'],
                    "sign" : s['sign'],
                    "n_frames": s["length"],
                    "tgt_text" : s['text'],
                    "src_text" : s['gloss'],
                    "src_lang" : "zh",
                    "tgt_lang" : "zh",
                }
                samples.append(sample)            
        if len(samples) == 0:
            raise ValueError(f"Empty manifest: {pickle_path}")
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
            splits: str,
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
    ) -> SignToTextDataset:
        samples = []
        _splits = splits.split(",")
        datasets = [
            cls._load_dataset_from_pickle(
                name,
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
            )
            for name in _splits
        ]

        if is_train_split and len(_splits) > 1 and data_cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls._get_size_ratios(
                _splits, [len(s) for s in samples], alpha=data_cfg.sampling_alpha
            )
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for d, r in zip(datasets, size_ratios)
            ]
        return ConcatDataset(datasets)
