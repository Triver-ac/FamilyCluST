# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import json
import os.path as op
import numpy as np
from argparse import Namespace
from pathlib import Path
from fairseq import metrics, utils
from fairseq.data import Dictionary, encoders
from fairseq.scoring.tokenizer import EvaluationTokenizer
from fairseq.tasks import LegacyFairseqTask, register_task
import torch

import sys
sys.path.append("..") 
from data import (
    S2TDataConfig,
    SignToTextDataset,
    SignToTextDatasetCreator,
)
from sequence_generator import SequenceGenerator
EVAL_BLEU_ORDER = 4
logger = logging.getLogger(__name__)

import pdb
@register_task("sign_to_text")
class SignToTextTask(LegacyFairseqTask):
    @staticmethod
    def add_args(parser):
        parser.add_argument("data", help="manifest root path")
        parser.add_argument(
            "--config-yaml",
            type=str,
            default="config.yaml",
            help="Configuration YAML filename (under manifest root)",
        )
        parser.add_argument(
            "--max-source-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the source sequence",
        )
        parser.add_argument(
            "--max-target-positions",
            default=1024,
            type=int,
            metavar="N",
            help="max number of tokens in the target sequence",
        )
        parser.add_argument(
            "--use-aligned-text",
            default=False,
            action="store_true",
            help="use aligned text for loss",
        )

        # options for reporting BLEU during validation
        parser.add_argument(
            "--eval-bleu",
            default=False,
            action="store_true",
            help="evaluation with BLEU scores",
        )
        parser.add_argument(
            "--eval-bleu-args",
            default="{}",
            type=str,
            help='generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string',
        )
        parser.add_argument(
            "--eval-bleu-detok",
            default="space",
            type=str,
            help="detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
                 "use 'space' to disable detokenization; see fairseq.data.encoders for other options",
        )

        parser.add_argument(
            "--eval-bleu-detok-args",
            default="{}",
            type=str,
            help="args for building the tokenizer, if needed, as JSON string",
        )
        parser.add_argument(
            "--eval-tokenized-bleu",
            default=False,
            action="store_true",
            help="compute tokenized BLEU instead of sacrebleu",
        )
        parser.add_argument(
            "--eval-bleu-remove-bpe",
            nargs="?",
            const="@@ ",
            default=None,
            help="remove BPE before computing BLEU",
        )
        parser.add_argument(
            "--eval-bleu-print-samples",
            default=False,
            action="store_true",
            help="print sample generations during validation",
        )
        parser.add_argument(
            "--src-langs",
            type=str,
            default="en",
            help="source sign language categories",
        )
        parser.add_argument(
            "--tgt-langs",
            type=str,
            default="en",
            help="target text language categories",
        )
        parser.add_argument(
            "--generate-task-type",
            default='st',
            choices=["st", "mt"],
            help="Choosing a task for inference."
        )

    def __init__(self, args, tgt_dict, src_dict=None):
        super().__init__(args)
        self.src_dict = src_dict
        self.tgt_dict = tgt_dict
        self.data_cfg = S2TDataConfig(Path(args.data) / args.config_yaml)

    @classmethod
    def setup_task(cls, args, **kwargs):
        data_cfg = S2TDataConfig(Path(op.join(args.data, args.config_yaml)))
        # load target dict
        dict_path = op.join(args.data, data_cfg.vocab_filename)
        if not op.isfile(dict_path):
            raise FileNotFoundError(f"Target dict not found: {dict_path}")
        tgt_dict = Dictionary.load(dict_path)
        logger.info(
            f"target dictionary size ({data_cfg.vocab_filename}): " f"{len(tgt_dict):,}"
        )
        # load source dict
        src_dict_path = op.join(args.data, data_cfg.src_vocab_filename)
        if not op.isfile(src_dict_path):
            raise FileNotFoundError(f"Source dict not found: {src_dict_path}")
        src_dict = Dictionary.load(src_dict_path)
        logger.info(
            f"source dictionary size ({data_cfg.src_vocab_filename}): " f"{len(src_dict):,}"
        )

        if getattr(args, "train_subset", None) is not None:
            if not all(s.startswith("train") for s in args.train_subset.split(",")):
                raise ValueError('Train splits should be named like "train*".')
        return cls(args, tgt_dict, src_dict)

    def build_criterion(self, args):
        from fairseq import criterions

        # if self.data_cfg.prepend_tgt_lang_tag and args.ignore_prefix_size != 1:
        #     raise ValueError(
        #         'Please set "--ignore-prefix-size 1" since '
        #         "target language ID token is prepended as BOS."
        #     )
        return criterions.build_criterion(args, self)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_bpe_tokenizer = self.build_src_bpe_tokenizer(self.args)
        assert src_bpe_tokenizer is not None
        tgt_bpe_tokenizer = self.build_tgt_bpe_tokenizer(self.args)
        assert tgt_bpe_tokenizer is not None
        joint_bpe_tokenizer = self.build_joint_bpe_tokenizer(self.args)
        self.datasets[split] = SignToTextDatasetCreator.from_pickle(
            self.args.data,
            self.args.src_langs.split(","),
            self.args.tgt_langs.split(","),
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            src_dict=self.src_dict,
            src_bpe_tokenizer=src_bpe_tokenizer,
            tgt_bpe_tokenizer=tgt_bpe_tokenizer,
            joint_bpe_tokenizer=joint_bpe_tokenizer,
        )

    @property
    def target_dictionary(self):
        return self.tgt_dict

    @property
    def source_dictionary(self):
        return self.src_dict

    def max_positions(self):
        return self.args.max_source_positions, self.args.max_target_positions

    def build_model(self, args):
        args.input_feat_per_channel = self.data_cfg.input_feat_per_channel
        args.input_channels = self.data_cfg.input_channels

        model = super(SignToTextTask, self).build_model(args)

        if self.args.eval_bleu or self.args.eval_mt_bleu:
            detok_args = json.loads(self.args.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.args.eval_bleu_detok, **detok_args)
            )
            self.args.eval_bleu_args = '{"beam": 4, "lenpen": 0.6}'
            gen_args = json.loads(self.args.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], args=Namespace(**gen_args),seq_gen_cls=SequenceGenerator
            )
        return model

    def build_tokenizer(self, args):
        logger.info(f"pre-tokenizer: {self.data_cfg.pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.pre_tokenizer))

    def build_bpe(self, args):
        logger.info(f"tokenizer: {self.data_cfg.bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.bpe_tokenizer))

    def build_src_bpe(self, args):
        logger.info(f"src tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def build_src_bpe_tokenizer(self, args):
        logger.info(f"src-bpe-tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def build_tgt_bpe_tokenizer(self, args):
        logger.info(f"tgt-bpe-tokenizer: {self.data_cfg.tgt_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.tgt_bpe_tokenizer))

    def build_joint_bpe_tokenizer(self, args):
        logger.info(f"joint-bpe-tokenizer: {self.data_cfg.joint_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.joint_bpe_tokenizer))
    # def get_interactive_tokens_and_lengths(self, lines, encode_fn):
    #     n_frames = [get_features_or_waveform(p).shape[0] for p in lines]
    #     return lines, n_frames
    #
    # def build_dataset_for_inference(self, src_tokens, src_lengths, **kwargs):
    #     return SpeechToTextDataset(
    #         "interactive", False, self.data_cfg, src_tokens, src_lengths
    #     )

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output["_bleu_sys_len"] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]

        return loss, sample_size, logging_output

    @staticmethod
    def forward_and_get_hidden_state_step(sample, model, modal="sign"):

        if modal == "text":
            sample["net_input"] = sample["net_text_input"]

        decoder_output, extra = model(
            src_tokens=sample['net_input']['src_tokens'],
            src_lengths=sample['net_input']['src_lengths'],
            prev_output_tokens=sample['net_input']['prev_output_tokens'],
            return_all_hiddens=False,
            features_only=True,
            modal=modal
        )
        return decoder_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:
            def sum_logs(key):
                import torch

                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))
            
            # TODO 这里要非常小心，可能会不返回bleu，导致计算错误
            if max(totals) >= 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    import inspect

                    try:
                        from sacrebleu.metrics import BLEU

                        comp_bleu = BLEU.compute_bleu
                    except ImportError:
                        # compatibility API for sacrebleu 1.x
                        import sacrebleu

                        comp_bleu = sacrebleu.compute_bleu

                    fn_sig = inspect.getfullargspec(comp_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = comp_bleu(
                        correct=[int(x) for x in meters["_bleu_counts"].sum],
                        total=[int(x) for x in meters["_bleu_totals"].sum],
                        sys_len=int(meters["_bleu_sys_len"].sum),
                        ref_len=int(meters["_bleu_ref_len"].sum),
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def _inference_with_bleu(self, generator, sample, model):
        import sacrebleu
        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"]))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                )
            )
        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

