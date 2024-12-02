# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
import os
from argparse import Namespace
from pathlib import Path
from fairseq import metrics, utils
import torch
from fairseq.data import (
    encoders,
    Dictionary,
    ResamplingDataset,
    TransformEosLangPairDataset,
    ConcatDataset,
)
from fairseq.data.iterators import GroupedEpochBatchIterator
from fairseq.data.audio.multi_modality_dataset import (
    MultiModalityDataset,
    ModalityDatasetItem,
)
import numpy as np

from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
# from fairseq.tasks.translation import load_langpair_dataset
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    LanguagePairDataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    data_utils,
    encoders,
    indexed_dataset,
)
import itertools
from .sign_to_text import (
    SignToTextTask
)
from data import (
    S2TJointDataConfig,
    SignToTextDataset,
    SignToTextDatasetCreator,
    SignToTextJointDatasetCreator,
)
import pdb
logger = logging.getLogger(__name__)
LANG_TAG_TEMPLATE = "<lang:{}>"
EVAL_BLEU_ORDER = 4
from sequence_generator import SequenceGenerator


def  load_langpair_dataset(
    data_path,
    split,
    src,
    src_dict,
    tgt,
    tgt_dict,
    combine,
    dataset_impl,
    upsample_primary,
    left_pad_source,
    left_pad_target,
    max_source_positions,
    max_target_positions,
    prepend_bos=False,
    load_alignments=False,
    truncate_source=False,
    append_source_id=False,
    num_buckets=0,
    shuffle=True,
    pad_to_multiple=1,
    prepend_bos_src=None,
):
    def split_exists(split, src, tgt, lang, data_path):
        filename = os.path.join(data_path, "{}.{}-{}.{}".format(split, src, tgt, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    src_datasets = []
    tgt_datasets = []
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else "")
        # infer langcode
        if split_exists(split_k, src, tgt, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, src, tgt))
        elif split_exists(split_k, tgt, src, src, data_path):
            prefix = os.path.join(data_path, "{}.{}-{}.".format(split_k, tgt, src))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError(
                    "Dataset not found: {} ({})".format(split, data_path)
                )

        src_dataset = data_utils.load_indexed_dataset(
            prefix + src, src_dict, dataset_impl
        )
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        tgt_dataset = data_utils.load_indexed_dataset(
            prefix + tgt, tgt_dict, dataset_impl
        )
        if tgt_dataset is not None:
            tgt_datasets.append(tgt_dataset)

        logger.info(
            "{} {} {}-{} {} examples".format(
                data_path, split_k, src, tgt, len(src_datasets[-1])
            )
        )

        if not combine:
            break

    assert len(src_datasets) == len(tgt_datasets) or len(tgt_datasets) == 0

    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
        tgt_dataset = tgt_datasets[0] if len(tgt_datasets) > 0 else None
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)
        if len(tgt_datasets) > 0:
            tgt_dataset = ConcatDataset(tgt_datasets, sample_ratios)
        else:
            tgt_dataset = None

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())
        if tgt_dataset is not None:
            tgt_dataset = PrependTokenDataset(tgt_dataset, tgt_dict.bos())
    elif prepend_bos_src is not None:
        logger.info(f"prepending src bos: {prepend_bos_src}")
        src_dataset = PrependTokenDataset(src_dataset, prepend_bos_src)

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(
            src_dataset, src_dict.index("<lang:{}>".format(src))
        )
        # pdb.set_trace()
        # if tgt_dataset is not None:
        #     tgt_dataset = AppendTokenDataset(
        #         tgt_dataset, tgt_dict.index("<lang:{}>".format(tgt))
        #     )
        # eos = tgt_dict.index("<lang:{}>".format(tgt))

    align_dataset = None
    if load_alignments:
        align_path = os.path.join(data_path, "{}.align.{}-{}".format(split, src, tgt))
        if indexed_dataset.dataset_exists(align_path, impl=dataset_impl):
            align_dataset = data_utils.load_indexed_dataset(
                align_path, None, dataset_impl
            )

    tgt_dataset_sizes = tgt_dataset.sizes if tgt_dataset is not None else None
    return LanguagePairDataset(
        src_dataset,
        src_dataset.sizes,
        src_dict,
        tgt_dataset,
        tgt_dataset_sizes,
        tgt_dict,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset,
        eos=eos,
        num_buckets=num_buckets,
        shuffle=shuffle,
        pad_to_multiple=pad_to_multiple,
    )


@register_task("sign_text_joint_to_text")
class SignTextJointToTextTask(SignToTextTask):
    """
    Task for joint training sign and text to text.
    """

    @classmethod
    def add_args(cls, parser):
        """Add task-specific arguments to the parser."""
        super(SignTextJointToTextTask, cls).add_args(parser)
        ###
        parser.add_argument(
            '--langs',  
            type=str, 
            metavar='LANG',            
            help='comma-separated list of monolingual language, '
                'for example, "en,de,fr". These should match the '
                'langs from pretraining (and be in the same order). '
                'You should always add all pretraining language idx '
                'during finetuning.')
        parser.add_argument(
            "--parallel-text-data",
            default="",
            help="path to parallel text data directory",
        )
        parser.add_argument(
            "--max-tokens-text",
            type=int,
            metavar="N",
            default=1024,
            help="maximum tokens for encoder text input ",
        )
        parser.add_argument(
            "--max-positions-text",
            type=int,
            metavar="N",
            default=1024,
            help="maximum tokens for per encoder text input ",
        )
        parser.add_argument(
            "--mt-text-langpairs",
            default=None,
            metavar="S",
            help='language pairs for text training, separated with ","',
        )
        parser.add_argument(
            "--sign-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for sign dataset with transcripts ",
        )
        parser.add_argument(
            "--text-sample-ratio",
            default=1,
            type=float,
            metavar="N",
            help="Multiple Ratio for text set ",
        )
        parser.add_argument(
            "--update-mix-data",
            action="store_true",
            help="use mixed data in one update when update-freq  > 1",
        )
        parser.add_argument(
            "--use-src-lang-id",
            action="store_true",
            help="attach src_lang_id to the src_tokens as eos"
        )
        parser.add_argument(
            "--eval-mt-bleu",
            default=False,
            action="store_true",
            help="evaluation with BLEU scores",
        )

    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, tgt_dict)
        self.src_dict = src_dict
        self.data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        assert self.tgt_dict.pad() == self.src_dict.pad()
        assert self.tgt_dict.eos() == self.src_dict.eos()
        # add langs token to mbart dict
        # 这里应该是也是深浅拷贝的问题
        if args.langs:
            self.langs = args.langs.split(",")
            for d in [src_dict, tgt_dict]:
                for l in self.langs:
                    d.add_symbol("<lang:{}>".format(l))
                d.add_symbol("<mask>")
        # whether to use lang_id
        self.use_src_lang_id = args.use_src_lang_id

        if self.data_cfg.prepend_tgt_lang_tag:
            self.infer_tgt_lang_ids = []
            for tgt_lang in self.args.tgt_langs.split(","):
                tgt_lang_tag = SignToTextDataset.LANG_TAG_TEMPLATE.format(tgt_lang)
                infer_tgt_lang_id = tgt_dict.index(tgt_lang_tag)
                assert infer_tgt_lang_id != tgt_dict.unk()
                self.infer_tgt_lang_ids.append(infer_tgt_lang_id)
        else:
            self.infer_tgt_lang_ids = None


    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        data_cfg = S2TJointDataConfig(Path(args.data) / args.config_yaml)
        tgt_dict_path = Path(args.data) / data_cfg.vocab_filename
        src_dict_path = Path(args.data) / data_cfg.src_vocab_filename
        if (not os.path.isfile(src_dict_path)) or (not os.path.isfile(tgt_dict_path)):
            raise FileNotFoundError("Dict not found: {}".format(args.data))
        src_dict = Dictionary.load(src_dict_path.as_posix())
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())

        print("| src dictionary: {} types".format(len(src_dict)))
        print("| tgt dictionary: {} types".format(len(tgt_dict)))

        if args.parallel_text_data != "" and args.mt_text_langpairs is None:
            raise Exception(
                "Could not infer language pair, please provide it explicitly"
            )

        return cls(args, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        is_train_split = "train" in split
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_bpe_tokenizer = self.build_src_bpe_tokenizer(self.args)
        assert src_bpe_tokenizer is not None
        tgt_bpe_tokenizer = self.build_tgt_bpe_tokenizer(self.args)
        assert tgt_bpe_tokenizer is not None
        joint_bpe_tokenizer = self.build_joint_bpe_tokenizer(self.args)
        sign_dataset = SignToTextJointDatasetCreator.from_pickle(
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
            append_eos=True,
            use_src_lang_id=self.use_src_lang_id,
        )
        text_dataset = None
        if self.args.parallel_text_data != "" and is_train_split:
            text_dataset = self.load_langpair_dataset(
                self.data_cfg.prepend_tgt_lang_tag, 1.0, epoch=epoch,
            )

        if text_dataset is not None:
            mdsets = [
                ModalityDatasetItem(
                    "sign",
                    sign_dataset,
                    (self.args.max_source_positions, self.args.max_target_positions),
                    self.args.max_tokens,
                    self.args.batch_size,
                ),
                ModalityDatasetItem(
                    "text",
                    text_dataset,
                    (self.args.max_positions_text, self.args.max_target_positions),
                    self.args.max_tokens_text if self.args.max_tokens_text is not None else self.args.max_tokens,
                    self.args.batch_size,
                ),
            ]
            sign_dataset = MultiModalityDataset(mdsets)
        self.datasets[split] = sign_dataset


    def load_langpair_dataset(
        self, prepend_tgt_lang_tag=False, sampling_alpha=1.0, epoch=0
    ):
        lang_pairs = []
        text_dataset = None
        split = "train"
        for lp in self.args.mt_text_langpairs.split(','):
            src, tgt = lp.split("-")
            text_dataset = load_langpair_dataset(
                self.args.parallel_text_data,
                split,
                src,
                self.src_dict,
                tgt,
                self.tgt_dict,
                combine=True,
                dataset_impl=None,
                upsample_primary=1,
                left_pad_source=False,
                left_pad_target=False,
                max_source_positions=self.args.max_positions_text,
                max_target_positions=self.args.max_target_positions,
                load_alignments=False,
                truncate_source=False,
                append_source_id=self.use_src_lang_id,
            )
            if prepend_tgt_lang_tag:
                # TODO
                text_dataset = TransformEosLangPairDataset(
                    text_dataset,
                    src_eos=self.src_dict.eos(),
                    tgt_bos=self.tgt_dict.eos(),  # 'prev_output_tokens' starts with eos
                    new_tgt_bos=self.tgt_dict.index(LANG_TAG_TEMPLATE.format(tgt)),
                )
            lang_pairs.append(text_dataset)
        if len(lang_pairs) > 1:
            if sampling_alpha != 1.0:
                size_ratios = SignToTextJointDatasetCreator.get_size_ratios(
                    self.args.mt_text_langpairs.split(","),
                    [len(s) for s in lang_pairs],
                    alpha=sampling_alpha,
                )
                lang_pairs = [
                    ResamplingDataset(d, size_ratio=r, epoch=epoch, replace=(r >= 1.0))
                    for d, r in zip(lang_pairs, size_ratios)
                ]
            return ConcatDataset(lang_pairs)
        return text_dataset

    def valid_step(self, sample, model, criterion):
        # 基类的valid_step
        loss, sample_size, logging_output = super(SignToTextTask, self).valid_step(sample, model, criterion)

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

        if self.args.eval_mt_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model, mt_generate=True)
            logging_output["_mt_bleu_sys_len"] = bleu.sys_len
            logging_output["_mt_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_mt_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_mt_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

        
    def _inference_with_bleu(self, generator, sample, model, mt_generate=None):
        import sacrebleu
        def decode(toks, escape_unk=False,extra_symbols_to_ignore=None):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                extra_symbols_to_ignore=extra_symbols_to_ignore,
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
        if self.data_cfg.prepend_tgt_lang_tag:
            # 注意，只能batch为1进行推理
            infer_lang = sample["tgt_langs"][0].item()
        else:
            infer_lang = self.tgt_dict.eos()
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=None, mt_generate=mt_generate, bos_token=infer_lang)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]["tokens"], extra_symbols_to_ignore=self.infer_tgt_lang_ids))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,  # don't count <unk> as matches to the hypo
                    extra_symbols_to_ignore=self.infer_tgt_lang_ids
                )
            )
        if self.args.eval_bleu_print_samples:
            if mt_generate:
                logger.info("example mt hypothesis: " + hyps[0])
                logger.info("example mt references: " + refs[0])
            else:
                logger.info("example s2t hypothesis: " + hyps[0])
                logger.info("example s2t references: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            return sacrebleu.corpus_bleu(hyps, [refs])

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None, mt_generate=None, bos_token=None
    ):
        if mt_generate:
            sample['net_input'] = sample['net_text_input']
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=bos_token,
                mt_generate=mt_generate,
            )

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):
        if not isinstance(dataset, MultiModalityDataset):
            return super(SignToTextTask, self).get_batch_iterator(
                dataset,
                max_tokens,
                max_sentences,
                max_positions,
                ignore_invalid_inputs,
                required_batch_size_multiple,
                seed,
                num_shards,
                shard_id,
                num_workers,
                epoch,
                data_buffer_size,
                disable_iterator_cache,
                skip_remainder_batch=skip_remainder_batch,
                update_epoch_batch_itr=update_epoch_batch_itr,
            )

        mult_ratio = [self.args.sign_sample_ratio, self.args.text_sample_ratio]
        assert len(dataset.datasets) == 2

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        batch_samplers = dataset.get_batch_samplers(
            mult_ratio, required_batch_size_multiple, seed
        )

        # return a reusable, sharded iterator
        epoch_iter = GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            mult_rate=1 if self.args.update_mix_data else max(self.args.update_freq),
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        if self.args.eval_mt_bleu:
            def sum_logs(key):
                import torch
                result = sum(log.get(key, 0) for log in logging_outputs)
                if torch.is_tensor(result):
                    result = result.cpu()
                return result

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_mt_bleu_counts_" + str(i)))
                totals.append(sum_logs("_mt_bleu_totals_" + str(i)))
            
            # TODO 这里要非常小心，可能会不返回bleu，导致计算错误
            if max(totals) >= 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_mt_bleu_counts", np.array(counts))
                metrics.log_scalar("_mt_bleu_totals", np.array(totals))
                metrics.log_scalar("_mt_bleu_sys_len", sum_logs("_mt_bleu_sys_len"))
                metrics.log_scalar("_mt_bleu_ref_len", sum_logs("_mt_bleu_ref_len"))

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
                        correct=meters["_mt_bleu_counts"].sum,
                        total=meters["_mt_bleu_totals"].sum,
                        sys_len=meters["_mt_bleu_sys_len"].sum,
                        ref_len=meters["_mt_bleu_ref_len"].sum,
                        **smooth,
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("mt_bleu", compute_bleu)


    def build_generator(self, models, args, **unused):
        return SequenceGenerator(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            symbols_to_strip_from_output=set(self.infer_tgt_lang_ids) if self.infer_tgt_lang_ids else None,
        )

    def build_src_tokenizer(self):
        logger.info(f"src-pre-tokenizer: {self.data_cfg.src_pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.src_pre_tokenizer))

    def build_src_bpe(self):
        logger.info(f"tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.tgt_dict

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return self.src_dict
    


    # def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
    #     src_lang_id = self.source_dictionary.index("[{}]".format(self.args.source_lang))
    #     source_tokens = []
    #     for s_t in src_tokens:
    #         s_t = torch.cat([s_t, s_t.new(1).fill_(src_lang_id)])
    #         source_tokens.append(s_t)
    #     dataset = LanguagePairDataset(
    #         source_tokens,
    #         src_lengths,
    #         self.source_dictionary,
    #         tgt_dict=self.target_dictionary,
    #         constraints=constraints,
    #     )
    #     return dataset