# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional
import os.path as op
from fairseq.data import Dictionary


def get_config_from_yaml(yaml_path: Path):
    try:
        import yaml
    except ImportError:
        print("Please install PyYAML: pip install PyYAML")
    config = {}
    if yaml_path.is_file():
        try:
            with open(yaml_path) as f:
                config = yaml.load(f, Loader=yaml.FullLoader)
        except Exception as e:
            raise Exception(f"Failed to load config from {yaml_path.as_posix()}: {e}")
    else:
        raise FileNotFoundError(f"{yaml_path.as_posix()} not found")

    return config


class S2TDataConfig(object):
    """Wrapper class for data config YAML"""

    def __init__(self, yaml_path):
        try:
            import yaml
        except ImportError:
            print("Please install PyYAML to load YAML files for " "S2TJointMT data config")
        self.config = {}
        if op.isfile(yaml_path):
            try:
                with open(yaml_path) as f:
                    self.config = yaml.load(f, Loader=yaml.FullLoader)
            except Exception as e:
                raise Exception(f"Failed to load config from {yaml_path}: {e}")
        else:
            raise FileNotFoundError(f"{yaml_path} not found")

    @property
    def data_name(self):
        """data_name"""
        return self.config.get("data_name", "sp-10")
    
    @property
    def vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("vocab_filename", "dict.txt")

    @property
    def src_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_vocab_filename", "dict.txt")

    @property
    def shuffle(self) -> bool:
        """Shuffle dataset samples before batching"""
        return self.config.get("shuffle", False)

    @property
    def pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("pre_tokenizer", {"tokenizer": None})

    @property
    def bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply after pre-tokenization. Returning
        a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("bpe_tokenizer", {"bpe": None})

    @property
    def src_bpe_tokenizer(self) -> Dict:
        return self.config.get("src_bpe_tokenizer", {"src_bpe_tokenizer": None})

    @property
    def tgt_bpe_tokenizer(self) -> Dict:
        return self.config.get("tgt_bpe_tokenizer", {"tgt_bpe_tokenizer": None})

    @property
    def joint_bpe_tokenizer(self) -> Dict:
        return self.config.get("joint_bpe_tokenizer", {"joint_bpe_tokenizer": None})

    @property
    def prepend_tgt_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_tgt_lang_tag", False)

    @property
    def input_feat_per_channel(self):
        """The dimension of input features (per sign channel)"""
        return self.config.get("input_feat_per_channel", 80)

    @property
    def input_channels(self):
        """The number of channels in the input sign"""
        return self.config.get("input_channels", 1)

    @property
    def sampling_alpha(self):
        """Hyper-parameter alpha = 1/T for temperature-based resampling.
        (alpha = 1 for no resampling)"""
        return self.config.get("sampling_alpha", 1.0)

    @property
    def use_sign_input(self):
        """Needed by the dataset loader to see if the model requires
        raw sign as inputs."""
        return self.config.get("use_sign_input", False)

    @property
    def sign_root(self):
        return self.config.get("sign_root", "")


class S2TJointDataConfig(S2TDataConfig):
    """Wrapper class for data config YAML"""

    @property
    def src_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_vocab_filename", "src_dict.txt")

    @property
    def src_pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("src_pre_tokenizer", {"tokenizer": None})

    @property
    def src_bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply on source text after pre-tokenization.
        Returning a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("src_bpe_tokenizer", {"bpe": None})
    @property
    def extra_sign_path(self):
        return self.config.get("extra_sign_path", {"extra_sign_path": None})
    @property
    def sampling_text_alpha(self):
        """Hyper-parameter alpha = 1/T for temperature-based resampling. (text
        input only) (alpha = 1 for no resampling)"""
        return self.config.get("sampling_text_alpha", 1.0)
    @property
    def combine_lang_datasets(self):
        return self.config.get("combine_lang_datasets", 1.0)
