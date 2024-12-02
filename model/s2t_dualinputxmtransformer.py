# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy

import torch.nn as nn
from fairseq import checkpoint_utils
from fairseq import utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.models import (
    register_model,
    register_model_architecture,
    FairseqEncoder,
)
from fairseq.models.transformer import TransformerEncoder, TransformerDecoder
from fairseq.utils import safe_hasattr

from .s2t_dualinputtransformer import (
    DualInputS2TTransformerModel,
    TransformerMultiInputDecoder,
    DualInputEncoder,
)
from .s2t_transformer import (
    S2TTransformerEncoder
)
def need_finetuning(ft_params, param_name):
    if ft_params == "all":
        return True
    ft_params_list = ft_params.split(",")
    for ft_param in ft_params_list:
        if ft_param in param_name:
            return True
    return False


# # TODO retire SharedEncoder
# class SharedEncoder(FairseqEncoder):

# class StackedWav2VecEncoderWithAdaptor(FairseqEncoder):

# Note:
# dual input transformer:
#    encoder: wav2vec for speech + mbart encoder for text
#    decoder: mbart decoder  for text
@register_model("dual_input_xm_transformer")
class DualInputXMTransformerModel(DualInputS2TTransformerModel):
    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""

        # add_decoder_args(parser)
        # mbart Transformer
        parser.add_argument(
            "--activation-fn",
            type=str,
            default="relu",
            choices=utils.get_available_activation_fns(),
            help="activation function to use",
        )

        parser.add_argument(
            "--mbart-dropout", type=float, metavar="D", help="dropout probability"
        )
        parser.add_argument(
            "--mbart-attention-dropout",
            type=float,
            metavar="D",
            help="dropout probability for attention weights",
        )
        parser.add_argument(
            "--mbart-activation-dropout",
            type=float,
            metavar="D",
            help="dropout probability after activation in FFN.",
        )

        parser.add_argument(
            "--encoder-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension",
        )
        parser.add_argument(
            "--encoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="encoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--encoder-layers", type=int, metavar="N", help="num encoder layers"
        )
        parser.add_argument(
            "--encoder-attention-heads",
            type=int,
            metavar="N",
            help="num encoder attention heads",
        )
        parser.add_argument(
            "--encoder-normalize-before",
            action="store_true",
            help="apply layernorm before each encoder block",
        )

        parser.add_argument(
            "--decoder-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension",
        )
        parser.add_argument(
            "--decoder-ffn-embed-dim",
            type=int,
            metavar="N",
            help="decoder embedding dimension for FFN",
        )
        parser.add_argument(
            "--decoder-layers", type=int, metavar="N", help="num decoder layers"
        )
        parser.add_argument(
            "--decoder-attention-heads",
            type=int,
            metavar="N",
            help="num decoder attention heads",
        )
        parser.add_argument(
            "--decoder-normalize-before",
            action="store_true",
            help="apply layernorm before each decoder block",
        )
        parser.add_argument(
            "--layernorm-embedding",
            action="store_true",
            help="add layernorm to embedding",
        )
        parser.add_argument(
            "--no-scale-embedding",
            action="store_true",
            help="if True, dont scale embeddings",
        )
        parser.add_argument(
            "--load-pretrained-mbart-from",
            type=str,
            metavar="STR",
            help="model to take text encoder decoder weights from (for initialization)",
        )
        parser.add_argument(
            "--finetune-mbart-decoder-params",
            type=str,
            metavar="STR",
            help="comma-separated param strings to finetune.",
        )
        parser.add_argument(
            "--finetune-mbart-encoder-params",
            type=str,
            metavar="STR",
            help="comma-separated param strings to finetune.",
        )
        parser.add_argument(
            "--skip-encoder-projection",
            action="store_true",
            help="skip the projection layer in encoder",
        )

        parser.add_argument(
            "--enc-grad-mult",
            type=float,
            metavar="V",
            default=1.0,
            help="multiply enc1 and enc2 gradient by V",
        )
        parser.add_argument(
            "--enc2-along-grad-mult",
            type=float,
            metavar="V",
            default=1.0,
            help="multiply enc2 gradient by V if only enc2 is used",
        )
        parser.add_argument(
            "--text-input-cost-ratio",
            type=float,
            default=1.0,
            metavar="V",
            help="text input cost ratio relative to speech input cost",
        )



    @classmethod
    def build_encoder(cls, args, task):
        _args = copy.deepcopy(args)
        _args.dropout = args.mbart_dropout
        _args.attention_dropout = args.mbart_attention_dropout
        _args.activation_dropout = args.mbart_activation_dropout
        _args.max_source_positions = 1024
        enc_emb = nn.Embedding(
            len(task.src_dict), _args.encoder_embed_dim, task.src_dict.pad()
        )
        text_encoder = TransformerEncoder(_args, task.src_dict, enc_emb)
        spch_encoder = S2TTransformerEncoder(args, task)
        if getattr(args, "load_pretrained_mbart_from", None):
            text_encoder = checkpoint_utils.load_pretrained_component_from_model(
                component=text_encoder, checkpoint=args.load_pretrained_mbart_from
            )
        for k, p in text_encoder.named_parameters():
            # Freeze pretrained models by default
            if safe_hasattr(
                args, "finetune_mbart_encoder_params"
            ) and need_finetuning(
                args.finetune_mbart_encoder_params, k
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False
        cross_attentive_loss_before_last_layer = (
            0 if getattr(args, "attentive_cost_regularization", 0.0) > 0.0 else -1
        )
        encoder = DualInputEncoder(
            args,
            spch_encoder,
            text_encoder,
            task.src_dict,
            cross_attentive_loss_before_last_layer,
        )
        return encoder

    @classmethod
    def build_decoder(cls, args, task):
        _args = copy.deepcopy(args)
        _args.dropout = args.mbart_dropout
        _args.attention_dropout = args.mbart_attention_dropout
        _args.activation_dropout = args.mbart_activation_dropout
        _args.max_target_positions = 1024
        dec_emb = nn.Embedding(
            len(task.tgt_dict), _args.encoder_embed_dim, task.tgt_dict.pad()
        )
        decoder = TransformerDecoder(_args, task.tgt_dict, dec_emb)
        if getattr(args, "load_pretrained_mbart_from", None):
            decoder = checkpoint_utils.load_pretrained_component_from_model(
                component=decoder, checkpoint=args.load_pretrained_mbart_from
            )
        if getattr(args, "no_final_norm_decoder", False):
            decoder.layer_norm = None
        for k, p in decoder.named_parameters():
            # Freeze pretrained models by default
            if safe_hasattr(
                args, "finetune_mbart_decoder_params"
            ) and need_finetuning(
                args.finetune_mbart_decoder_params, k
            ):
                p.requires_grad = True
            else:
                p.requires_grad = False

        compute_cross_attentive_loss = (
            True if getattr(args, "attentive_cost_regularization", 0.0) > 0.0 else False
        )
        cross_attentive_loss_without_norm = getattr(
            args, "attentive_cost_without_normalize", False
        )
        cross_attentive_loss_reverse = (
            False  # getattr(args, "attentive_cost_reverse", False)
        )
        decoder = TransformerMultiInputDecoder(
            dictionary=task.target_dictionary,
            spch_decoder=decoder,
            text_decoder=decoder,
            compute_cross_attentive_loss=compute_cross_attentive_loss,
            cross_attentive_loss_with_norm=True
            if not cross_attentive_loss_without_norm
            else False,
            cross_attentive_loss_reverse=cross_attentive_loss_reverse,
        )
        return decoder

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""
        # make sure that all args are properly defaulted
        # (in case there are any new ones)
        dualinputxmtransformer_base(args)

        encoder = cls.build_encoder(args, task)
        decoder = cls.build_decoder(args, task)
        return cls(encoder, decoder)


@register_model_architecture("dual_input_xm_transformer", "dualinputxmtransformer_base")
def dualinputxmtransformer_base(args):

    # mbart model
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(
        args, "encoder_ffn_embed_dim", 4 * args.encoder_embed_dim
    )
    args.encoder_layers = getattr(args, "encoder_layers", 12)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)

    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", 1024)
    args.decoder_ffn_embed_dim = getattr(args, "decoder_ffn_embed_dim", 4 * 1024)
    args.decoder_layers = getattr(args, "decoder_layers", 12)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 16)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", True)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", True)
    args.decoder_layerdrop = getattr(args, "decoder_layerdrop", 0.0)

    args.adaptive_input = getattr(args, "adaptive_input", False)

    args.mbart_attention_dropout = getattr(args, "mbart_attention_dropout", 0.0)
    args.mbart_activation_dropout = getattr(args, "mbart_activation_dropout", 0.0)
    args.mbart_dropout = getattr(args, "mbart_dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", True
    )
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", True)

    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)
