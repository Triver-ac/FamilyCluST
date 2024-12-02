#!/usr/bin/env/python
import math


import torch
from torch import nn
import torch.nn.functional as F

from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import LabelSmoothedCrossEntropyCriterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II
import pdb
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

@register_criterion(
    "label_smoothed_cross_entropy_with_hidden_mapping_sign")
class LabelSmoothedCrossEntropyCriterionWithHiddenMappingSign(LabelSmoothedCrossEntropyCriterion):
    def __init__(
            self,
            task,
            sentence_avg,
            label_smoothing,
            ignore_prefix_size=0,
            report_accuracy=False,
            hidden_mapping_loss_type="mse",
            similarity_loss_type="l2",
            input_type="sign",
    ):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.similarity_loss_type = similarity_loss_type
        self.hidden_mapping_loss_type = hidden_mapping_loss_type
        self.input_type = input_type
        self.extra = []

    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        LabelSmoothedCrossEntropyCriterion.add_args(parser)
        parser.add_argument(
            "--input-type",
            default="sign",
            type=str,
            help="sign, text, multi"
        )
        parser.add_argument(
            "--hidden-mapping-loss-type",
            default="mse",
            type=str,
            help="hidden mapping loss type: mse, kl, none..."
        )
        parser.add_argument(
            "--similarity-loss-type",
            default="l2",
            type=str,
            help="hidden mapping loss type: cos, l2, none..."
        )
        parser.add_argument(
            "--hidden-mapping-task-flag",
            action='store_true',
            help="Whether to perform the Hidden Mapping task."
        )



    def compute_decoder_loss(self, sample, sign_net_output, text_net_output, model):
        target = sample["target"]  # [B, S]
        non_pad_mask = target.ne(self.padding_idx)  # [B, S], 1: not pad, 0: pad
        non_pad_mask = non_pad_mask.view(-1, 1)  # [B * S, 1]
        non_pad_idx = non_pad_mask.nonzero(as_tuple=True)[0]  # [ Select Size, 1]

        sign_decoder_hidden = sign_net_output[0]
        text_decoder_hidden = text_net_output[0]

        sign_decoder_hidden = \
            sign_decoder_hidden.reshape(-1, sign_decoder_hidden.size(-1))
        text_decoder_hidden = \
            text_decoder_hidden.reshape(-1, text_decoder_hidden.size(-1))

        sign_decoder_hidden = sign_decoder_hidden.index_select(dim=0, index=non_pad_idx)
        text_decoder_hidden = text_decoder_hidden.index_select(dim=0, index=non_pad_idx)

        # hidden mapping loss func
        if self.hidden_mapping_loss_type == "kl":

            #输出之前需不需要log正则化
            # student_logit = model.get_normalized_probs(net_output, log_probs=True, sample=sample)
            # teacher_logit = model.get_normalized_probs(gloss_decoder_output, log_probs=True, sample=sample)

            target = model.get_targets(sample, sign_net_output)
            if self.ignore_prefix_size > 0:
                target = target[:, self.ignore_prefix_size :].contiguous()

            hidden_mapping_loss = self.get_kl_loss(sign_decoder_hidden,text_decoder_hidden,target)

        elif self.hidden_mapping_loss_type == "mse":
            self.mse_loss = nn.MSELoss(reduction="none")

            hidden_mapping_loss = self.mse_loss(
                sign_decoder_hidden, text_decoder_hidden)  # [ not pad size, H]

            hidden_mapping_loss = hidden_mapping_loss.sum(dim=-1).sum(dim=-1)
        return hidden_mapping_loss

    def compute_sentenceLevel_loss(self, sign_encoder_hidden, text_encoder_hidden, sign_encoder_mask, text_encoder_mask):
        # 将mask转为浮点数
        sign_encoder_padding_mask = (~sign_encoder_mask).float().unsqueeze(2)
        text_encoder_padding_mask = (~text_encoder_mask).float().unsqueeze(2)

        # 将不需要的位置置零
        masked_sign_encoder_output = sign_encoder_hidden.transpose(0,1) * sign_encoder_padding_mask
        masked_text_encoder_output = text_encoder_hidden.transpose(0,1) * text_encoder_padding_mask

        # 平均池化
        sentence_sign_hidden = masked_sign_encoder_output.sum(dim=1) / sign_encoder_padding_mask.sum(dim=1)
        sentence_text_hidden = masked_text_encoder_output.sum(dim=1) / text_encoder_padding_mask.sum(dim=1)

        if self.similarity_loss_type == "l2":
            l2_distance = torch.norm(sentence_sign_hidden - sentence_text_hidden, dim=-1)
            loss = l2_distance.sum(dim=-1)
        elif self.similarity_loss_type == "cos":
            similarity = F.cosine_similarity(sentence_sign_hidden, sentence_text_hidden, dim=-1)
            loss = 1.0 - similarity.sum(dim=-1)
        return loss
    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        sign_loss, sign_nll_loss = torch.zeros(1).to(device), torch.zeros(1).to(device)
        text_loss, text_nll_loss = torch.zeros(1).to(device), torch.zeros(1).to(device)
        similarity_loss, hidden_mapping_loss =  torch.zeros(1).to(device), torch.zeros(1).to(device)
        if self.input_type == "sign" or self.input_type == "multi":
            # if "mode" in sample["net_input"].keys()
            if "net_text_input" in sample.keys():
                sign_encoder_output, sign_net_output = model(**sample["net_input"])
                # self.extra.append({"id":sample["id"],"lang":sample["lang"],"extract_feature":sign_encoder_output["encoder_out"][0]})
                sign_loss, sign_nll_loss = self.compute_loss(model, sign_net_output, sample, reduce=reduce)
        if self.input_type == "text" or self.input_type == "multi":
            if "net_text_input" in sample.keys():
                text_encoder_output, text_net_output = model(**sample["net_text_input"], mode="text")
            else:
                text_encoder_output, text_net_output = model(**sample["net_input"])
            text_loss, text_nll_loss = self.compute_loss(model, text_net_output, sample, reduce=reduce)

        if self.input_type == "multi" and self.task.args.hidden_mapping_task_flag and "net_text_input" in sample.keys():

            # similarity_loss = self.compute_sentenceLevel_loss(sign_encoder_output["encoder_out"][0], text_encoder_output["encoder_out"][0], \
            #                                                   sign_encoder_output["encoder_padding_mask"][0], text_encoder_output["encoder_padding_mask"][0])
            hidden_mapping_loss = self.compute_decoder_loss(sample, sign_net_output, text_net_output, model)


        loss = sign_loss + text_loss + hidden_mapping_loss

        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        logging_output = {
            "loss": loss.data,
            "hidden_mapping_loss": hidden_mapping_loss.data,
            "similarity_loss": similarity_loss.data,
            "sign_loss": sign_loss.data,
            "sign_nll_loss": sign_nll_loss.data,
            "text_loss": text_loss.data,
            "text_nll_loss": text_nll_loss.data,
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            if self.input_type == "sign" or self.input_type == "multi" and "net_text_input" in sample.keys():
                n_correct, total = self.compute_accuracy(model, sign_net_output, sample)
                logging_output["sign_n_correct"] = utils.item(n_correct.data)
                logging_output["sign_total"] = utils.item(total.data)
            if self.input_type == "text" or self.input_type == "multi":
                n_correct, total = self.compute_accuracy(model, text_net_output, sample)
                logging_output["text_n_correct"] = utils.item(n_correct.data)
                logging_output["text_total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def get_kl_loss(self, student_logit, teacher_logit, target, temperature=1.0):

        if student_logit.shape[0] != teacher_logit.shape[0]:
            raise ValueError("can not do KD")
        else:
            # padding_mask = target.eq(self.padding_idx)
            p_loss = F.kl_div(F.log_softmax(student_logit / temperature, dim=-1), F.softmax(teacher_logit / temperature, dim=-1), log_target=False, reduction='none')
            q_loss = F.kl_div(F.log_softmax(teacher_logit / temperature, dim=-1), F.softmax(student_logit / temperature, dim=-1), log_target=False, reduction='none')

            # p_loss = p_loss.sum(-1).masked_fill_(padding_mask, 0.0).sum()
            # q_loss = q_loss.sum(-1).masked_fill_(padding_mask, 0.0).sum()
            p_loss = p_loss.sum(-1).sum()
            q_loss = q_loss.sum(-1).sum()

            loss = (p_loss + q_loss) / 2
        return loss

    @classmethod
    def reduce_metrics(cls, logging_outputs, ) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        hidden_mapping_loss_sum = sum(log.get("hidden_mapping_loss", 0) for log in logging_outputs)
        similarity_loss_sum = sum(log.get("similarity_loss", 0) for log in logging_outputs)
        sign_loss_sum = sum(log.get("sign_loss", 0) for log in logging_outputs)
        sign_nll_loss_sum = sum(log.get("sign_nll_loss", 0) for log in logging_outputs)
        text_loss_sum = sum(log.get("text_loss", 0) for log in logging_outputs)
        text_nll_loss_sum = sum(log.get("text_nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "hidden_mapping_loss", hidden_mapping_loss_sum / sample_size / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "similarity_loss", similarity_loss_sum / sample_size / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "sign_loss", sign_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "sign_nll_loss", sign_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "sign_ppl", lambda meters: utils.get_perplexity(meters["sign_nll_loss"].avg)
        )
        sign_total = utils.item(sum(log.get("sign_total", 0) for log in logging_outputs))
        if sign_total > 0:
            metrics.log_scalar("sign_total", sign_total)
            sign_n_correct = utils.item(
                sum(log.get("sign_n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("sign_n_correct", sign_n_correct)
            metrics.log_derived(
                "sign_accuracy",
                lambda meters: round(
                    meters["sign_n_correct"].sum * 100.0 / meters["sign_total"].sum, 3
                )
                if meters["sign_total"].sum > 0
                else float("nan"),
            )

        metrics.log_scalar(
            "text_loss", text_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "text_nll_loss", text_nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "text_ppl", lambda meters: utils.get_perplexity(meters["text_nll_loss"].avg)
        )
        text_total = utils.item(sum(log.get("text_total", 0) for log in logging_outputs))
        if sign_total > 0:
            metrics.log_scalar("text_total", text_total)
            text_n_correct = utils.item(
                sum(log.get("text_n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("text_n_correct", text_n_correct)
            metrics.log_derived(
                "text_accuracy",
                lambda meters: round(
                    meters["text_n_correct"].sum * 100.0 / meters["text_total"].sum, 3
                )
                if meters["text_total"].sum > 0
                else float("nan"),
            )
