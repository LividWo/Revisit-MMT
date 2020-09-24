# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
import torch.nn.functional as F

from fairseq import utils
import torch
from . import FairseqCriterion, register_criterion


def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None):
    # lprobs: [batch_size*seq_length, vocab_size]
    # target: [batch_size*seq_length, 1]
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        nll_loss[pad_mask] *= 0.
        smooth_loss[pad_mask] *= 0.
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


@register_criterion('joint_label_smoothed_cross_entropy')
class JointLabelSmoothedCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, task):
        super().__init__(args, task)
        self.eps = args.label_smoothing
    
    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--label-smoothing', default=0.1, type=float, metavar='D',
                            help='epsilon for label smoothing, 0 means no label smoothing')
        # fmt: on

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample['net_input'])
        loss, _ = self.compute_loss(model, net_output, sample, reduce=reduce)
        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens']
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'nll_loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample['target'].size(0),
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    def compute_loss(self, model, net_output, sample, reduce=True):
        target = model.get_targets(sample, net_output)
        padding_mask = ~torch.eq(target, self.padding_idx)
        padding_mask = padding_mask.float()
        # [batch_size, seq_length]
        batch_shape = (target.size(0), target.size(1))
        # [batch_size*seq_length]
        target = target.view(-1)
        total_loss = []
        
        for decoder_out, dot_product_log_softmax in zip(net_output[0], net_output[1]):
            # [batch_size, seq_length, vocab_size]
            lprobs = model.get_normalized_probs(decoder_out, log_probs=True)
            # [batch_size*seq_length, vocab_size]
            lprobs = lprobs.view(-1, lprobs.size(-1))
            # ps1: the results of nll_loss is already negatived.
            # [batch_size*seq_length]
            loss, nll_loss = label_smoothed_nll_loss(
                lprobs, target, self.eps, ignore_index=self.padding_idx,
            )
            loss = -loss

            loss = loss.view(batch_shape[0], batch_shape[1])
            z_loss = dot_product_log_softmax
            # [batch_size, 1]
            z_loss = z_loss.view(-1, 1)
            assert z_loss.shape[0] == loss.shape[0]
            # [batch_size, seq_length]
            loss = z_loss + loss
            # [batch_size, 1, seq_length]
            loss = loss.unsqueeze(1)
            # z_loss = z_loss.view(z_loss.shape[0], 1)
            total_loss.append(loss)
        # [batch_size, top-k, seq_length]
        total_loss = torch.cat(total_loss, dim=1)
        # [batch_size, seq_length]
        y = -torch.logsumexp(total_loss, 1).sum()
        # print("logsumexp shape:", y.shape, y[:2, ])
        # y = y * padding_mask
        # y = y.sum()
        return y, y

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2) if sample_size > 0 else 0.,
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
