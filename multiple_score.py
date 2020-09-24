#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
BLEU scoring of generated translations against reference translations.
"""

import argparse
import os
import sys

from fairseq import bleu
from fairseq.data import dictionary
from vizseq.scorers.meteor import METEORScorer


def read_file(path):
    sys_toks = {}
    ref_toks = {}

    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            head = line[0][:2]
            if head == 'H-':
                idx = int(line[0][2:])
                sys_toks[idx] = line[2]
            
            if head == 'T-':
                idx = int(line[0][2:])
                ref_toks[idx] = line[1]

    return sys_toks, ref_toks


def get_parser():
    parser = argparse.ArgumentParser(description='Command-line script for BLEU scoring.')
    # fmt: off
    parser.add_argument('-i', '--input', default='-', help='generate.py output')
    parser.add_argument('-o', '--order', default=4, metavar='N',
                        type=int, help='consider ngrams up to this order')
    parser.add_argument('--ignore-case', action='store_true',
                        help='case-insensitive scoring')
    parser.add_argument('--sentence-bleu', action='store_true',
                        help='report sentence-level BLEUs (i.e., with +1 smoothing)')
    # fmt: on
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    print(args)

    dict = dictionary.Dictionary()

    sys_toks, ref_toks = read_file(args.input)
    bleu_scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())
    translations, ref = [], []
    for k in sys_toks.keys():
        translations.append(sys_toks[k])
        ref.append(ref_toks[k])

    for k in sys_toks.keys():
        sys_tok = dict.encode_line(sys_toks[k])
        ref_tok = dict.encode_line(ref_toks[k])
        bleu_scorer.add(ref_tok, sys_tok)
        # print(sys_tok, ref_tok)
    print(bleu_scorer.result_string(args.order))

    meteor_score = METEORScorer(sent_level=False, corpus_level=True).score(
        translations, [ref]
    )
    print(meteor_score)


if __name__ == '__main__':
    main()
