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
from heapq import nlargest

from fairseq import bleu
from fairseq.data import dictionary


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


def read_src_en(full_en, mask_en):
    en = {}
    mask = {}
    with open(full_en) as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            en[i] = line
    
    with open(mask_en) as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip()
            mask[i] = line
    return en, mask



def get_parser():
    parser = argparse.ArgumentParser(description='Command-line script for BLEU scoring.')
    # fmt: off
    parser.add_argument('--trans', default='-', help='generate.py output')
    parser.add_argument('--mmt', default='-', help='generate.py output')
    parser.add_argument('--en', default='-', help='generate.py output')
    parser.add_argument('--masked_en', default='-', help='generate.py output')
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

    trans_toks, ref_toks = read_file(args.trans)
    mmt_toks, _ = read_file(args.mmt)
    en, masked_en = read_src_en(args.en, args.masked_en)

    bleu_scorer = bleu.Scorer(dict.pad(), dict.eos(), dict.unk())

    trans_score = {}
    for k in trans_toks.keys():
        bleu_scorer.reset(one_init=True)
        sys_tok = dict.encode_line(trans_toks[k])
        ref_tok = dict.encode_line(ref_toks[k])
        bleu_scorer.add(ref_tok, sys_tok)
        trans_score[k] = bleu_scorer.score(args.order)
    
    mmt_score = {}
    for k in mmt_toks.keys():
        bleu_scorer.reset(one_init=True)
        sys_tok = dict.encode_line(mmt_toks[k])
        ref_tok = dict.encode_line(ref_toks[k])
        bleu_scorer.add(ref_tok, sys_tok)
        mmt_score[k] = bleu_scorer.score(args.order)
    
    gap_score = {}
    for k in mmt_score.keys():
        gap_score[k] = mmt_score[k] - trans_score[k]

    topk_gap_score = nlargest(10, gap_score, key=gap_score.get)
    for k in topk_gap_score:
        print("**"*30)
        print(k)
        print("src en:", en[k])
        print("masked en:", masked_en[k])
        print("target de:", ref_toks[k])
        print("text-only translation:", trans_toks[k], trans_score[k])
        print("mmt translation:", mmt_toks[k], mmt_score[k])



if __name__ == '__main__':
    main()
