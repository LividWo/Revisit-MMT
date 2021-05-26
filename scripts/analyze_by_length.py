import argparse
from collections import defaultdict
from pprint import pprint
import numpy as np


def read_gate(path):
    
    padded_length = []
    gate = []
    with open(path) as f:
        for line in f.readlines():
            if line.startswith('shape:'):
                line = line.strip().split('[')[1]
                shape = line.split(']')[0].split(',')
                pad = int(shape[1])
                batch = int(shape[0])
                padded_length.extend([pad] * batch)
                # print(pad, batch)
            else:
                gate.append([float(num) for num in line.strip()[1:-1].split(',')])
    def list_split(items, n):
        return [items[i:i+n] for i in range(0, len(items), n)]

    gate_by_sent_word = []  # 1000, length, 128  
    for padding, sent_gate in zip(padded_length, gate):
        n = int(len(sent_gate) / padding)
        gate_by_sent_word.append(list_split(sent_gate, n))
    
    return gate_by_sent_word


def read_sentence(path):
    en_toks = []

    with open(path) as f:
        for i, line in enumerate(f.readlines()):
            line = line.strip().split('\t')
            head = line[0][:2]
            if head == 'S-':
                en_toks.append(line[1])
            
    return en_toks

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    # fmt: off
    parser.add_argument('--gate', required=True)
    parser.add_argument('--sent', required=True)
    # fmt: on
    args = parser.parse_args()
    
    data = defaultdict(list)
    for gate, en in zip(read_gate(args.gate), read_sentence(args.sent)):
        length = len(en.split())
        c, v = 0., 0.
        for i in range(length):
            for num in gate[i]:
                if num == 0: 
                    continue
                else:
                    v += num
                    c += 1
        if c == 0:
            data[length].append(v)
        else:
            data[length].append(v/c)
    out = {}
    for k,v in data.items():
        out[k] = np.mean(v)
    pprint(out)