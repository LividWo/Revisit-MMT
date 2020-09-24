import argparse
from collections import defaultdict
import random
from tqdm import tqdm
import pickle


def read_and_cat(img_path, text_path, idx, output_path):
    imgs = []
    with open(img_path, 'r') as f_img:
        for line in f_img.readlines():
            line = line.strip()
            imgs.append(line)
    caps = []
    with open(text_path, 'r') as f_cap:
        for line in f_cap.readlines():
            line = line.strip()
            caps.append(line)
    out = open(output_path, 'w')
    for img, cap in zip(imgs, caps):
        out.write(img + '\t' + idx[img] + '\t' + cap + '\n')


# use all Multi30k train, valid, test to train retriever
def baseline_1():

    # step 1: read image_name -> id mapping
    idx = dict()
    with open('data/image_splits/all.dict', 'r') as f_dict:
        for line in f_dict:
            line = line.strip().split()
            idx[line[-1]] = line[0]
            # key: 1000919630.jpg value: 7
    
    # preprocess train file
    read_and_cat(
        img_path='data/image_splits/train_all.txt',
        text_path='data/tok/train_all.en',
        idx=idx,
        output_path='data/train.txt'
    )
    # preprocess valid file
    read_and_cat(
        img_path='data/image_splits/val.txt',
        text_path='data/tok/val.en',
        idx=idx,
        output_path='data/valid.txt'
    )
    # preprocess train file
    read_and_cat(
        img_path='data/image_splits/test_2016_flickr.txt',
        text_path='data/tok/test.en',
        idx=idx,
        output_path='data/test.txt'
    )


# use all Multi30k train, valid, to train retriever, test on test 2016
def baseline_2():

    # step 1: read image_name -> id mapping
    idx = dict()
    with open('../../feature_extractor/img_index.txt', 'r') as f_dict:
        for line in f_dict:
            line = line.strip().split()
            idx[line[-1]] = line[0]
            # key: 1000919630.jpg value: 7
    
    # preprocess train file
    read_and_cat(
        img_path='data/image_splits/train.txt',
        text_path='data/tok/train.en',
        idx=idx,
        output_path='data/train.txt'
    )
    # preprocess valid file
    read_and_cat(
        img_path='data/image_splits/val.txt',
        text_path='data/tok/val.en',
        idx=idx,
        output_path='data/valid.txt'
    )
    # preprocess train file
    read_and_cat(
        img_path='data/image_splits/test_2016_flickr.txt',
        text_path='data/tok/test.2016.en',
        idx=idx,
        output_path='data/test.txt'
    )


# use flickr30k train retriever, test on val
def flickr30k():

    # step 1: read image_name -> id mapping
    idx = dict()
    # with open('data/image_splits/all.dict', 'r') as f_dict:
    with open('../../feature_extractor/img_index.txt', 'r') as f_dict:
        for line in f_dict:
            line = line.strip().split()
            idx[line[-1]] = line[0]
            # key: 1000919630.jpg value: 7
    
    # preprocess train file
    read_and_cat(
        img_path='data/image_splits/flickr30k.txt',
        text_path='data/tok/flickr30k.en',
        idx=idx,
        output_path='data/train.txt'
    )
    # preprocess valid file
    read_and_cat(
        img_path='data/image_splits/val.txt',
        text_path='data/tok/val.en',
        idx=idx,
        output_path='data/valid.txt'
    )
    # preprocess train file
    read_and_cat(
        img_path='data/image_splits/test_2016_flickr.txt',
        text_path='data/tok/test.2016.en',
        idx=idx,
        output_path='data/test.txt'
    )


def vatex():
    def process(split=None):
        imgs = []
        with open('vatex/raw_file/{}.video'.format(split), 'r') as f_video:
            for line in f_video.readlines():
                line = line.strip()
                imgs.append(line)
        caps = []
        with open('vatex/raw_file/{}.en'.format(split), 'r') as f_cap:
            for line in f_cap.readlines():
                line = line.strip().lower()
                caps.append(line)
        out = open('vatex/{}.txt'.format(split), 'w')
        for img, cap in zip(imgs, caps):
            out.write(img + '\t' + img + '\t' + cap + '\n')
    process('all')
    process('valid')

if __name__ == '__main__':
    vatex()
