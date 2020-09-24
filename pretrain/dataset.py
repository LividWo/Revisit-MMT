import os
import torch
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
import numpy as np


def pickle_dump(data, file_path):
    f_write = open(file_path, 'wb')
    pickle.dump(data, f_write, True)


def pickle_load(file_path):
    f_read = open(file_path, 'rb')
    data = pickle.load(f_read)
    return data


class BERTTokenization(object):
    def __init__(self, tokenizer, max_len=512):
        self.tokenizer = tokenizer
        self.max_len = max_len

        self.cls_id = self.tokenizer.convert_tokens_to_ids(['[CLS]'])[0]
        self.sep_id = self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]
        self.pad_id = self.tokenizer.convert_tokens_to_ids(['[PAD]'])[0]

    def __call__(self, caption):
        tokenized_dict = self.tokenizer.encode_plus(
            text=caption,
            max_length=self.max_len,
            padding='max_length',
            truncation=True
        )

        input_ids, input_masks, input_types = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']

        # if len(input_ids) >= self.max_len:
        #     input_ids = input_ids[:self.max_len - 1] + [self.sep_id]
        #     input_types = input_types[:self.max_len]
        #     input_masks = input_masks[:self.max_len]

        # input_ids += [self.pad_id] * (self.max_len - len(input_ids))
        # input_types += [0] * (self.max_len - len(input_types))
        # input_masks += [0] * (self.max_len - len(input_masks))

        assert len(input_ids) == self.max_len
        assert len(input_types) == self.max_len
        assert len(input_masks) == self.max_len

        return input_ids, input_types, input_masks

    def __str__(self) -> str:
        return 'maxlen%d' % (self.max_len)
        

class CaptionImageDataset(Dataset):
    def __init__(self, file_path, caption_transform, sample=None):
        self.caption_transform = caption_transform
        self.data_source = []
        self.transformed_data = {}

        cache_path = file_path + '_' + str(caption_transform) + '.cache'
        if os.path.exists(cache_path):
            self.transformed_data = pickle_load(cache_path)
            self.data_source = [0] * len(self.transformed_data)
        else:
            with open(file_path, encoding='utf-8') as f:
                for line in f:
                    split = line.strip().split('\t')
                    image_name, image_id, caption = split[0], int(split[1]), split[2]
                    self.data_source.append({
                        'image_id': image_id,
                        'caption': caption,
                        'label': image_id,
                    })

                    if sample is not None and len(self.data_source) >= sample:
                        break
            for idx in tqdm(range(len(self.data_source))):
                self.transformed_data[idx] = self.__get_single_item__(idx)
            pickle_dump(self.transformed_data, cache_path)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self.__get_single_item__(index) for index in indices]
        return self.__get_single_item__(indices)

    def __get_single_item__(self, index):
        if index in self.transformed_data:
            key_data = self.transformed_data[index]
            return key_data
        else:
            group = self.data_source[index]
            image_id, caption, labels = group['image_id'], group['caption'], group['label']
            transformed_caption = self.caption_transform(caption)
            key_data = transformed_caption, image_id, labels

            return key_data

    def get_batch(self, batch):
        caption_token_ids_batch, caption_segment_ids_batch, caption_masks_batch = [], [], []
        labels_batch, image_ids_batch = [], []
        for sample in batch:
            caption_token_ids, caption_segment_ids, caption_masks = sample[0]
            caption_token_ids_batch.append(caption_token_ids)
            caption_segment_ids_batch.append(caption_segment_ids)
            caption_masks_batch.append(caption_masks)

            image_ids_batch.append(sample[1])
            # label = sample[2]
            # one_hot = np.zeros(29001, dtype=int)
            # one_hot[label] = 1
            labels_batch.append(sample[2])

        caption_token_ids_batch = torch.tensor(caption_token_ids_batch, dtype=torch.long)
        caption_segment_ids_batch = torch.tensor(caption_segment_ids_batch, dtype=torch.long)
        caption_masks_batch = torch.tensor(caption_masks_batch, dtype=torch.long)

        image_ids_batch = torch.tensor(image_ids_batch, dtype=torch.long)
        labels_batch = torch.tensor(labels_batch, dtype=torch.long)

        return caption_token_ids_batch, caption_segment_ids_batch, caption_masks_batch, image_ids_batch, labels_batch

