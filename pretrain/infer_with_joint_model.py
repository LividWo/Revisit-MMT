import os
import time
import random
import torch
import shutil
import argparse
import numpy as np
from tqdm import tqdm


from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss

# from bert import
from transformers import BertModel, BertConfig
from transformers.tokenization_bert import BertTokenizer

from model import CaptionImageRetriever
from dataset import BERTTokenization, CaptionImageDataset


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


if __name__ == '__main__':
    # 0: 设置参数parser
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument("--bert-model-name", default='bert-base-uncased', type=str)
    parser.add_argument("--checkpoint", default='../checkpoints/top1.pt', required=True)
    parser.add_argument("--output_dir", default='checkpoints/', type=str)
    parser.add_argument("--train_dir", default='30k', type=str)

    parser.add_argument("--feature_dim", default=128, type=int,
                        help="Hidden size of matching features (for both T/image)")

    # image 相关参数
    parser.add_argument('--retriever_dropout', type=float, default=0.1,
                        help='dropout probability for retriever')
    parser.add_argument('--image_emb_fix', action='store_true',
                        help='fix image embedding')
    parser.add_argument('--image_embedding_file', type=str, default='../../feature_extractor/resnet50-avgpool.npy',
                        help='image_embedding_file')

    # GPU 相关参数
    parser.add_argument('--seed', type=int, default=666, help="random seed for initialization")
    parser.add_argument('--gpu', type=int, default=0)

    # 数据处理参数
    parser.add_argument("--max_length", default=50, type=int)
    parser.add_argument("--train_batch_size", default=128, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=200, type=int, help="Total batch size for eval.")

    # 训练参数
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--warmup_steps", default=500, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--print_freq", default=100, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_freq", default=100, type=int, help="Total batch size for eval.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")

    args = parser.parse_args()
    print(args)

    # 1：set GPU environment
    os.environ["CUDA_VISIBLE_DEVICES"] = "%d" % args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2：随机种子设置
    set_seed(args)

    # 3: BERT 初始化
    tokenizer = BertTokenizer.from_pretrained(args.bert_model_name)
    bert_tokenization = BERTTokenization(tokenizer=tokenizer, max_len=args.max_length)
    bert = BertModel.from_pretrained(args.bert_model_name)
    bert_config = BertConfig.from_pretrained(args.bert_model_name)

    # 4：加载数据集，设置dataloader
    test_dataset = CaptionImageDataset(os.path.join(args.train_dir, 'valid.txt'), bert_tokenization)
    test_dataloader = DataLoader(test_dataset,
                                batch_size=args.eval_batch_size, collate_fn=test_dataset.get_batch,
                                shuffle=False)

    # 5：加载预训练模型
    print('=' * 80)
    print('Loading pre-trained retriever dir:', args.checkpoint)
    print('=' * 80)
    model_state_dict = torch.load(args.checkpoint, map_location="cpu")
    model_state_dict = model_state_dict['model']

    # 6：build model
    model = CaptionImageRetriever(bert_config, bert=bert, args=args)
    # for name, param in model.named_parameters():
        # print(name)
    filter_model_state_dict = {}
    top1_name, top1_idx = 'encoder.retriever.', 18
    topk_name, topk_idx = 'retriever.', 10
    for k, v in model_state_dict.items():
        if k.startswith(top1_name):
            filter_model_state_dict[k[top1_idx:]] = v

    model.load_state_dict(filter_model_state_dict, strict=False)

    model.to(device)
    model.eval()

    eval_top1, eval_top5 = 0, 0
    eval_top10, eval_top50 = 0, 0
    nb_eval_examples = 0
    loss_fct = CrossEntropyLoss()
    dist = dict()
    for step, batch in tqdm(enumerate(test_dataloader, start=1)):
        batch = tuple(t.to(device) for t in batch)
        caption_input_ids, caption_segment_ids, caption_input_masks, image_ids, labels = batch
        with torch.no_grad():
            dot_product = model(caption_input_ids, caption_segment_ids, caption_input_masks)
            logits = torch.nn.functional.softmax(dot_product, dim=-1)

        # eval_top1 += (logits.argmax(-1) == torch.argmax(labels, 1)).sum().item()
        # nb_eval_examples += labels.size(0)
        # np_log = logits.numpy()
        find_idx = logits.argmax(-1)
        # print(find_idx)
        eval_top1 += (find_idx == image_ids).sum().item()
        _, top1_idx = torch.topk(logits, 1)
        top1_idx = top1_idx.cpu().numpy()
        # print(top1_idx)
        # print(top1_idx.size)
        
        _, top5_idx = torch.topk(logits, 5)  # B, 5
        top5_idx = top5_idx.cpu().numpy()
        for idx5 in top1_idx:
            for idx in idx5:
                # idx = idx[0]
                if idx in dist.keys():
                    dist[idx] += 1
                else:
                    dist[idx] = 1 
        _, top10_idx = torch.topk(logits, 10)  # B, 5
        top10_idx = top10_idx.cpu().numpy()
        _, top50_idx = torch.topk(logits, 50)  # B, 5
        top50_idx = top50_idx.cpu().numpy()
        for i, img_id in enumerate(image_ids.cpu().numpy()):
            # if img_id not in top1_idx[i]:
                # print(img_id, top1_idx[i])
            if img_id in top5_idx[i]:
                eval_top5 += 1
            if img_id in top10_idx[i]:
                eval_top10 += 1
            if img_id in top50_idx[i]:
                eval_top50 += 1
        nb_eval_examples += image_ids.size(0)
    #
    print(eval_top1 / nb_eval_examples, eval_top1)
    print(eval_top5 / nb_eval_examples)
    print(eval_top10 / nb_eval_examples)
    print(eval_top50 / nb_eval_examples)

    print(sorted(dist.items(), key=lambda x: x[1], reverse=True))
    total = 0
    for k,v in dist.items():
        total += v
    print("repeat rate:", total, total/len(dist))
    print("types of retrieved images:", len(dist))