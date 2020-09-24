import os
import time
import random
import torch
import argparse
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from dataset import CaptionImageDataset, BERTTokenization
from transformers.tokenization_bert import BertTokenizer
from transformers import BertModel
from model import CaptionImageRetriever


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


def eval_running_model(dataloader, test=False):
    loss_fct = CrossEntropyLoss()
    model.eval()
    eval_loss, eval_hit_times = 0, 0
    nb_eval_steps, nb_eval_examples = 0, 0
    eval_top5, eval_top10, eval_top50 = 0, 0, 0
    for step, batch in enumerate(dataloader, start=1):
        batch = tuple(t.to(device) for t in batch)
        caption_input_ids, caption_segment_ids, caption_input_masks, image_ids, labels = batch

        with torch.no_grad():
            dot_product = model(caption_input_ids, caption_segment_ids, caption_input_masks)
            loss = loss_fct(dot_product, labels)
            logits = F.softmax(dot_product, dim=-1)

        eval_hit_times += (logits.argmax(-1) == labels).sum().item()
        eval_loss += loss.item()

        find_idx = logits.argmax(-1)
        top5_idx = torch.topk(logits, 5)[1].cpu().numpy()
        top10_idx = torch.topk(logits, 10)[1].cpu().numpy()
        top50_idx = torch.topk(logits, 50)[1].cpu().numpy()
        for i, img_id in enumerate(image_ids.cpu().numpy()):
            if img_id in top5_idx[i]:
                eval_top5 += 1
            if img_id in top10_idx[i]:
                eval_top10 += 1
            if img_id in top50_idx[i]:
                eval_top50 += 1

        nb_eval_examples += labels.size(0)
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_hit_times / nb_eval_examples
    if not test:
        result = {
            'train_loss': tr_loss / nb_tr_steps,
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'epoch': epoch,
            'global_step': global_step,
            'r5': eval_top5 / nb_eval_examples,
            'r10': eval_top10 / nb_eval_examples,
            'r50': eval_top50 / nb_eval_examples
        }
    else:
        result = {
            'eval_loss': eval_loss,
            'eval_accuracy': eval_accuracy,
            'r5': eval_top5 / nb_eval_examples,
            'r10': eval_top10 / nb_eval_examples,
            'r50': eval_top50 / nb_eval_examples
        }
    return result


if __name__ == '__main__':
    # 0: 设置参数parser
    parser = argparse.ArgumentParser()
    # 模型参数
    parser.add_argument("--bert-model-name", default='bert-base-uncased', type=str)
    parser.add_argument("--output_dir", default='checkpoints/multi30k/', required=True, type=str)
    parser.add_argument("--train_dir", default='30k', required=True, type=str)

    parser.add_argument("--architecture", default='bi', type=str, help='[poly, bi]')

    parser.add_argument("--poly_m", default=16, type=int, help="Total batch size for eval.")

    parser.add_argument("--feature_dim", default=128, type=int,
                        help="Hidden size of matching features (for both T/image)")

    # image 相关参数
    parser.add_argument('--retriever_dropout', type=float, default=0.1,
                        help='dropout probability for retriever')
    parser.add_argument('--image_embedding_file', type=str, required=True, default='../../feature_extractor/resnet50-avgpool.npy',
                        help='image_embedding_file')

    # GPU 相关参数
    parser.add_argument('--seed', type=int, default=123, help="random seed for initialization")
    parser.add_argument('--gpu', type=int, default=0)

    # 数据处理参数
    parser.add_argument("--max_length", default=50, type=int)
    parser.add_argument("--train_batch_size", default=512, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=256, type=int, help="Total batch size for eval.")

    # 训练参数
    parser.add_argument("--learning_rate", default=3e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0001, type=float)
    parser.add_argument("--warmup_steps", default=1000, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--print_freq", default=100, type=int, help="Total batch size for eval.")
    parser.add_argument("--eval_freq", default=100, type=int, help="Total batch size for eval.")
    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--pretrain", action="store_true")
    parser.add_argument("--pretrain_matcher", default=None)

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

    # 4：加载数据集，设置dataloader
    if not args.eval:
        train_dataset = CaptionImageDataset(os.path.join(args.train_dir, 'train.txt'), bert_tokenization)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size, 
            collate_fn=train_dataset.get_batch,
            shuffle=True
        )
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
        val_dataset = CaptionImageDataset(os.path.join(args.train_dir, 'valid.txt'), bert_tokenization)
        print("length of train_dataloader", len(train_dataloader))
        print("total num of training steps:", t_total)
    else:
        val_dataset = CaptionImageDataset(os.path.join(args.train_dir, 'valid.txt'), bert_tokenization)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.eval_batch_size,
        collate_fn=val_dataset.get_batch,
        shuffle=False
    )
    

    # 5：准备模型存储
    epoch_start = 1
    global_step = 0
    best_eval_accuracy = float(0)
    best_eval_loss = float('inf')

    args.output_dir = args.output_dir + args.bert_model_name
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    log_wf = open(os.path.join(args.output_dir, 'log.txt'), 'a', encoding='utf-8')

    state_save_path = os.path.join(args.output_dir, 'retriever.bin')

    print('=' * 80)
    print('Train dir:', args.train_dir)
    print('Output dir:', args.output_dir)
    print('=' * 80)
    
    bert = BertModel.from_pretrained(args.bert_model_name)
    model = CaptionImageRetriever(bert.config, bert=bert, args=args)
    if args.pretrain:
        print('continue training, loading pretrained retriever from:', args.pretrain_matcher)
        model.load_state_dict(torch.load(args.pretrain_matcher), strict=True)
    model.to(device)
    # for name, p in model.named_parameters():
        # print(name,  p.requires_grad)
    
    if args.eval:
        print('Loading parameters from', state_save_path)
        model.load_state_dict(torch.load(state_save_path))
        test_result = eval_running_model(val_dataloader, test=True)
        print(test_result)
        exit()

    # 7: 训练设置
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay,
        },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total
    )
    # print(optimizer.defaults['lr'])

    tr_total = int(
        train_dataset.__len__() / args.train_batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
    print_freq = args.print_freq
    eval_freq = min(len(train_dataloader) // 2, args.eval_freq)
    print('Print freq:', print_freq, "Eval freq:", eval_freq)

    for epoch in range(epoch_start, int(args.num_train_epochs) + 1):
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        with tqdm(total=len(train_dataloader)) as bar:
            for step, batch in enumerate(train_dataloader, start=1):
                model.train()
                optimizer.zero_grad()
                batch = tuple(t.to(device) for t in batch)
                caption_input_ids, caption_segment_ids, caption_input_masks, image_ids, labels = batch
                assert caption_input_ids[0][0] == tokenizer.cls_token_id
                loss = model(caption_input_ids, caption_segment_ids, caption_input_masks, labels)

                loss = loss / args.gradient_accumulation_steps

                loss.backward()
                tr_loss += loss.item()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    nb_tr_steps += 1

                    optimizer.step()
                    scheduler.step()
                    model.zero_grad()
                    global_step += 1

                    if nb_tr_steps and nb_tr_steps % print_freq == 0:
                        bar.update(min(print_freq, step))
                        time.sleep(0.02)
                        # print(global_step, tr_loss / nb_tr_steps)
                        log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))
                    # print(optimizer.param_groups[0]['lr'])
                    # print(scheduler.get_last_lr()[0])

                    if global_step and global_step % eval_freq == 0:
                        val_result = eval_running_model(val_dataloader)
                        print('Global Step %d VAL res:\n' % global_step, val_result)
                        log_wf.write('Global Step %d VAL res:\n' % global_step)
                        log_wf.write(str(val_result) + '\n')

                        if val_result['eval_loss'] < best_eval_loss:
                            best_eval_loss = val_result['eval_loss']
                            val_result['best_eval_loss'] = best_eval_loss
                            # save model
                            print('[Saving at]', state_save_path)
                            log_wf.write('[Saving at] %s\n' % state_save_path)
                            torch.save(model.state_dict(), state_save_path)
                log_wf.flush()
        # add a eval step after each epoch
        val_result = eval_running_model(val_dataloader)
        print('Epoch %d, Global Step %d VAL res:\n' % (epoch, global_step), val_result)
        log_wf.write('Global Step %d VAL res:\n' % global_step)
        log_wf.write(str(val_result) + '\n')

        if val_result['eval_loss'] < best_eval_loss:
            best_eval_loss = val_result['eval_loss']
            val_result['best_eval_loss'] = best_eval_loss
            # save model
            print('[Saving at]', state_save_path)
            log_wf.write('[Saving at] %s\n' % state_save_path)
            torch.save(model.state_dict(), state_save_path)
        print(global_step, tr_loss / nb_tr_steps)
        log_wf.write('%d\t%f\n' % (global_step, tr_loss / nb_tr_steps))

        # if val_result['eval_accuracy'] > best_eval_accuracy:
        #     best_eval_accuracy = val_result['eval_accuracy']
        #     val_result['best_eval_accuracy'] = best_eval_accuracy
        # #     # save model
        #     print('[Saving at]', state_save_path)
        #     log_wf.write('[Saving at] %s\n' % state_save_path)
        #     torch.save(model.state_dict(), state_save_path)