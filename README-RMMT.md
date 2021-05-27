# Reproduce results for RMMT


#### 1. Preprocess
```python
tgt=de
TEXT=data/multi30k-en-$tgt/
python bert_preprocess.py --source-lang en --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco \
  --destdir data-bin/multi30k.en-$tgt.bert \
  --workers 8 --joined-dictionary --bert-model-name bert-base-uncased
```

#### 2. Training(en->de as an example)
```bash
DATA='data-bin/multi30k.en-de.bert/'
ARCH='static_tiny'
SAVE='checkpoints/rmmt.en-de.tiny'
TOPK=5
RETRIEVER=xxxxx  # path to the pre-trained retriever you downloaded
FEATURE=xxxxx  # path to visual features you downloaded
tgt='de'

CUDA_VISIBLE_DEVICES=0,1 python train.py $DATA --task retrieval_translation \
      --arch $ARCH --share-all-embeddings --dropout 0.3 \
      --warmup-updates 2000 --lr 0.004 \
      --max-tokens 4096 \
      --max-update 15000 --target-lang $tgt \
      --save-dir $SAVE \
      --image_embedding_file $FEATURE \
      --image_feature_file $FEATURE \
      --pretrained_retriever $RETRIEVER\
      --find-unused-parameters --merge_option max \
      --feature_dim 128 --topk $TOPK \
      --bert-model-name bert-base-uncased \
      --patience 10 
```

#### 3. Evaluate
```
bash evaluate.sh -g 0 -d $DATA -s test -p -t retrieval_translation $SAVE
```
evaluation script parameters:

- -s test subset {chose from test(2016)/test1(2017)/test2(coco)}
- -g gpu id you want to use
- -d input data
- -p checkpoint path (note: just path to checkpoint dir, not to the file)
- -b beam size, default to 5
- -t task name, {translation/mmt/retrieval_translation}

Run the evaluation commanda above, you are supposed to see:
> | Generate test with beam=5: BLEU4 = 41.96, 71.4/48.1/35.0/25.7 (BP=1.000, ratio=1.001, syslen=12121, reflen=12103)


#### 4. Model analysis
If you would like to analysis the gate value, uncomment L314-315 in [rmmt.py](https://github.com/LividWo/Revisit-MMT/blob/master/fairseq/models/gated.py)

And then run the evaluation script as above to write gating matrix to a local file, then you can compute the averaged gate value using:
```python
python scripts/visual_awareness.py --input checkpoints/rmmt.en-de.tiny/gated.txt 
```

You shall see:
> average gate value: 4.045268239535059e-21
>1e-10 rate: 0.0