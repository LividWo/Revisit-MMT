# Reproduce results for Gated Fusion


#### 1. Preprocess
```python
tgt=de
TEXT=data/multi30k-en-$tgt/
python preprocess.py --source-lang en --target-lang $tgt \
  --trainpref $TEXT/train \
  --validpref $TEXT/valid \
  --testpref $TEXT/test.2016,$TEXT/test.2017,$TEXT/test.coco \
  --destdir data-bin/multi30k.en-$tgt \
  --workers 8 --joined-dictionary
```

#### 2. Training(en->de as an example)
```python
DATA='data-bin/multi30k.en-de/'  # input data
ARCH='gated_tiny'  # model structure
SAVE='checkpoints/gated.en-de.tiny'  # save dir
FEATURE=xxxxx  # path to visual features you downloaded
tgt='de'

CUDA_VISIBLE_DEVICES=0,1 python train.py $DATA --task mmt \
      --arch $ARCH --share-all-embeddings --dropout 0.3 \
      --warmup-updates 2000 --lr 0.005 \
      --max-tokens 4096 \
      --max-update 8000 --target-lang $tgt \
      --save-dir $SAVE \
      --visual_feature_file $FEATURE \
      --find-unused-parameters --patience 10 
```

#### 3. Evaluate
```
bash evaluate.sh -g 0 -d $DATA -s test -p -t mmt $SAVE
```
evaluation script parameters:

- -s test subset {chose from test(2016)/test1(2017)/test2(coco)}
- -g gpu id you want to use
- -d input data
- -p checkpoint path (note: just path to checkpoint dir, not to the file)
- -b beam size, default to 5
- -t task name, {translation/mmt/retrieval_translation}

Run the evaluation commanda above, you are supposed to see:
> Generate test with beam=5: BLEU4 = 41.96, 71.4/48.1/35.0/25.7 (BP=1.000, ratio=1.001, syslen=12121, reflen=12103)


#### 4. Model analysis
If you would like to analysis the gate value, uncomment L313-314 in [gated.py](https://github.com/LividWo/Revisit-MMT/blob/master/fairseq/models/gated.py)

And then run the evaluation script as above to write gating matrix to a local file, then you can compute the averaged gate value using:
```python
python scripts/visual_awareness.py --input checkpoints/gated.en-de.tiny/gated.txt 
```