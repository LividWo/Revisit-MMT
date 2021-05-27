# Reproduce results for Transformer-tiny


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
DATA='data-bin/multi30k.en-de/'  
ARCH='transformer_tiny'  
SAVE='checkpoints/transformer.en-de.tiny'
tgt='de'

CUDA_VISIBLE_DEVICES=0,1 python train.py $DATA --task translation \
      --arch $ARCH --share-all-embeddings --dropout 0.3 \
      --warmup-updates 2000 --lr 0.005 \
      --max-tokens 4096 \
      --max-update 8000 --target-lang $tgt \
      --save-dir $SAVE \
      --find-unused-parameters --patience 10 
```

#### 3. Evaluate
```
bash evaluate.sh -d $DATA -s test -t translation -p $SAVE
```
evaluation script parameters:

- -s test subset {chose from test(2016)/test1(2017)/test2(coco)}
- -g gpu id you want to use
- -d input data
- -p checkpoint path (note: just path to checkpoint dir, not to the file)
- -b beam size, default to 5
- -t task name, {translation/mmt/retrieval_translation}

Run the evaluation commanda above, you are supposed to see:
> | Generate test with beam=5: BLEU4 = 41.02, 71.0/47.5/34.3/24.5 (BP=0.999, ratio=0.999, syslen=12096, reflen=12103)