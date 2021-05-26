tgt=$1

for MASK in v15 v30 v45 
do
    DATA=data-bin/multi30k.en-$tgt.mask.$MASK/
    SAVE=checkpoints/mask.$MASK.gated.en-$tgt.tiny
    python generate.py $DATA --task mmt --beam 5 --batch-size 128 --remove-bpe --gen-subset test --quiet --path $SAVE/checkpoint_last10_avg.pt
    python scripts/visual_awareness.py --input $SAVE/gated.txt
    python generate.py $DATA --task mmt --beam 5 --batch-size 128 --remove-bpe --gen-subset test1 --quiet --path $SAVE/checkpoint_last10_avg.pt
    python scripts/visual_awareness.py --input $SAVE/gated.txt
done
