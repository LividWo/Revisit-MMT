tgt=$1

for MASK in v15
do
    DATA=data-bin/multi30k.en-$tgt.bert.mask.$MASK/
    SAVE=checkpoints/mask.$MASK.static.en-$tgt.tiny
    python generate.py $DATA --task retrieval_translation --beam 5 --batch-size 128 --bert-model-name bert-base-uncased --remove-bpe --gen-subset test --quiet --path $SAVE/checkpoint_last10_avg.pt
    python scripts/visual_awareness.py --input $SAVE/retrieval.txt
    # python generate.py $DATA --task retrieval_translation --beam 5 --batch-size 128 --bert-model-name bert-base-uncased --remove-bpe --gen-subset test1 --quiet --path $SAVE/checkpoint_last10_avg.pt
    # python scripts/visual_awareness.py --input $SAVE/retrieval.txt
done

