#!/bin/bash

usage(){
        echo "
        Usage:
         -p, --path             checkpoint dir
         -g, --gpu              gpus id
         -s, --split            dataset split
         -b, --beam             beam size
         -d, --data             data set
         -c, --checkpoint     
         -t, --task
        "
}
GPUS=0
CKP=checkpoint_last10_avg.pt
BEAM=5
TASK=mmt
while getopts ":p:g:s:b:d:c:t:" arg; do
        case "${arg}" in
                p)
                        CKPT_DIR=${OPTARG}
                        echo "dir:              $CKPT_DIR"
                        ;;
                g)
                        GPUS=${OPTARG}
                        echo "using gpus:       $GPUS"
                        ;;
                s)
                        SPLIT=${OPTARG}
                        echo "evaluating on the split: $SPLIT"
                        ;;
                b)
                        BEAM=${OPTARG}
                        echo "beam size: $BEAM"
                        ;;
                d)
                        DATA=${OPTARG}
                        echo "data set: $DATA"
                        ;;
                c)
                        CKP=${OPTARG}
                        echo "checkpoint: $CKP"
                        ;;
                t)
                        TASK=${OPTARG}
                        echo "task: $TASK"
                        ;;
                *)
                        echo "unexpected parameter"
                        usage
                        ;;
        esac
done

if [ ! -f "$CKPT_DIR/checkpoint_last10_avg.pt" ]; then
python scripts/average_checkpoints.py \
--inputs $CKPT_DIR \
--output $CKPT_DIR/checkpoint_last10_avg.pt \
--num-epoch-checkpoints  10 
fi

if [[ "$TASK" == "retrieval_translation" ]]; then
CUDA_VISIBLE_DEVICES=$GPUS python generate.py $DATA --task $TASK --bert-model-name bert-base-uncased --beam $BEAM --batch-size 128 --remove-bpe --quiet --gen-subset $SPLIT --path $CKPT_DIR/$CKP 
else
CUDA_VISIBLE_DEVICES=$GPUS python generate.py $DATA --task $TASK --beam $BEAM --batch-size 128 --remove-bpe --quiet --gen-subset $SPLIT --path $CKPT_DIR/$CKP 
fi