# The name of this experiment.
name=$2

# Save logs and models under snap/gqa; make backup.
output=/data/b/vipul/lxmert/logs/gqa/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python3 src/tasks/gqa_swapmix.py \
    --train train --valid valid  \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA /data/b/vipul/pretrained/lxmert/snap/pretrained/model \
    --batchSize 32 --optim bert --lr 1e-5 --epochs 4 \
    --tqdm --output $output ${@:3}
