# The name of this experiment.
name=$2

# Save logs and models under snap/gqa; make backup.
output=snap/swapmix/$name
mkdir -p $output/src
cp -r src/* $output/src/
cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=$PYTHONPATH:./src \
    python src/tasks/gqa_scene_swapmix.py \
    --train train --valid valid \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --loadLXMERTQA /home/vipul/chenyu/lxmert/snap/pretrained/model \
    --batchSize 96 --optim bert --lr 1e-5 --epochs 12 \
    --tqdm --output $output ${@:3}
