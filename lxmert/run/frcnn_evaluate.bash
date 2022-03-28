## Commenting starting lines as it is not useful during evaluation
# The name of this experiment.
#name=$2
#
## Save logs and models under snap/gqa; make backup.
#output=/data/b/vipul/lxmert/test/$name
#mkdir -p $output/src
#cp -r src/* $output/src/
#cp $0 $output/run.bash

# See Readme.md for option details.
CUDA_VISIBLE_DEVICES=$1 PYTHONPATH=$PYTHONPATH:./src \
    python3 src/tasks/frcnn_evaluate.py \
    --train train --valid "" \
    --llayers 9 --xlayers 5 --rlayers 5 \
    --tqdm --output $output ${@:3}
