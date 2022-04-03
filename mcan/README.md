This work is taken from MCAN repo (https://github.com/MILVLG/mcan-vqa). Download the pretrained epoch13.pkl file given by the authors.


To test performance using swapmix on objects/attribures of a trained model
```
python3 run_files/run_evaluate.py --CKPT_PATH=/data/b/vipul/pretrained/ckpt_small/epoch13.pkl --GPU='7' --OUTPUT_JSON=<output file path> --TYPE <object/attributes>
```

To test performance using swapmix on objects/attribures of a model trained using perfect sight
```
python3 run_files/run_evaluate.py --CKPT_PATH=/data/b/vipul/pretrained/ckpt_small/epoch13.pkl --GPU='7' --OUTPUT_JSON=<output file path> --TYPE <object/attributes> --FEATURES scene_graph
```

To perform swapmix training
```
python3 run_files/run_swapmix.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id>
```

To perform swapmix training using perfect sight embeddings
```
python3 run_files/run_swapmix.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id> --FEATURES='scene_graph'
```

To finetune pretrained model on GQA dataset
```
python3 run_files/run_train.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id>
```

To perform vanilla training using scene graph embeddings 
```
python3 run_files/run_train.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id> --FEATURES='scene_graph'
```

