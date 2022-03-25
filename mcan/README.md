This work is taken from MCAN repo (https://github.com/MILVLG/mcan-vqa). Download the pretrained epoch13.pkl file given by the authors.

To test performance using swapmix on attributes of a trained model 
```
python3 benchmark_frcnn/run_evaluate.py --CKPT_PATH=/data/b/vipul/pretrained/ckpt_small/epoch13.pkl --GPU='7' --OUTPUT_JSON=/data/b/vipul/output_vqa/results/fastrcnn/irrelevant_objects_temp.json --TYPE attributes
```

To test performance using swapmix on objects of a trained model 
```
python3 benchmark_frcnn/run_evaluate.py --CKPT_PATH=/data/b/vipul/pretrained/ckpt_small/epoch13.pkl --GPU='7' --OUTPUT_JSON=/data/b/vipul/output_vqa/results/fastrcnn/irrelevant_objects_temp.json --TYPE objects
```

To perform swapmix training using FasterRCNN features 
```
python3 run_files/run_swapmix.py --RUN='train' --GPU='7' --FEATURES='frcnn' --CKPT_PATH=/data/b/vipul/pretrained/ckpt_small/epoch13.pkl
```

To perform swapmix training using scene graph embeddings 
```
python3 run_files/run_swapmix.py --RUN='train' --GPU='7' --FEATURES='scene_graph' --CKPT_PATH=/data/b/vipul/pretrained/ckpt_small/epoch13.pkl
```

To perform vanilla training using FasterRCNN features 
```
python3 run_files/run_train.py --RUN='train' --CKPT_PATH=/data/b/vipul/pretrained/ckpt_small/epoch13.pkl --GPU='7' --VERSION=temp --FEATURES='frcnn'
```

To perform vanilla training using scene graph embeddings 
```
 python3 run_files/run_train.py --RUN='train' --CKPT_PATH=/data/b/vipul/pretrained/ckpt_small/epoch13.pkl --GPU='7' --VERSION=temp --FEATURES='scene_graph'
```

