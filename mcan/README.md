This folder contains SwapMix implementation on [MCAN](https://github.com/MILVLG/mcan-vqa). 

Install the requirements dependencies given by authors of MCAN and provided in the requirement.txt file.

Download the questions and annotations provided by MCAN or you can download it from [here](https://drive.google.com/file/d/1wTCRC1wM8pGju_Krd1f1s-Gv38P2ck56/view?usp=sharing) and extract it at <code>data/vqa</code>.

Download the object features provided by [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) and place them at <code>data/obj_features</code>.

## Use a pretrained model
Download the pretrained epoch13.pkl file given by the authors.


## Measuring visual bias
We measure visual bias based on irrelevant objects and attributes.

To measure visual bias based on objects/attribures of a trained model
```
python3 run_files/run_evaluate.py --CKPT_PATH=<path to ckpt file> --GPU=<gpu id> --OUTPUT_JSON=<output file path> --TYPE <object/attributes>
```

To measure visual bias based on objects/attribures of a model trained using perfect sight embeddings
```
python3 run_files/run_evaluate.py --CKPT_PATH=<path to ckpt file> --GPU=<gpu id> --OUTPUT_JSON=<output file path> --TYPE <object/attributes> --FEATURES scene_graph
```


## Training using SwapMix
We also finetune models using SwapMix as data augmentation technique and show that context reliance of the model decreases and effective accuracy increases.

To finetune a model using SwapMix as data augmentation
```
python3 run_files/run_swapmix.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id>
```

To finetune a model using SwapMix as data augmentation using perfect sight embeddings
```
python3 run_files/run_swapmix.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id> --FEATURES='scene_graph'
```


## Finetuning on GQA dataset 
To finetune the pretrained model provided by authors on GQA dataset
```
python3 run_files/run_train.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id>
```

To perform training using scene graph embeddings 
```
python3 run_files/run_train.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id> --FEATURES='scene_graph'
```

