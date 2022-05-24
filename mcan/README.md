This folder contains SwapMix implementation on [MCAN](https://github.com/MILVLG/mcan-vqa). 

## Dependencies and dataset
Install the requirements dependencies given by authors of MCAN and provided in the requirement.txt file.

Download the questions and annotations provided by MCAN or you can download it from [here](https://drive.google.com/file/d/1wTCRC1wM8pGju_Krd1f1s-Gv38P2ck56/view?usp=sharing) and extract it at <code>data/vqa</code>.

Download the object features provided by [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) and place them at <code>data/obj_features</code>.

## Download models
Download the pretrained epoch13.pkl model given by the authors. This model gives bad performance as it has not been trained on GQA train set.  We provide (1) finetuned model (2) model finetuned using SwapMix as data augmentation (3) model trained with perfect sight (4) model trained with perfect sight and using SwapMix as data augmentation technique. Please download the models from[MCAN trained models](https://drive.google.com/drive/folders/1PJmj2fnNM-ixoD4v54GEkRl0Uquuc8QT?usp=sharing)

We finetune models over the pretrained epoch13.pkl model.


## Measuring visual bias
We measure context reliance on irrelevant objects and attributes. 

For calculating context reliance on irrelevant objects, we swap the object with anotherobject of different class. For example, swapping bus with car. To calculate context reliance on attributes, we swap the irrelevant object with object of same class but with differnt attributes. For example, swapping blue bus with red bias.

To measure context reliance based on objects/attribures of a trained model
```
python3 run_files/run_evaluate.py --CKPT_PATH=<path to ckpt file> --GPU=<gpu id> --OUTPUT_JSON=<output file path> --TYPE <object/attributes>
```

To measure context reliance based on objects/attribures of a model trained using perfect sight embeddings
```
python3 run_files/run_evaluate.py --CKPT_PATH=<path to ckpt file> --GPU=<gpu id> --OUTPUT_JSON=<output file path> --TYPE <object/attributes> --FEATURES scene_graph
```


## Training using SwapMix
We also finetune models using SwapMix as data augmentation technique and show that context reliance of the model decreases and effective accuracy increases. Use below commands to train a model using SwapMix.

To finetune a model using SwapMix as data augmentation
```
python3 run_files/run_swapmix.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id>
```

To finetune a model using SwapMix as data augmentation using perfect sight embeddings
```
python3 run_files/run_swapmix.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id> --FEATURES='scene_graph'
```


## Finetuning on GQA dataset 
To finetune the pretrained model provided by authors on GQA dataset. This is done to improve the accuracy of the model on GQA val set.
```
python3 run_files/run_train.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id>
```

To perform training using perfect sight embeddings
```
python3 run_files/run_train.py --CKPT_PATH=<pretrained ckpt path> --GPU=<gpu id> --FEATURES='scene_graph'
```

