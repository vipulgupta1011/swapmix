This folder contains code for measuring visual bias on [LXMERT](https://github.com/airsplay/lxmert). 


## Dependencies and dataset
Install the requirements provided in the requirement.txt file.

For this model, we use the same faster-rcnn feature file provided by the authors of the paper. Download the feature file and place it at <code>data/obj_features</code>

## Download models
You can download the model provided by LXMERT author.

Additionally, we provide (1) finetuned model using GQA train dataset (2) model finetuned using SwapMix as data augmentation (3) model trained with perfect sight (4) model trained with perfect sight and using SwapMix as data augmentation technique. These models can be downloaded from [LXMERT trained models](https://drive.google.com/drive/folders/1t0dfYG3A0YvFFvpHXhLEmugpu95Lbl0f?usp=sharing).

## Measuring context reliance

We measure context reliance based on irrelevant objects and attributes. 

For calculating context reliance on irrelevant objects, we swap the object with another object of different class. For example, swapping bus with car. To calculate context reliance on attributes, we swap the irrelevant object with object of same class but with differnt attributes. For example, swapping blue bus with red bias.

To measure context reliance based on objects/attribures of a trained model
```
bash run/frcnn_evaluate.bash <gpu id> --load <model path> --test valid --OUTPUT_JSON <output file path> --CLASS <objects/attributes>
```

To measure visual bias of a model trained with perfect sight on object or attribute perturbations 
```
bash run/scene_evaluate.bash <gpu id> --load <model path> --test valid --OUTPUT_JSON <output file path> --CLASS <objects/attributes>
```

## Finetuning on GQA dataset

To finetune model on GQA dataset. This helps in improving the performance of the model on GQA val set
```
bash run/frcnn_train.bash <gpu id> <experiment name>
```

To train the model with perfect sight :
```
bash run/scene_train.bash <gpu id> <experiment name>
```

## Training using SwapMix

We also finetune models using SwapMix as data augmentation technique and show that context reliance of the model decreases and effective accuracy increases.

To train the model using SwapMix as data augmentation
```
bash run/frcnn_train.bash <gpu id> <experiment name> --SwapMix
```

To train the model with perfect sight using SwapMix as data augmentation:
```
bash run/scene_train.bash <gpu id> <experiment name> --SwapMix
```

