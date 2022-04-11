This work is built on LXMERT repo (https://github.com/airsplay/lxmert). 

Install the requirements provided in the requirement.txt file.

## Measuring visual bias

We measure visual bias based on irrelevant objects and attributes.

To measure visual bias based on objects/attribures of a trained model
```
bash run/frcnn_evaluate.bash <gpu id> --load <model path> --test valid --OUTPUT_JSON <output file path> --CLASS <objects/attributes>
```

## Finetuning on GQA dataset

To finetune model on GQA dataset :
```
bash run/frcnn_train.bash <gpu id> <experiment name>
```

## Training using SwapMix

We also finetune models using SwapMix as data augmentation technique and show that context reliance of the model decreases and effective accuracy increases.

To finetune a model using SwapMix as data augmentation
```
bash run/frcnn_train.bash <gpu id> <experiment name> --SwapMix
```

## Using perfect sight embeddings
To measure visual bias of a model trained with perfect sight on object or attribute perturbations 
```
bash run/scene_evaluate.bash <gpu id> --load <model path> --test valid --OUTPUT_JSON <output file path> --CLASS <objects/attributes>
```

To train the model with perfect sight :
```
bash run/scene_train.bash <gpu id> <experiment name>
```

To train the model with perfect sight using SwapMix as data augmentation:
```
bash run/scene_train.bash <gpu id> <experiment name> --SwapMix
```

