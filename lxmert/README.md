This work is built on LXMERT repo (https://github.com/airsplay/lxmert). 

To evaluate model on object or attribute perturbations 
```
bash run/frcnn_evaluate.bash <gpu id> --load <model path> --test valid --OUTPUT_JSON <output file path> --CLASS <objects/attributes>
```

To finetune model on GQA dataset :
```
bash run/frcnn_train.bash <gpu id> <experiment name>
```

To train model using SwapMix :
```
bash run/frcnn_train.bash <gpu id> <experiment name> --SwapMix
```

To evaluate model trained with perfect sight on object or attribute perturbations 
```
bash run/scene_evaluate.bash <gpu id> --load <model path> --test valid --OUTPUT_JSON <output file path> --CLASS <objects/attributes>
```

To train the model with perfect sight :
```
bash run/scene_train.bash <gpu id> <experiment name>
```

To train the model with perfect sight using SwapMix :
```
bash run/scene_train.bash <gpu id> <experiment name> --SwapMix
```

