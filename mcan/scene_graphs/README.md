To train the scene graphs model :
```
python3 run_scene.py --RUN='train'
```

To train the scene grapsh model using SwapMix :
```
python3 run_scene_swapmix.py --RUN='train'
```

To generate object and attribute perturbation results :

```
python3 benchmark_scene/run_attr_unrelevant.py --RUN='val' --CKPT_PATH=<pkl path> --OUTPUTJSON=<output json path> 

python3 benchmark_scene/run_obj_unrelevant.py --RUN='val' --CKPT_PATH=<pkl path> --OUTPUT_JSON=<output json path>

```


Benchmark results for scene graphs :
```
cd ../../scripts
python benchmark_scene_combine.py --file <object perturbation file> --file1 <attribute perturbation file>
```

