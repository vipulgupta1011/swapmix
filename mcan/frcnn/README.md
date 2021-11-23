To finetune fast-rcnn model :
```
python3 run.py --RUN='train' --CKPT_PATH=<path to epoch13.pkl file>
```


To finetune fast-rcnn model with SwapMix:
```
python3 run_swapmix.py --RUN='train' --CKPT_PATH=<path to epoch13.pkl file>
```


To generate object and attribute perturbation results :
```
python3 benchmark_frcnn/run_irrelevant_object.py --RUN='val' --CKPT_PATH=<path to pkl file>  --OUTPUT_JSON=<output json file> --NO_OF_CHANGES=10

To generate results without random selection of objects while generating perturbations :
python3 benchmark_frcnn/run_irrelevant_object.py --RUN='val' --CKPT_PATH=<path to pkl file>  --OUTPUT_JSON=<output json file> --NO_OF_CHANGES=10 --ALLOW_RANDOM=False
```


Benchmarking perturbation results for faster-rcnn:
```
cd ../../scripts
python benchmark_frcnn_combine.py --file <object perturbations json file> --file1 <attr perturbations json file>
```
