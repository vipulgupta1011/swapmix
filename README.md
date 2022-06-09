# SwapMix

Implementation of SwapMix approach to measure visual bias for visual question answering([SwapMix: Diagnosing and Regularizing the Over-Reliance on Visual Context in Visual Question Answering, Vipul et al., CVPR22](https://arxiv.org/abs/2204.02285))

![Motivation_new](https://user-images.githubusercontent.com/42180235/162658751-74ab2baf-2133-499c-8433-c44d5a2b4a4b.png)


## Introduction
We provide a new way to benchmark in a VQA model by perturbing the visual context i.e. irrelevant objects in the image.

The model looks at an image and a question. Then we change the visual context (irrelevant objects to the question) in the image. For each question we make multiple copies of image by changing context. Ideally, we would expect the model's prediction to remain consistent with context switch.

In this repo, we apply SwapMix to [MCAN](https://github.com/MILVLG/mcan-vqa) and [LXMERT](https://github.com/airsplay/lxmert). We use [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) dataset for our analysis.


## Implementation Details
The code has been divided into [MCAN](https://github.com/vipulgupta1011/swapmix/tree/master/mcan#readme) and [LXMERT](https://github.com/vipulgupta1011/swapmix/tree/master/lxmert#readme) folders. Inside each folder we provide implementation for 
1. Measuring visual bias using SwapMix
2. Finetuning models using SwapMix as data augmentation technique 
3. Training model with perfect sight.

Perfect sight implementation differs from traditional FasterRCNN at net.py(input embedding layer is changed frmo 1024 dim to 256 dim) and swapping part. Rest all the code remains the same.

## Download Dataset

We restructured the format of question, answer, and scene graph files provided by [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) a bit. You can download these files along with other files needed for SwapMix implementation from [here](https://drive.google.com/file/d/1Zas1Nag3n-ipvNYW_zSkL7Ipo0ap8aj_/view?usp=sharing) and place it at <code>data/gqa</code> folder.
 
We recommend to use object features provided by GQA. Download the features from [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html)


## Evaluation

We measure visual bias of the model for both irrelevant object changes and attribute changes seperately. We combine the results for both changes to measure visual bias of the model

Before benchmarking visual bias for these models, we finetune them on GQA train dataset for better performance. Models are evaluated on GQA val set.

We provide (1) finetuned model (2) model finetuned using SwapMix as data augmentation (3) model trained with perfect sight (4) model trained with perfect sight and using SwapMix as data augmentation technique. Please download the models from here : [MCAN trained models](https://drive.google.com/drive/folders/1PJmj2fnNM-ixoD4v54GEkRl0Uquuc8QT?usp=sharing), [LXMERT trained models](https://drive.google.com/drive/folders/1t0dfYG3A0YvFFvpHXhLEmugpu95Lbl0f?usp=sharing)

To measure context reliance after calculating object and attribute results :
```
cd scripts
python benchmark_frcnn.py --obj <SwapMix object json file>   --attr <SwapMix attribute json file>
```

## Evaluating new model for visual bias
SwapMix can be used to measure visual bias on any VQA model

Changes are needed on data loading and testing part. The current code iterates over each question indiviually to get predictions for all SwapMix perturbations. 

Details for measuring visual bias on a new model can be found [here](https://github.com/vipulgupta1011/swapmix/tree/master/swapmix)


## Citation
If you like our work and find this code useful, consider citing our work :
```
@inproceedings{gupta2022swapmix,
    title={SwapMix: Diagnosing and Regularizing the Over-Reliance on Visual Context in Visual Question Answering},
    author={Gupta, Vipul and Li, Zhuowan and Kortylewski, Adam and Zhang, Chenyu and Li, Yingwei and Yuille, Alan},
    booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    year={2022}
}
```


## References
- [Deep Modular Co-Attention Networks for Visual Question Answering](https://openaccess.thecvf.com/content_CVPR_2019/html/Yu_Deep_Modular_Co-Attention_Networks_for_Visual_Question_Answering_CVPR_2019_paper.html), Zhou et al., CVPR 2019
- [LXMERT: Learning Cross-Modality Encoder Representations from Transformers](https://aclanthology.org/D19-1514.pdf), Hao et  al., EMNLP 2019
- [VQA : Visual Question Answering](http://arxiv.org/abs/1505.00468), Antol et al., ICCV15

