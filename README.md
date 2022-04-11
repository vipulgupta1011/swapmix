# SwapMix

Implementation of SwapMix approach to measure visual bias for visual question answering([SwapMix: Diagnosing and Regularizing the Over-Reliance on Visual Context in Visual Question Answering, Vipul et al., CVPR22](https://arxiv.org/abs/2204.02285))

![img](https://photos.app.goo.gl/VAZT18nDVPNyiUoP8)


The model looks at an image and a question. Then we change the visual context (irrelevant objects to the question) in the image. For each question we make multiple copies of image by changing context. Ideally, we would expect the model's prediction to remain consistent with context switch.

In this repo, we apply SwapMix to [MCAN](https://github.com/MILVLG/mcan-vqa) and [LXMERT](https://github.com/airsplay/lxmert). We use [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) dataset for our analysis.

The code has been divided into mcan and lxmert folders. Inside each folder we provide implementation for 
1. Measuring visual bias using SwapMix
2. Finetuning models using SwapMix as data augmentation technique 
3. Training model with perfect sight.

We also provide the model trained using SwapMix as data augmentation and trained with perfect sight. Link to the models is provided inside each folder.

We restructured the format of question, answer, and scene graph files provided by [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) a bit. You can download these files along with other files needed for SwapMix implementation from [here](https://drive.google.com/file/d/1Zas1Nag3n-ipvNYW_zSkL7Ipo0ap8aj_/view?usp=sharing) and place it at <code>data/gqa</code> folder. 


