Implementation of SwapMix approach to measure visual bias for visual question answering

The model looks at an image and a question. Then we change the visual context (irrelevant objects to the question) in the image. For each question we make multiple copies of image by changing context. Ideally, we would expect the model's prediction to remain consistent with context switch.

In this repo, we apply SwapMix to [MCAN](https://github.com/MILVLG/mcan-vqa) and [LXMERT](https://github.com/airsplay/lxmert). We use [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) dataset for our analysis.}

The code has been divided into mcan and lxmert folders. Inside each folder we provide implementation for 
1. Measuring visual bias using SwapMix
2. Finetuning models using SwapMix as data augmentation technique 
3. Training model with perfect sight.

We modified the questions and answers file, and scene graph files provided at [GQA](https://cs.stanford.edu/people/dorarad/gqa/download.html) site. You can download these files along with other files needed for SwapMix implementation from [here](https://drive.google.com/file/d/1Zas1Nag3n-ipvNYW_zSkL7Ipo0ap8aj_/view?usp=sharing) and place it at <code>.


Initialize a virtual envirnoment using the requirements.txt file in each folder

Download the questions and annotation files provided by https://github.com/MILVLG/mcan-vqa and place them in "data/vqa" folder

We took the train questions and answers dataset from https://cs.stanford.edu/people/dorarad/gqa/download.html. We restructed the format a bit. The restructed files, along with other useful mapping can be downloaded from "here"

Download the objectfeatures.zip file from https://cs.stanford.edu/people/dorarad/gqa/download.html and put the extracted files in "data/obj_features"
