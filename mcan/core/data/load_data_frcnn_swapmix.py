# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans, proc_ans_modify
from core.dataloader.frcnn_modify import FRCNN
import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
import pdb

from time import time
from core.dataloader.gqa_frcnn import GQADataset


class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        # if self.__C.PRELOAD:
        self.frcnn = FRCNN(__C)
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        ## split_list = ['train', 'val', 'vg']
        #for split in split_list:
        #    if split in ['train', 'val', 'test']:
        #        self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')

        # if __C.EVAL_EVERY_EPOCH and __C.RUN_MODE in ['train']:
        #     self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz')

        # else:
        #     self.img_feat_path_list = \
        #         glob.glob(__C.IMG_FEAT_PATH['train'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['test'] + '*.npz')

        # Loading question word list
        self.stat_ques_list = \
            json.load(open(__C.QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['val'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']

        # Loading answer word list
        self.stat_ans_list = \
            json.load(open(__C.ANSWER_PATH['train'], 'r'))['annotations'] + \
            json.load(open(__C.ANSWER_PATH['val'], 'r'))['annotations']

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []
        self.val_ques_list = []
        self.val_ans_list = []
        
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            ## Adding new question list to create mapping between questions and irrelated objects
            if split in ['train'] :
                self.ques_list = json.load(open(__C.GQA_QUESTION_PATH[split], 'r'))['questions']
                self.ques_list_new = json.load(open(__C.GQA_QUESTION_PATH_NEW[split], 'r'))
                self.ans_list = json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']
            if split in ['val'] :
                self.val_ques_list = json.load(open(__C.GQA_QUESTION_PATH[split], 'r'))['questions']
                self.val_ans_list = json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']

            #if __C.RUN_MODE in ['val']:
            #    self.ans_list = json.load(open(__C.VAL_ANS_PATH, 'r'))

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.val_ques_list.__len__()

        print('== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}
        start_time = time()
        self.img_to_unrelevant_objs = {}
        if self.__C.IS_GQA :
            for split in split_list:
                if split in ['train'] :
                    self.list_images = list(json.load(open(__C.IMG_FEAT_PATH[split])).keys())
                    GQA = GQADataset(self.__C.TRAIN_SCENE_GRAPH, self.ques_list_new)
                    self.img_to_unrelevant_objs = GQA.extractembeddings_questions(self.list_images, self.img_to_unrelevant_objs)
                '''
                if split in ['val'] :
                    self.list_images = list(json.load(open(__C.IMG_FEAT_PATH[split])).keys())
                    GQA = GQADataset(self.__C.VAL_SCENE_GRAPH, self.ques_list)
                    self.iid_to_img_feat = GQA.extractembeddings(self.list_images, self.iid_to_img_feat)
                '''

        print ('GQA extraction time : ', time() - start_time)
        self.qid_to_ques = ques_load(self.ques_list)
        self.val_qid_to_ques = ques_load(self.val_ques_list)

        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        print('== Question token vocab size:', self.token_size)

        #self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')


    def __getitem__(self, idx):
    #def extract(self, idx):

        # For code safety
        feature = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            ans = self.ans_list[idx]
            img = ans['image_id']

            ques = self.qid_to_ques[str(ans['question_id'])]

            unrelevant_objs = self.img_to_unrelevant_objs[img][ques['question_id']]['unrelevant_objects']

            '''
            Get unrelevant objects for each question
            Perform perturbations in the faster rcnn-features with probability 0.5 for each unrelevant object
            '''

            ## Extracting fastrcnn feature
            feature = self.frcnn.get_feature_swapmix(img, unrelevant_objs)
            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)

        else:
            ans = self.val_ans_list[idx]
            img = ans['image_id']

            ques = self.val_qid_to_ques[str(ans['question_id'])]

            #unrelevant_objs = self.img_to_unrelevant_objs[img][ques['question_id']]['unrelevant_objects']

            ## Extracting fastrcnn feature
            feature = self.frcnn.get_feature_single(img)
            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)

        return torch.from_numpy(feature), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter)



    def __len__(self):
        return self.data_size


