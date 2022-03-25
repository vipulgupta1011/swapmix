# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core.data.data_utils import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat_old, ans_stat
from core.data.data_utils import proc_img_feat, proc_ques, proc_ans

import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
import pdb

from core.dataloader.gqa_scene_train import GQADataset
from time import time


class DataSet(Data.Dataset):
    def __init__(self, __C):
        self.__C = __C

        # --------------------------
        # ---- Raw data loading ----
        # --------------------------

        # Loading all image paths
        # if self.__C.PRELOAD:
        self.img_feat_path_list = []
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        ## split_list = ['train', 'val', 'vg']
        #pdb.set_trace()
        #for split in split_list:
        #    #if split in ['train', 'test']:
        #    #    self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH[split] + '*.npz')
        #    if split in ['train', 'test', 'val'] :
        #        self.img_feat_path_list = self.img_feat_path_list.append(__C.IMG_FEAT_PATH[split])

        # if __C.EVAL_EVERY_EPOCH and __C.RUN_MODE in ['train']:
        #     self.img_feat_path_list += glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz')

        # else:
        #     self.img_feat_path_list = \
        #         glob.glob(__C.IMG_FEAT_PATH['train'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['val'] + '*.npz') + \
        #         glob.glob(__C.IMG_FEAT_PATH['test'] + '*.npz')

        # Loading question word list
        start_time = time()
        self.stat_ques_list = \
            json.load(open(__C.GQA_QUESTION_PATH['train'], 'r'))['questions'] + \
            json.load(open(__C.GQA_QUESTION_PATH['val'], 'r'))['questions'] 
            #json.load(open(__C.QUESTION_PATH['test'], 'r'))['questions'] + \
            #json.load(open(__C.QUESTION_PATH['vg'], 'r'))['questions']
        print ('stat_ques_list time : ', time() - start_time)

        start_time = time()
        # Loading answer word list
        self.stat_ans_list = \
            json.load(open(__C.ANSWER_PATH['train'], 'r'))['annotations'] + \
            json.load(open(__C.ANSWER_PATH['val'], 'r'))['annotations']
        #pdb.set_trace()
        print ('stat_ans_list time : ', time() - start_time)

        # Loading question and answer list
        self.ques_list = []
        self.ans_list = []

        start_time = time()
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ['train','val'] :
                self.ques_list += json.load(open(__C.GQA_QUESTION_PATH[split], 'r'))['questions']
                #self.ques_list += json.load(open(__C.GQA_QUESTION_PATH_NEW[split], 'r'))
                if __C.RUN_MODE in ['train']:
                    self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']
                if __C.RUN_MODE in ['val']:
                    self.ans_list += json.load(open(__C.VAL_ANS_PATH, 'r'))

        print ('ans list time : ', time() - start_time)

        # Define run data size
        if __C.RUN_MODE in ['train']:
            self.data_size = self.ans_list.__len__()
        else:
            self.data_size = self.ques_list.__len__()

        print('== Dataset size:', self.data_size)


        # ------------------------
        # ---- Data statistic ----
        # ------------------------

        # {image id} -> {image feature absolutely path}

        start_time = time()
        self.iid_to_img_feat = {}
        if self.__C.IS_GQA :
            for split in split_list:
                if split in ['train'] :
                    GQA = GQADataset(self.__C.TRAIN_SCENE_GRAPH, self.__C.GQA_VOCAB, self.__C.GQA_EMBEDDING)
                    self.iid_to_img_feat = GQA.extractembeddings(__C.IMG_FEAT_PATH[split], self.iid_to_img_feat)
                if split in ['val'] :
                    GQA = GQADataset(self.__C.VAL_SCENE_GRAPH, self.__C.GQA_VOCAB, self.__C.GQA_EMBEDDING)
                    self.iid_to_img_feat = GQA.extractembeddings(__C.IMG_FEAT_PATH[split], self.iid_to_img_feat)

        else : 
            if self.__C.PRELOAD:
                print('==== Pre-Loading features ...')
                time_start = time.time()
                self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
                time_end = time.time()
                print('==== Finished in {}s'.format(int(time_end-time_start)))
            else:
                self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        print ('GQA loading time : ', time() - start_time)
        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

        start_time = time()
        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        #self.token_size = 20572
        print('== Question token vocab size:', self.token_size)
        print ('Tokenize time : ', time() - start_time)

        start_time = time()
        self.ans_to_ix, self.ix_to_ans = ans_stat_old(self.stat_ans_list, __C.ANS_FREQ)
        #self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')
        print ('end time : ', time() - start_time)


    def __getitem__(self, idx):

        # For code safety
        img_feat_iter = np.zeros(1)
        ques_ix_iter = np.zeros(1)
        ans_iter = np.zeros(1)

        # Process ['train'] and ['val', 'test'] respectively
        if self.__C.RUN_MODE in ['train']:
            # Load the run data from list
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]

            # Process image feature from (.npz) file
            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            else:
                #img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]

            img_feat_obj = []
            img_feat_attr = []
            for obj in img_feat_x['objandattr'].values() :
                try :
                    obj_embed = obj['obj_embed']
                except :
                    obj_embed = np.zeros(300)
                img_feat_obj.append(obj_embed)

                try :
                    attr_embed = obj['attr_embed']
                    if len(attr_embed) == 0:
                        #attr_concat = np.zeros(300)
                        attr_concat = obj_embed
                    else:
                        attr_concat = attr_embed[0]
                        if len(attr_embed) > 1 :
                            for i in range(1,len(attr_embed)) :
                                attr_concat = np.add(attr_concat,attr_embed[i])
                            attr_concat = attr_concat/len(attr_embed)
                except :
                    #attr_concat = np.zeros(300)
                    attr_concat = obj_embed

                img_feat_attr.append(attr_concat)
            img_feat_obj = np.array(img_feat_obj)
            img_feat_obj_iter = proc_img_feat(img_feat_obj, self.__C.IMG_FEAT_PAD_SIZE)
            img_feat_attr = np.array(img_feat_attr)
            img_feat_attr_iter = proc_img_feat(img_feat_attr, self.__C.IMG_FEAT_PAD_SIZE)
            #img_feat_iter = img_feat_x['objandattr']

            img_feat_bbox = np.asarray(img_feat_x['bboxes'])
            try :
                img_feat_bbox = proc_img_feat(img_feat_bbox, self.__C.IMG_FEAT_PAD_SIZE) 
            except :
                img_feat_bbox = np.zeros((100,4))
            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)

        else:
            # Load the run data from list
            #ques = self.ques_list[idx]
            ans = self.ans_list[idx]
            ques = self.qid_to_ques[str(ans['question_id'])]

            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            else:
                img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]

            # Process image feature
            pdb.set_trace()
            img_feat_iter = proc_img_feat(img_feat_x['objandattr'], self.__C.IMG_FEAT_PAD_SIZE)
            #img_feat_iter = img_feat_x['objandattr']
            img_feat_bbox = np.asarray(img_feat_x['bboxes'])
            try :
                img_feat_bbox = proc_img_feat(img_feat_bbox, self.__C.IMG_FEAT_PAD_SIZE) 
            except :
                img_feat_bbox = np.zeros((100,4))

            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)


        return torch.from_numpy(img_feat_obj_iter), \
               torch.from_numpy(img_feat_attr_iter), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter), \
               torch.from_numpy(img_feat_bbox)


    def __len__(self):
        return self.data_size


