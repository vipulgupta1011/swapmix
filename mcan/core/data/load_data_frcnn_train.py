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

        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            self.ques_list += json.load(open(__C.GQA_QUESTION_PATH[split], 'r'))['questions']
            #self.ques_list = json.load(open(__C.GQA_QUESTION_PATH_NEW[split], 'r'))
            if __C.RUN_MODE in ['train']:
                self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']
            if __C.RUN_MODE in ['val']:
                self.ans_list = json.load(open(__C.VAL_ANS_PATH, 'r'))

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
        #pdb.set_trace()
        start_time = time()
        #self.iid_to_img_feat = {}
        #if self.__C.IS_GQA :
        #    ''' This needs to be debug '''
        #    for split in split_list:
        #        if split in ['train'] :
        #            GQA = GQADataset(self.__C.TRAIN_SCENE_GRAPH, self.__C.GQA_EMBEDDING)
        #            self.iid_to_img_feat = GQA.extractembeddings(__C.IMG_FEAT_PATH[split], self.iid_to_img_feat)
        #        if split in ['val'] :
        #            self.list_images = list(json.load(open(__C.IMG_FEAT_PATH[split])).keys())
        #            GQA = GQADataset(self.__C.VAL_SCENE_GRAPH, self.ques_list)
        #            self.iid_to_img_feat = GQA.extractembeddings(self.list_images, self.iid_to_img_feat)

        #else : 
        #    if self.__C.PRELOAD:
        #        print('==== Pre-Loading features ...')
        #        time_start = time.time()
        #        self.iid_to_img_feat = img_feat_load(self.img_feat_path_list)
        #        time_end = time.time()
        #        print('==== Finished in {}s'.format(int(time_end-time_start)))
        #    else:
        #        ''' This is being called -> list of paths '''
        #        self.iid_to_img_feat_path = img_feat_path_load(self.img_feat_path_list)

        # {question id} -> {question}
        self.qid_to_ques = ques_load(self.ques_list)

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

            # Process image feature from (.npz) file
            #if self.__C.PRELOAD:
            #    img_feat_x = self.iid_to_img_feat[str(ans['image_id'])]
            #else:
            #    #img_feat = np.load(self.iid_to_img_feat_path[str(ans['image_id'])])
            #    #img_feat_x = img_feat['x'].transpose((1, 0))
            #    img_feat_x = self.iid_to_img_feat[str(ques['image_id'])]
            #img_feat_iter = proc_img_feat(img_feat_x['objandattr'], self.__C.IMG_FEAT_PAD_SIZE)
            #img_feat_iter = proc_img_feat(img_feat_x, self.__C.IMG_FEAT_PAD_SIZE)

            ## Extracting fastrcnn feature
            feature = self.frcnn.get_feature_single(img)
            # Process question
            ques_ix_iter = proc_ques(ques, self.token_to_ix, self.__C.MAX_TOKEN)

            # Process answer
            ans_iter = proc_ans(ans, self.ans_to_ix)

        else:
            # Load the run data from list
            #ques = self.ques_list[idx]
            answers = self.ans_list[img]
            questions = self.ques_list[img]

            if self.__C.PRELOAD:
                img_feat_x = self.iid_to_img_feat[img]
            else:
                img_feat_x = self.iid_to_img_feat[img]
            #img_feat_iter = {}
            #img_feat_bbox = {}
            relevant_questions = {}
            names = {}
            bbox = {}
            #changes = {}
            ques_ix_iter = {}
            ans_iter = {}

            for obj in img_feat_x :
                #img_8ddfeat_iter[obj] = []
                #img_feat_x_list = img_feat_x[obj]['objandattr']
                #for i in range(len(img_feat_x_list)) :
                #    img_feat_x_ = img_feat_x_list[i]
                #    img_feat_iter_single = proc_img_feat(img_feat_x_, self.__C.IMG_FEAT_PAD_SIZE)
                #    img_feat_iter[obj].append(img_feat_iter_single)
                #img_feat_iter = img_feat_x['objandattr']
                #changes[obj] = img_feat_x[obj]['changes']

                relevant_questions[obj] = img_feat_x[obj]['relevant_questions']
                names[obj] = img_feat_x[obj]['name']
                bbox[obj] = img_feat_x[obj]['bbox']

                #img_feat_bbox_single = np.asarray(img_feat_x[obj]['bboxes'])
                #try :
                #    img_feat_bbox_single = proc_img_feat(img_feat_bbox_single, self.__C.IMG_FEAT_PAD_SIZE) 
                #except :
                #    img_feat_bbox_single = np.zeros((100,4))
                #img_feat_bbox[obj] = img_feat_bbox_single
            
            # Process question and answer
            for ques in questions :
                ques_ix_iter[ques] = proc_ques(questions[ques], self.token_to_ix, self.__C.MAX_TOKEN)
                ans_iter[ques] = proc_ans_modify(answers[ques], self.ans_to_ix)

            # Process answer
            #ans_iter_temp = {}
            #ans_iter_temp[ans['question_id']] = proc_ans(ans, self.ans_to_ix)
            #ans_iter.append(ans_iter_temp)


        return torch.from_numpy(feature), \
               torch.from_numpy(ques_ix_iter), \
               torch.from_numpy(ans_iter)



    def __len__(self):
        return self.data_size


