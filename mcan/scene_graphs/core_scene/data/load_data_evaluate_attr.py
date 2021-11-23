# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core_scene.data.data_utils_evaluate import img_feat_path_load, img_feat_load, ques_load, tokenize, ans_stat
from core_scene.data.data_utils_evaluate import proc_img_feat, proc_ques, proc_ans

import numpy as np
import glob, json, torch, time
import torch.utils.data as Data
import pdb

from core_scene.scene_graph.gqa_evaluate_attr import GQADataset
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
        self.ques_list = {}
        self.ans_list = {}

        start_time = time()
        split_list = __C.SPLIT[__C.RUN_MODE].split('+')
        for split in split_list:
            if split in ['train','val'] :
                #self.ques_list += json.load(open(__C.GQA_QUESTION_PATH[split], 'r'))['questions']
                self.ques_list = json.load(open(__C.GQA_SMALL_QUESTION_PATH[split], 'r'))
                #if __C.RUN_MODE in ['train']:
                if __C.RUN_MODE in ['train']:
                    self.ans_list += json.load(open(__C.ANSWER_PATH[split], 'r'))['annotations']
                if __C.RUN_MODE in ['val']:
                    self.ans_list = json.load(open(__C.SMALL_ANS_PATH, 'r'))

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
            ''' This needs to be debug '''
            for split in split_list:
                if split in ['train'] :
                    GQA = GQADataset(self.__C.TRAIN_SCENE_GRAPH, self.__C.GQA_VOCAB, self.__C.GQA_EMBEDDING)
                    self.iid_to_img_feat = GQA.extractembeddings(__C.IMG_FEAT_PATH[split], self.iid_to_img_feat)
                if split in ['val'] :
                    self.list_images = list(json.load(open(__C.IMG_FEAT_PATH[split])).keys())
                    GQA = GQADataset(self.__C.VAL_SCENE_GRAPH, self.__C.GQA_VOCAB, self.__C.GQA_EMBEDDING, self.__C.OBJ_MATCHING, self.__C.ATTR_MATCHING, self.ques_list, self.__C.SCENE_OBJECT_MATCHING , self.__C.NO_OF_CHANGES)
                    self.iid_to_img_feat = GQA.extractembeddings(self.list_images, self.iid_to_img_feat)

        else : 
        ## all changes needs to be done here
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
        #self.qid_to_ques = ques_load(self.ques_list)

        start_time = time()
        # Tokenize
        self.token_to_ix, self.pretrained_emb = tokenize(self.stat_ques_list, __C.USE_GLOVE)
        self.token_size = self.token_to_ix.__len__()
        #self.token_size = 20572
        print('== Question token vocab size:', self.token_size)
        print ('Tokenize time : ', time() - start_time)

        # Answers statistic
        # Make answer dict during training does not guarantee
        # the same order of {ans_to_ix}, so we published our
        # answer dict to ensure that our pre-trained model
        # can be adapted on each machine.

        # Thanks to Licheng Yu (https://github.com/lichengunc)
        # for finding this bug and providing the solutions.

        start_time = time()
        self.ans_to_ix, self.ix_to_ans = ans_stat(self.stat_ans_list, __C.ANS_FREQ)
        #self.ans_to_ix, self.ix_to_ans = ans_stat('core/data/answer_dict.json')
        self.ans_size = self.ans_to_ix.__len__()
        print('== Answer vocab size (occurr more than {} times):'.format(8), self.ans_size)
        print('Finished!')
        print('')
        print ('end time : ', time() - start_time)


    def extract(self, idx):
    #def loading(self, idx):

        # For code safety
        #img_feat_iter = np.zeros(1)
        #ques_ix_iter = np.zeros(1)
        #ans_iter = np.zeros(1)

        img = self.list_images[idx]
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
                #img_feat_x = img_feat['x'].transpose((1, 0))
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
            answers = self.ans_list[img]
            questions = self.ques_list[img]
            #ques = self.qid_to_ques[str(ans['question_id'])]

            img_feat_obj_iter = {}
            img_feat_attr_iter = {}
            img_feat_bbox = {}
            relevant_questions = {}
            changes = {}
            ques_ix_iter = {}
            ans_iter = {}

            img_feat_x = self.iid_to_img_feat[img]

            for obj_iter in img_feat_x :
                img_feat_obj_iter[obj_iter] = []
                img_feat_attr_iter[obj_iter] = []
                img_feat_x_dict = img_feat_x[obj_iter]['objandattr']
                try :
                    for l in range(len(img_feat_x_dict[obj_iter]['obj_embed'])) :
                        img_feat_obj = []
                        img_feat_attr = []
                        ## creating an object and attribute list for each change
                        ## Object of interest will have a list of object embeddings and other will have single embedding
                        for obj_no in img_feat_x_dict :
                            obj = img_feat_x_dict[obj_no]
                            if obj_no == obj_iter :
                                obj_embed = obj['obj_embed'][l]
                            else :
                                try :
                                    obj_embed = obj['obj_embed']
                                except :
                                    obj_embed = np.zeros(300)
                            img_feat_obj.append(obj_embed)

                            ## Getting attributes for each object and taking average of them to convert them to 300x1
                            if obj_no == obj_iter : 
                                attr_embed = obj['attr_embed'][l]
                            else :
                                attr_embed = obj['attr_embed']
                            if len(attr_embed) == 0:
                                attr_concat = obj_embed
                            else:
                                attr_concat = attr_embed[0]
                                if len(attr_embed) > 1 :
                                    for i in range(1,len(attr_embed)) :
                                        attr_concat = np.add(attr_concat,attr_embed[i])
                                    attr_concat = attr_concat/len(attr_embed)

                            img_feat_attr.append(attr_concat)
                        img_feat_obj = np.array(img_feat_obj)
                        img_feat_attr = np.array(img_feat_attr)
                        img_feat_obj_single = proc_img_feat(img_feat_obj, self.__C.IMG_FEAT_PAD_SIZE)
                        img_feat_attr_single = proc_img_feat(img_feat_attr, self.__C.IMG_FEAT_PAD_SIZE)

                        img_feat_obj_iter[obj_iter].append(img_feat_obj_single)
                        img_feat_attr_iter[obj_iter].append(img_feat_attr_single)

                except :
                    pass
                changes[obj_iter] = img_feat_x[obj_iter]['changes']

                relevant_questions[obj_iter] = img_feat_x[obj_iter]['relevant_questions']
                ## extracting bounding box coordinates in same order as objects to maintain symmetry
                img_feat_bbox_single = []
                for obj_box in img_feat_x :
                    if obj_box in img_feat_x[obj_iter]['bboxes'] :
                        img_feat_bbox_single.append(img_feat_x[obj_iter]['bboxes'][obj_box])
                img_feat_bbox_single = np.asarray(img_feat_bbox_single)

                try :
                    img_feat_bbox_single = proc_img_feat(img_feat_bbox_single, self.__C.IMG_FEAT_PAD_SIZE) 
                except :
                    img_feat_bbox_single = np.zeros((100,4))
                img_feat_bbox[obj_iter] = img_feat_bbox_single
            for ques in questions :
                ques_ix_iter[ques] = proc_ques(questions[ques], self.token_to_ix, self.__C.MAX_TOKEN)
                ans_iter[ques] = proc_ans(answers[ques], self.ans_to_ix)
            

        return img_feat_obj_iter, \
               img_feat_attr_iter, \
               ques_ix_iter, \
               ans_iter, \
               img_feat_bbox, \
               changes, \
               relevant_questions, \
               img



    def __len__(self):
        return self.data_size


