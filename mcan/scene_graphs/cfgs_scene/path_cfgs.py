# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self):

        # vqav2 dataset root path
        self.DATASET_PATH = '/data/b/vipul/datasets/vqa/'

        # bottom up features root path
        self.FEATURE_PATH = '/data/b/vipul/datasets/coco_extract/'
        self.DIR = '/data/b/vipul/output_vqa/'
        self.GQA_PATH = '/data/b/vipul/datasets/gqa/'

        self.init_path()


    def init_path(self):

        self.IMG_FEAT_PATH = {
            #'train': self.GQA_PATH + 'images_split/small_train/small_train.json',
            'train': self.GQA_PATH + 'images_split/train.json',
            #'val': self.GQA_PATH + 'images_split/small_val.json',
            'val': self.GQA_PATH + 'images_split/val.json',
#            'test': self.FEATURE_PATH + 'images_split/test.json',
        }

        self.QUESTION_PATH = {
            'train': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions.json',
            'val': self.DATASET_PATH + 'v2_OpenEnded_mscoco_val2014_questions.json',
            'test': self.DATASET_PATH + 'v2_OpenEnded_mscoco_test2015_questions.json',
            #'vg': self.DATASET_PATH + 'VG_questions.json',
        }

        self.GQA_QUESTION_PATH = {
            #'train': self.GQA_PATH + 'images_split/small_train/small_train_questions.json',
            'train': self.GQA_PATH + 'images_split/train_questions.json',
            #'val': self.GQA_PATH + 'images_split/small_train/small_val_questions.json',
            'val': self.GQA_PATH + 'images_split/val_questions.json',
        }

        self.GQA_SMALL_QUESTION_PATH = {
            'train': self.GQA_PATH + 'images_split/train_questions_new.json',
            #'val': self.GQA_PATH + 'images_split/small_ques.json',
            'val': self.GQA_PATH + 'images_split/val_questions_new.json',
        }

        self.ANSWER_PATH = {
            #'train': self.GQA_PATH + 'images_split/small_train/small_train_answers.json',
            'train': self.GQA_PATH + 'images_split/train_answers.json',
            'val': self.GQA_PATH + 'images_split/val_answers.json',
            #'val': self.GQA_PATH + 'images_split/small_train/small_val_answers.json',
            #'vg': self.GQA_PATH + 'VG_annotations.json',
        }

        #self.SMALL_ANS_PATH = self.GQA_PATH + 'images_split/small_ans.json'
        self.SMALL_ANS_PATH = self.GQA_PATH + 'images_split/val_answers_new.json'
        self.VAL_SCENE_GRAPH = self.GQA_PATH + 'scene_graphs/val_sceneGraphs.json'
        #self.VAL_SCENE_GRAPH = self.GQA_PATH + 'scene_graphs/small_val_sceneGraphs.json'
        self.TRAIN_SCENE_GRAPH = self.GQA_PATH + 'scene_graphs/train_sceneGraphs.json'
        self.GQA_VOCAB = self.GQA_PATH + 'gqa_vocab_taxo.json'
        self.GQA_EMBEDDING = self.GQA_PATH + 'attrlabel_glove_taxo.npy'
        self.RESULT_PATH = self.DIR + 'results/scene_graphs/'
        self.PRED_PATH = self.DIR + 'results/pred/'
        self.CACHE_PATH = self.DIR + 'results/cache/'
        self.LOG_PATH = self.DIR + 'results/log/'
        self.CKPTS_PATH = self.DIR + 'ckpts/'
        self.OBJ_MATCHING = self.GQA_PATH + 'matching/obj_matching.json'
        self.ATTR_MATCHING = self.GQA_PATH + 'matching/attr_matching.json'
        self.SCENE_OBJECT_MATCHING = self.GQA_PATH + 'scene_graphs/object_mapping.json'
        self.OUTPUT_JSON = self.RESULT_PATH + 'scene_graph_object_perturbations_real.json'

        if 'result_test' not in os.listdir(self.DIR + 'results'):
            os.mkdir(self.DIR + 'results/result_test')

        if 'pred' not in os.listdir(self.DIR + 'results'):
            os.mkdir(self.DIR + 'results/pred')

        if 'cache' not in os.listdir(self.DIR + 'results'):
            os.mkdir(self.DIR + 'results/cache')

        if 'log' not in os.listdir(self.DIR + 'results'):
            os.mkdir(self.DIR + 'results/log')

        if 'ckpts' not in os.listdir(self.DIR + ''):
            os.mkdir(self.DIR + 'ckpts')


    def check_path(self):
        print('Checking dataset ...')

        for mode in self.IMG_FEAT_PATH:
            if not os.path.exists(self.IMG_FEAT_PATH[mode]):
                print(self.IMG_FEAT_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.QUESTION_PATH:
            if not os.path.exists(self.QUESTION_PATH[mode]):
                print(self.QUESTION_PATH[mode] + 'NOT EXIST')
                exit(-1)

        for mode in self.ANSWER_PATH:
            if not os.path.exists(self.ANSWER_PATH[mode]):
                print(self.ANSWER_PATH[mode] + 'NOT EXIST')
                exit(-1)

        print('Finished')
        print('')

