# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

import os

class PATH:
    def __init__(self):

        # vqav2 dataset root path
        self.ROOT_DIR = '~/swapmix/data/'
        self.DATASET_PATH = self.ROOT + 'vqa/'

        # bottom up features root path
        self.DIR = self.ROOT + 'output/'
        self.GQA_PATH = self.ROOT + 'gqa/'
        #self.FASTRCNN_FEATURES_PATH = "/data/c/vipul/obj_features"
        self.FASTRCNN_FEATURES_PATH = self.ROOT + "obj_features/"
        self.FASTRCNN_FEATURES_INFO = self.ROOT + "gqa/frcnn_feature_info/"

        self.init_path()



    def init_path(self):

        self.IMG_FEAT_PATH = {
            #'train': self.GQA_PATH + 'images_split/small_train/small_train.json',
            'train': self.GQA_PATH + 'images_split/train.json',
            'val': self.GQA_PATH + 'images_split/val.json',
            #'val': self.GQA_PATH + 'images_split/small_val.json',
        }

        self.QUESTION_PATH = {
            'train': self.DATASET_PATH + 'v2_OpenEnded_mscoco_train2014_questions.json',
            'val': self.DATASET_PATH + 'v2_OpenEnded_mscoco_val2014_questions.json',
            'test': self.DATASET_PATH + 'v2_OpenEnded_mscoco_test2015_questions.json',
            'vg': self.DATASET_PATH + 'VG_questions.json',
        }

        self.GQA_QUESTION_PATH = {
            #'train': self.GQA_PATH + 'images_split/small_train/small_train_questions.json',
            'train': self.GQA_PATH + 'images_split/train_questions.json',
            #'val': self.GQA_PATH + 'images_split/small_train/small_val_questions.json',
            'val': self.GQA_PATH + 'images_split/val_questions.json',
        }

        ##self.GQA_SMALL_QUESTION_PATH = {
        self.GQA_QUESTION_PATH_NEW = {
            'train': self.GQA_PATH + 'images_split/train_questions_new.json',
            'val': self.GQA_PATH + 'images_split/val_questions_new.json',
            #'val': self.GQA_PATH + 'images_split/small_ques.json',
        }

        self.ANSWER_PATH = {
            #'train': self.GQA_PATH + 'images_split/small_train/small_train_answers.json',
            'train': self.GQA_PATH + 'images_split/train_answers.json',
            #'val': self.GQA_PATH + 'images_split/small_train/small_val_answers.json',
            'val': self.GQA_PATH + 'images_split/val_answers.json',
        }
        self.f0_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_0.h5'
        self.f1_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_1.h5'
        self.f2_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_2.h5'
        self.f3_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_3.h5'
        self.f4_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_4.h5'
        self.f5_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_5.h5'
        self.f6_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_6.h5'
        self.f7_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_7.h5'
        self.f8_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_8.h5'
        self.f9_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_9.h5'
        self.f10_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_10.h5'
        self.f11_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_11.h5'
        self.f12_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_12.h5'
        self.f13_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_13.h5'
        self.f14_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_14.h5'
        self.f15_path = self.FASTRCNN_FEATURES_PATH + 'gqa_objects_15.h5'

        self.FASTRCNN_INFO_FILE = self.FASTRCNN_FEATURES_INFO + 'gqa_objects_info.json'
        self.FASTRCNN_MATCHING = self.FASTRCNN_FEATURES_INFO + 'object_mapping_new.json'
        ##self.SMALL_ANS_PATH = self.GQA_PATH + 'images_split/small_ans.json'
        self.VAL_ANS_PATH = self.GQA_PATH + 'images_split/val_answers_new.json'
        self.VAL_SCENE_GRAPH = self.GQA_PATH + 'scene_graphs/val_sceneGraphs.json'
        ##self.VAL_SCENE_GRAPH = self.GQA_PATH + 'scene_graphs/small_val_sceneGraphs.json'
        self.TRAIN_SCENE_GRAPH = self.GQA_PATH + 'scene_graphs/train_sceneGraphs.json'
        self.GQA_VOCAB = self.GQA_PATH + 'glove_embds/gqa_vocab_taxo.json'
        self.GQA_EMBEDDING = self.GQA_PATH + 'glove_embds/attrlabel_glove_taxo.npy'
        self.RESULT_PATH = self.DIR + 'results/fastrcnn/'
        self.PRED_PATH = self.DIR + 'results/pred/'
        self.CACHE_PATH = self.DIR + 'results/cache/'
        self.LOG_PATH = self.DIR + 'results/log/'
        self.CKPTS_PATH = self.DIR + 'ckpts/'
        self.OBJ_MATCHING = self.GQA_PATH + 'matching/obj_matching.json'
        self.ATTR_MATCHING = self.GQA_PATH + 'matching/attr_matching.json'
        self.SCENE_OBJECT_MATCHING = self.GQA_PATH + 'scene_graphs/object_mapping.json'
        self.OUTPUT_JSON = self.RESULT_PATH + 'temp.json' 
        self.ATTR_OUTPUT_JSON = self.RESULT_PATH + 'temp_attr.json'

        if 'results' not in os.listdir(self.DIR + ''):
            os.mkdir(self.DIR + 'results')

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

