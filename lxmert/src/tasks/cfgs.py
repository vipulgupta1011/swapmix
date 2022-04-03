class Cfgs():
    def __init__(self):

        self.ROOT_DIR = '~/swapmix/data/'
        self.ALLOW_RANDOM = True

        self.GQA_PATH = self.ROOT_DIR + 'gqa/'
        self.IMG_FEAT_PATH = self.ROOT + 'obj_features/'
        self.DATA_PATH =self.GQA_PATH + 'lxmert/'

        self.QUESTION_PATH = {
            'train' : self.DATA_PATH + 'train.json',
            'valid' : self.DATA_PATH + 'valid.json',
            'testdev' : self.DATA_PATH + 'testdev.json',
        }

        self.ANS2LABEL = self.DATA_PATH + 'trainval_ans2label.json'
        self.LABEL2ANS = self.DATA_PATH + 'trainval_label2ans.json'

        #Not useful for this experiment 
        #self.TESTDEV_FEAT_PATH = self.IMG_FEAT_PATH + 'gqa_testdev_obj36.tsv'
        self.GQA_FEAT_PATH = self.IMG_FEAT_PATH + 'vg_gqa_obj36.tsv'
        self.OBJ_MATCHING = self.GQA_PATH + 'matching/obj_matching.json'
        self.ATTR_MATCHING = self.GQA_PATH + 'matching/attr_matching.json'
        self.NO_OF_CHANGES = 5
        self.VAL_FASTRCNN_MATCHING = self.GQA_PATH + 'matching/val_dataset_mapping_new.json'
        self.TRAIN_FASTRCNN_MATCHING = self.GQA_PATH + 'matching/train_dataset_mapping.json' 
        self.OUTPUT_JSON = self.ROOT_DIR + 'output/temp.json' 

        self.SCENE_MATCHING = self.GQA_PATH + 'scene_graphs/object_mapping.json'

        self.TRAIN_IMG_LIST_PATH = self.GQA_PATH + 'images_split/train.json'
        self.VAL_IMG_LIST_PATH = self.GQA_PATH + 'images_split/val.json'

        self.TRAIN_SCENE_GRAPH = self.GQA_PATH + 'scene_graphs/train_sceneGraphs.json'
        self.VAL_SCENE_GRAPH = self.GQA_PATH + 'scene_graphs/val_sceneGraphs.json'

        self.TRAIN_QUES_PATH = self.GQA_PATH + 'images_split/train_questions_new.json'
        self.VAL_QUES_PATH = self.GQA_PATH + 'images_split/val_questions_new.json'

        self.VAL_ANS_PATH = self.GQA_PATH + 'images_split/val_answers_new.json'

        self.GQA_VOCAB = self.GQA_PATH + 'glove_embds/gqa_vocab_taxo.json'
        self.GQA_EMBEDDING = self.GQA_PATH + 'glove_embds/attrlabel_glove_taxo.npy'


