# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

from swapmix.gqa import GQADataset_swapmix

from cfgs import Cfgs

import pdb
from swapmix.frcnn_modify import FRCNN

TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class GQADataset:
    """
    A GQA data example in json file:
    {
        "img_id": "2375429",
        "label": {
            "pipe": 1.0
        },
        "question_id": "07333408",
        "sent": "What is on the white wall?"
    }
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        __C = Cfgs()

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open(__C.QUESTION_PATH[split])))
            #self.data.extend(json.load(open("/data/c/zhuowan/gqa_project/lxmert/data/gqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open(__C.ANS2LABEL))
        self.label2ans = json.load(open(__C.LABEL2ANS))
        assert len(self.ans2label) == len(self.label2ans)
        for ans, label in self.ans2label.items():
            assert self.label2ans[label] == ans

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


class GQABufferLoader():
    def __init__(self):
        self.key2data = {}
        self.__C = Cfgs()

    def load_data(self, name, number):
        if name == 'testdev':
            path = self.__C.TESTDEV_FEAT_PATH
        else:
            path = self.__C.GQA_FEAT_PATH

        key = "%s_%d" % (path, number)
        if key not in self.key2data:
            self.key2data[key] = load_obj_tsv(
                path,
                topk=number
            )
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()


"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        img_data = []
        if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:     # Always loading all the data in testdev
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            img_data.extend(gqa_buffer_loader.load_data('train', topk))
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-5)
        np.testing.assert_array_less(-boxes, 0+1e-5)

        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques

class GQATorchDataset_swapmix(Dataset):
    def __init__(self, dataset: GQADataset):
        super().__init__()
        self.raw_dataset = dataset

        self.__C = Cfgs()

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        # Since images in train and valid both come from Visual Genome,
        # buffer the image loading to save memory.
        self.img_data = []
        if 'testdev' in dataset.splits or 'testdev_all' in dataset.splits:     # Always loading all the data in testdev
            img_data.extend(gqa_buffer_loader.load_data('testdev', -1))
        else:
            self.img_data.extend(gqa_buffer_loader.load_data('train', topk))
        self.imgid2img = {}
        for img_datum in self.img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)

        self.list_images = list(json.load(open(self.__C.TRAIN_IMG_LIST_PATH)).keys())
        self.ques_list_new = json.load(open(self.__C.TRAIN_QUES_PATH, 'r'))
        GQA_train = GQADataset_swapmix(self.__C.TRAIN_SCENE_GRAPH, self.ques_list_new)
        self.img_to_gqa_info = {}
        self.img_to_gqa_info = GQA_train.extractembeddings_questions(self.list_images, self.img_to_gqa_info)

        self.frcnn = FRCNN(self.__C, self.img_data, mode='train')

        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
    #def extract(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']

        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        #np.testing.assert_array_less(boxes, 1+1e-5)
        #np.testing.assert_array_less(-boxes, 0+1e-5)

        try : 
            unrelevant_objs = self.img_to_gqa_info[img_id][ques_id]['unrelevant_objects']

            feats = self.frcnn.get_feature_swapmix(img_id, unrelevant_objs, feats, boxes) 
        except :
            pass
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h

        # Create target
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                if ans in self.raw_dataset.ans2label:
                    target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target
        else:
            return ques_id, feats, boxes, ques


class GQAEvaluator:
    def __init__(self, dataset: GQADataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump the result to a GQA-challenge submittable json file.
        GQA json file submission requirement:
            results = [result]
            result = {
                "questionId": str,      # Note: it's a actually an int number but the server requires an str.
                "prediction": str
            }

        :param quesid2ans: A dict mapping question id to its predicted answer.
        :param path: The file path to save the json file.
        :return:
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'questionId': ques_id,
                    'prediction': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)

