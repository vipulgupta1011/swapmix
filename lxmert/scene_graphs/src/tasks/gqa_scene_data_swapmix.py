# coding=utf-8
# Copyleft 2019 project LXRT.

import json

import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_scene_graphs

from gqa_scene_graph_irrelevant_obj import GQASceneDataset
# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
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

        # Loading datasets to data
        self.data = []
        for split in self.splits:
            self.data.extend(json.load(open("/home/vipul/chenyu/lxmert/data/gqa/%s.json" % split)))
        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        self.ans2label = json.load(open("/home/vipul/chenyu/lxmert/data/gqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("/home/vipul/chenyu/lxmert/data/gqa/trainval_label2ans.json"))
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

    def load_data(self, name, number):
        if name == 'train':
            scenegraphs_json = "/home/vipul/gqa_dataset/scene_graphs/train_sceneGraphs.json"
        else:
            scenegraphs_json = "/home/vipul/gqa_dataset/scene_graphs/val_sceneGraphs.json"
        key = "%s_%d" % (scenegraphs_json, number)
        vocab_json = '/home/vipul/gqa_dataset/gqa_vocab_taxo.json'
        embedding_json = '/home/vipul/gqa_dataset/attrlabel_glove_taxo.npy'
        if key not in self.key2data:
            self.key2data[key] = load_scene_graphs(
                scenegraphs_json,
                vocab_json,
                embedding_json,
                topk=number
            )
        return self.key2data[key]


gqa_buffer_loader = GQABufferLoader()


class GQATorchDataset_valid(Dataset):
    ###### Need to make changes here #####
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


        # loading image data
        img_data = []
        img_data.extend(gqa_buffer_loader.load_data('valid', -1))
        # if 'val' in dataset.splits:     # Always loading all the data in testdev
        #     img_data.extend(gqa_buffer_loader.load_data('val', -1))
        # else:
        #     img_data.extend(gqa_buffer_loader.load_data('train', topk))
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

        ## these 3 comes from train.json file
        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']



        # Get image info
        ''' Change needs to be made here'''
        ##### These comes from load_scene_graph
        ##### Need only boxes and feats frm scene graphs
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        boxes = img_info['boxes'].copy()
        feats = img_info['features'].copy()
        assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        #boxes[:, (0, 2)] /= img_w
        #boxes[:, (1, 3)] /= img_h
        #np.testing.assert_array_less(boxes, 1+1e-5)
        #np.testing.assert_array_less(-boxes, 0+1e-5)


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
"""
Example in obj tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
"""
class GQATorchDataset(Dataset):
    ###### Need to make changes here #####
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


        # loading image data
        scenegraphs_json = "/home/vipul/gqa_dataset/scene_graphs/train_sceneGraphs.json"
        vocab_json = '/home/vipul/gqa_dataset/gqa_vocab_taxo.json'
        embedding_json = '/home/vipul/gqa_dataset/attrlabel_glove_taxo.npy'
        obj_matching = '/home/vipul/gqa_dataset/matching/obj_matching.json'
        attr_matching = '/home/vipul/gqa_dataset/matching/attr_matching.json'
        ques_list_new = '/home/vipul/gqa_dataset/images_split/train_questions_new.json'
        object_mapping = '/home/vipul/gqa_dataset/scene_graphs/object_mapping.json'
        self.GQA_with_scene_graph = GQASceneDataset(scenegraphs_json, vocab_json, embedding_json, obj_matching, attr_matching, object_mapping, ques_list_new)
        #img_data = []
        #if 'valid' in dataset.splits:     
        #    img_data.extend(gqa_buffer_loader.load_data('valid', -1))
        #else:
        #    img_data.extend(gqa_buffer_loader.load_data('train', topk))
        # if 'val' in dataset.splits:     # Always loading all the data in testdev
        #     img_data.extend(gqa_buffer_loader.load_data('val', -1))
        # else:
        #     img_data.extend(gqa_buffer_loader.load_data('train', topk))
        #self.imgid2img = {}
        #for img_datum in img_data:
        #    self.imgid2img[img_datum['img_id']] = img_datum



            

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            #if datum['img_id'] in self.imgid2img:
            self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, item: int):
        datum = self.data[item]

        ## these 3 comes from train.json file
        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']



        feats, boxes = self.GQA_with_scene_graph.extractembeddings_swapmix_single(img_id, ques_id)
        # Get image info
        ''' Change needs to be made here'''
        ##### These comes from load_scene_graph
        ##### Need only boxes and feats frm scene graphs
        #img_info = self.imgid2img[img_id]
        #obj_num = img_info['num_boxes']
        #boxes = img_info['boxes'].copy()
        #feats = img_info['features'].copy()
        #assert len(boxes) == len(feats) == obj_num

        # Normalize the boxes (to 0 ~ 1)
        #img_h, img_w = img_info['img_h'], img_info['img_w']
        #boxes = boxes.copy()
        #boxes[:, (0, 2)] /= img_w
        #boxes[:, (1, 3)] /= img_h
        #np.testing.assert_array_less(boxes, 1+1e-5)
        #np.testing.assert_array_less(-boxes, 0+1e-5)


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
        if len(quesid2ans) == 0:
            return score
        else:
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


