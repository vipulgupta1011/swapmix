# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

import numpy as np
from numpyencoder import NumpyEncoder

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.gqa_model_scene import GQAModel

from tasks.gqa_scene_data_irrelevant_attr import GQADataset, GQATorchDataset, GQAEvaluator
import pdb
import json

import time
from cfgs import Cfgs

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    tset = GQATorchDataset(dset)
    evaluator = GQAEvaluator(dset)
 #   data_loader = DataLoader(
 #       tset, batch_size=bs,
 #       shuffle=shuffle, num_workers=args.num_workers,
 #       drop_last=drop_last, pin_memory=True )#

    return DataTuple(dataset=dset, loader=tset, evaluator=evaluator)

def get_tuple_train(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    #tset = GQATorchDataset(dset)
    tset = None
    evaluator = GQAEvaluator(dset)
 #   data_loader = DataLoader(
 #       tset, batch_size=bs,
 #       shuffle=shuffle, num_workers=args.num_workers,
 #       drop_last=drop_last, pin_memory=True )#

    return DataTuple(dataset=dset, loader=tset, evaluator=evaluator)

class GQA:
    def __init__(self):
        self.__C = Cfgs()
        if args.OUTPUT_JSON is not None :
            self.__C.OUTPUT_JSON = args.OUTPUT_JSON
        self.__C.NO_OF_CHANGES = args.NO_OF_CHANGES
        #self.__C.ALLOW_RANDOM = args.ALLOW_RANDOM

        self.train_tuple = get_tuple_train(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        #if args.valid != "":
        #    valid_bsize = 2048 if args.multiGPU else 512
        #    # valid_bsize = 300
        #    self.valid_tuple = get_tuple(
        #        args.valid, bs=valid_bsize,
        #        shuffle=False, drop_last=False
        #    )
        #else: 
        #    self.valid_tuple = None

        print ('pass1')
        self.model = GQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # GPU options
        self.model = self.model.cuda()
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()

        # Losses and optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        #if 'bert' in args.optim:
        #    batch_per_epoch = len(self.train_tuple.loader)
        #    t_total = int(batch_per_epoch * args.epochs)
        #    print("Total Iters: %d" % t_total)
        #    from lxrt.optimization import BertAdam
        #    self.optim = BertAdam(list(self.model.parameters()),
        #                          lr=args.lr,
        #                          warmup=0.1,
        #                          t_total=t_total)
        #else:
        #    self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        #self.output = args.output
        #os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        dset, loader, evaluator = train_tuple
        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

        best_valid = 0.
        for epoch in range(args.epochs):
            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                if args.mce_loss:
                    max_value, target = target.max(1)
                    loss = self.mce_loss(logit, target) * logit.size(1)
                else:
                    loss = self.bce_loss(logit, target)
                    loss = loss * logit.size(1)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def predict_gqa(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, gqa_loader, evaluator = eval_tuple
        quesid2ans = {}

        num_images = len(gqa_loader.list_images)
        print ('num_images : ', num_images)
        result = {}
        start_time = time.time()
        for idx in range(3000) :
            print (str(idx) + ' / ' + str(num_images))
            img, ques_dict, answers, img_feat_embed, changes_mapping, relevant_questions_mapping, img_feat_bbox = gqa_loader.extract(idx)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            result[img] = {}

            ## change has to be made
            result[img]['answers'] = answers
            for obj in img_feat_embed :
                result[img][obj] = {}
                result[img][obj]['changes'] = changes_mapping[obj]

                img_feat_embed_iters = img_feat_embed[obj]
                img_feat_bbox_iters  = img_feat_bbox[obj]
                img_feat_bbox_iters = torch.tensor(img_feat_bbox_iters, dtype=torch.float32)
                ques_dict_obj = {}
                ques_dict_obj = { relevant_key: ques_dict[relevant_key]  for relevant_key in relevant_questions_mapping[obj]}
                num_ques = len(ques_dict_obj)
                result[img][obj]['relevant_questions'] = relevant_questions_mapping[obj]
            
                if num_ques == 0:
                    continue

                c=0
                result[img][obj]["prediction"] = {}

                for q in range(len(img_feat_embed_iters)) :
                    c +=1
                    img_feat_embed_iter = img_feat_embed_iters[q]
                    img_feat_embed_iter = torch.tensor(img_feat_embed_iter, dtype=torch.float32)
                    img_feat_embed_iter = img_feat_embed_iter.unsqueeze(0).repeat(num_ques,1,1)
                    img_feat_embed_iter = img_feat_embed_iter.cuda()
                    #img_feat_embed_iter = torch.cat(num_ques * [img_feat_embed_iter])
                    img_feat_bbox_iter = img_feat_bbox_iters.unsqueeze(0).repeat(num_ques,1,1)
                    img_feat_bbox_iter = img_feat_bbox_iter.cuda()
                    #img_feat_bbox_iter = torch.cat(num_ques * [img_feat_bbox_iters])
                    question_ids = list(ques_dict_obj.keys())
                    question_iter = []
                    for ques in question_ids:
                        question_iter.append(ques_dict[ques]['question'])
                    
                    result[img][obj]["prediction"][q] = {} 
                    result[img][obj]["prediction"][q]["pred"] = []

                    n_batches = int(np.ceil(num_ques/64))
                    for i in range(n_batches) :
                        img_feat_embed_iter_batch = img_feat_embed_iter[64*i:64*(i+1)]
                        img_feat_bbox_iter_batch = img_feat_bbox_iter[64*i:64*(i+1)]
                        question_iter_batch = question_iter[64*i:64*(i+1)]

                        logit = self.model(img_feat_embed_iter_batch, img_feat_bbox_iter_batch, question_iter_batch)
                        score, label = logit.max(1)
                        result[img][obj]["prediction"][q]["pred"].append(label.cpu().data.numpy())

        print ('time taken :')
        print (time.time() - start_time)    
        with open(self.__C.OUTPUT_JSON, 'w+') as f :
            json.dump(result, f, indent=4, sort_keys=True, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)            
                    

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    def evaluate_gqa(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        self.predict_gqa(eval_tuple, dump)
        #return evaluator.evaluate(quesid2ans)
    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        for key in list(state_dict.keys()):
            if '.module' in key:
                state_dict[key.replace('.module', '')] = state_dict.pop(key)
        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":
    # Build Class
    gqa = GQA()


    # print("args.test:", args.test)
    # print("args.train:", args.train)
    # print("args.valid:", args.valid)

    # exit(0)

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'submit' in args.test:
            gqa.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'submit_predict.json')
            )
        if 'testdev' in args.test:
            result = gqa.evaluate(
                get_tuple('testdev', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)
        if 'valid' in args.test :
            result = gqa.evaluate_gqa(
                get_tuple('valid', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            
    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)


