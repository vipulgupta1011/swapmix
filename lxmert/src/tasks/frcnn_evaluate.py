# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections

import torch
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.frcnn_model import GQAModel
from tasks.dataloader.frcnn_data_evaluate import GQADataset, GQATorchDataset, GQAEvaluator
import pdb

import json
from numpyencoder import NumpyEncoder
import numpy as np
from swapmix.frcnn_modify import FRCNN

from cfgs import Cfgs
DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
    dset = GQADataset(splits)
    ## Change GQATorchDataset and iterate over each element
    tset = GQATorchDataset(dset)
    #pdb.set_trace()
    evaluator = GQAEvaluator(dset)
    data_loader = None
    #data_loader = DataLoader(
    #    tset, batch_size=bs,
    #    shuffle=shuffle, num_workers=args.num_workers,
    #    drop_last=drop_last, pin_memory=True
    #)

    return DataTuple(dataset=dset, loader=tset, evaluator=evaluator)


class GQA:
    def __init__(self):
        self.__C = Cfgs()
        if args.OUTPUT_JSON is not None :
            self.__C.OUTPUT_JSON = args.OUTPUT_JSON
        self.__C.NO_OF_CHANGES = args.NO_OF_CHANGES
        self.__C.ALLOW_RANDOM = args.ALLOW_RANDOM

        print (self.__C)
        #print ('allow random : ', str(self.__C.ALLOW_RANDOM))
        self.train_tuple = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

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
        if 'bert' in args.optim:
            #batch_per_epoch = len(self.train_tuple.loader)
            batch_per_epoch = 1
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

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

        #self.save("LAST")

    ## Need to make changes here - feature swapping and iterating over image and ques
    def predict_gqa(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, gqa_loader, evaluator = eval_tuple
        quesid2ans = {}

        self.frcnn = FRCNN(self.__C, gqa_loader.img_data)

        num_images = len(gqa_loader.list_images)
        print ('num_images : ', num_images)
        result = {}
        for i in range(num_images) :
            print (str(i) + ' / ' + str(num_images))

            ''' relevant_questions - > questions where object is relevant '''
            try :
                names_mapping, bbox_mapping, questions, answers, relevant_questions_mapping, img, boxes, feats, img_h, img_w = gqa_loader.extract(i)
            except :
                continue

            full_boxes = boxes.copy()
            result[img] = {}
            boxes[:, (0, 2)] /= img_w
            boxes[:, (1, 3)] /= img_h
            for ques in questions :
                question = questions[ques]
                answer = answers[ques]
                ## unrelevant_objects -> list of objects which are irrelevant to the question and can be perturbed
                unrelevant_objects  = {}
                for obj in relevant_questions_mapping :
                    ## not condition ensures that we are taking only questions where object is not relevant
                    if ques not in relevant_questions_mapping[obj] :
                        name = names_mapping[obj]
                        unrelevant_objects[name] = {}
                        unrelevant_objects[name]['bbox'] = bbox_mapping[obj]

                if args.CLASS == 'objects' :
                    ## extracting features corresponding to unrelevant objects
                    if self.__C.ALLOW_RANDOM :
                        features, changes = self.frcnn.get_features_including_random(img, unrelevant_objects, feats, full_boxes) 
                    else :
                        features, changes = self.frcnn.get_features(img, unrelevant_objects, feats, full_boxes)
                
                if args.CLASS == 'attributes' :
                    ## extracting features corresponding to attribute change for irrelevant objects
                    features, changes = self.frcnn.get_features_irrelevant_attr(img, unrelevant_objects, feats, full_boxes) 


                len_features = len(features)
                img_feat_iters = torch.zeros((len_features,36,2048))
                for j in range(len_features) :
                    img_feat_iters[j] = torch.tensor(features[j])
                img_feat_iters = torch.tensor(img_feat_iters, dtype=torch.float32)

                question_iters = len_features * [question['question']]
                boxes = torch.tensor(boxes, dtype=torch.float32)
                boxes_iters = boxes.unsqueeze(0).repeat(len_features,1,1)

                img_feat_iters = img_feat_iters.cuda()
                boxes_iters = boxes_iters.cuda()
                ## Performing predictions in batches
                n_batches = int(np.ceil(len_features/48))

                result[img][ques] = {}
                result[img][ques]["ans"] = answer
                result[img][ques]["changes"] = changes
                result[img][ques]["pred"] = []
                for k in range(n_batches) :
                    img_feat_iters_batch = img_feat_iters[48*k:48*(k+1)]
                    boxes_iters_batch = boxes_iters[48*k:48*(k+1)]
                    question_iters_batch = question_iters[48*k:48*(k+1)]

                    #Prediction
                    logit = self.model(img_feat_iters_batch, boxes_iters_batch, question_iters_batch)
                    score, label = logit.max(1)

                    result[img][ques]["pred"].append(label.cpu().data.numpy())

        with open(self.__C.OUTPUT_JSON,'w+') as f :
            json.dump(result, f, indent=4, sort_keys=True, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)





        #for i, datum_tuple in enumerate(loader):
        #    ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
        #    with torch.no_grad():
        #        feats, boxes = feats.cuda(), boxes.cuda()
        #        logit = self.model(feats, boxes, sent)
        #        score, label = logit.max(1)
        #        for qid, l in zip(ques_id, label.cpu().numpy()):
        #            ans = dset.label2ans[l]
        #            quesid2ans[qid] = ans
        #if dump is not None:
        #    evaluator.dump_result(quesid2ans, dump)
        #return quesid2ans

    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        pdb.set_trace()
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

    # Load Model
    if args.load is not None:
        gqa.load(args.load)

    # Test or Train
    if args.test is not None:
        #args.fast = args.tiny = False       # Always loading all data in test
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

        if 'valid' in args.test:
            #result = gqa.evaluate(
            result = gqa.evaluate_gqa(
                get_tuple('valid', bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'testdev_predict.json')
            )
            print(result)

    else:
        # print("Train Oracle: %0.2f" % (gqa.oracle_score(gqa.train_tuple) * 100))
        print('Splits in Train data:', gqa.train_tuple.dataset.splits)
        if gqa.valid_tuple is not None:
            print('Splits in Valid data:', gqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (gqa.oracle_score(gqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        gqa.train(gqa.train_tuple, gqa.valid_tuple)


