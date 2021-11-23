import json
import random
from pattern3.text.en import singularize
import pdb 
import torch
from torch import nn
import numpy as np

np.random.seed(10)
#random.seed(10)

class FRCNN() : 
    def __init__(self, __C, features_dataset, mode='val') :
        self.__C = __C

        self.obj_matching = json.load(open(__C.OBJ_MATCHING))
        if mode == 'val' :
            self.dataset_mapping = json.load(open(__C.VAL_FASTRCNN_MATCHING))
        else :
            self.dataset_mapping = json.load(open(__C.TRAIN_FASTRCNN_MATCHING))

        self.list_all_objects = list(self.dataset_mapping.keys())
        self.NO_OF_CHANGES = int(__C.NO_OF_CHANGES)

        self.allow_random = __C.ALLOW_RANDOM
        self.features_dataset = features_dataset

    def get_iou(self, a, b, epsilon=1e-5):
        """ Given two boxes `a` and `b` defined as a list of four numbers:
                [x1,y1,x2,y2]
            where:
                x1,y1 represent the upper left corner
                x2,y2 represent the lower right corner
            It returns the Intersect of Union score for these two boxes.

        Args:
            a:          (list of 4 numbers) [x1,y1,x2,y2]
            b:          (list of 4 numbers) [x1,y1,x2,y2]
            epsilon:    (float) Small value to prevent division by zero

        Returns:
            (float) The Intersect of Union score.
        """
        # COORDINATES OF THE INTERSECTION BOX
        x1 = max(a[0], b[0])
        y1 = max(a[1], b[1])
        x2 = min(a[2], b[2])
        y2 = min(a[3], b[3])

        # AREA OF OVERLAP - Area where the boxes intersect
        width = (x2 - x1)
        height = (y2 - y1)
        # handle case where there is NO overlap
        if (width<0) or (height <0):
            return 0.0
        area_overlap = width * height

        # COMBINED AREA
        area_a = (a[2] - a[0]) * (a[3] - a[1])
        area_b = (b[2] - b[0]) * (b[3] - b[1])
        area_combined = area_a + area_b - area_overlap

        # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
        iou = area_overlap / (area_combined+epsilon)
        return iou



    def get_overlap_box(self, boxes, ref_box):
        """ Given a list of boxes and a reference box, it returns the index from boxes list with maximum overlap with reference box """
        max_iou = 0.0
        obj_idx=100000000000

        for i in range(len(boxes)) :
            box = boxes[i]
            iou = self.get_iou(ref_box,box)
            if iou > max_iou :
                max_iou = iou
                obj_idx = i

        return obj_idx


    def get_similar_objects(self, name) :
        try :
            similar_objects = self.obj_matching[name]
        except :
            try :
                name1 = singularize(name)
                #name1 = name.rstrip('es')
                similar_objects = self.obj_matching[name1]
            except :
                name = name.rstrip('s')
                similar_objects = self.obj_matching[name]
        return similar_objects


    ''' Input - image number, unrelevant objects info, features corresponding to the image, bounding boxes corresponding to features , No. of changes argument
        
        Output :- List of features with perturbations corresponding to irrelevant objects

        Processing :- Iterates over list of unrelevant objects and finds a max of NO_OF_CHANGES perturbations for each.
                      How to find perturbation :- 
                      1. Find a object similar to object of interest and extract a corresponding faster rcnn feature from dataset. Swap the faster rcnn feature of object of interest with the new extracted feature.
    '''
    def get_features(self, img, unrelevant_objects, feats, boxes) :

        original_feature = feats
        features = []
        changes = []

        features.append(original_feature)
        changes.append('original')

        for name in unrelevant_objects :
            bbox = unrelevant_objects[name]['bbox']
            obj_idx = self.get_overlap_box(boxes, bbox)
            if obj_idx == 100000000000 :
                continue

            try :
                similar_objects = self.get_similar_objects(name)
            except :
                print ("Fail in similar objects : ", name )
                continue

            c=0
            for similar_obj in similar_objects :
                try :
                    features_list = self.dataset_mapping[similar_obj]
                except :
                    continue
                c += 1
                pick_file = random.choice(list(features_list.values()))

                sample_file_no = pick_file["file_no"]
                sample_obj_idx = pick_file["obj_idx"]
                pick_feature = torch.tensor(self.features_dataset[sample_file_no]["features"][sample_obj_idx])
                feature_temp = torch.tensor(original_feature)
                feature_temp[obj_idx] = pick_feature

                change = str(name) + '_to_' + str(similar_obj)

                features.append(feature_temp)
                changes.append(change)

                #feature_temp = original_feature

                if c >= self.NO_OF_CHANGES :
                    break
            
        return features, changes


    ''' Input - image number, unrelevant objects info, features corresponding to the image, bounding boxes corresponding to features , No. of changes argument
        
        Output :- List of features with changes in attributes corresponding to irrelevant objects.

        Processing :- Iterates over list of unrelevant objects and finds NO_OF_CHANGES perturbations for each.
                      How to find perturbation :- 
                      1. Extract another faster rcnn feature corresponding to object of interest.
                      2. Check if extracted feature and feature corresponding to object of interest are different -> key idea : if features belong to same object and if they are sufficiently different, them they must have different attributes.
    '''

    def get_features_irrelevant_attr(self, img, unrelevant_objects, feats, boxes) :

        original_feature = feats
        features = []
        changes = []

        features.append(original_feature)
        changes.append('original')

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)

        for name in unrelevant_objects :
            bbox = unrelevant_objects[name]['bbox']
            obj_idx = self.get_overlap_box(boxes, bbox)
            if obj_idx == 100000000000 :
                continue

            feature_of_interest = torch.tensor(original_feature[obj_idx])

            try :
                features_list = self.dataset_mapping[name]
            except :
                continue
            c =0
            i=0
            while(c<self.NO_OF_CHANGES) :
                i +=1
                if i>20 :
                    break

                pick_file = random.choice(list(features_list.values()))

                sample_file_no = pick_file["file_no"]
                sample_obj_idx = pick_file["obj_idx"]

                pick_feature = torch.tensor(self.features_dataset[sample_file_no]["features"][sample_obj_idx])
                
                if cos(feature_of_interest,pick_feature) > 0.7 :
                    continue
                
                feature_temp = torch.tensor(original_feature)
                feature_temp[obj_idx] = pick_feature

                change = str(name) + '_to_' + str(name)

                c +=1
                features.append(feature_temp)
                changes.append(change)

        return features, changes



    ''' Input - image number, unrelevant objects info, features corresponding to the image, bounding boxes corresponding to features , No. of changes argument
        
        Output :- List of features with perturbations corresponding to irrelevant objects

        Processing :- Iterates over list of unrelevant objects and finds NO_OF_CHANGES perturbations for each.
                      How to find perturbation :- 
                      1. Find a object similar to object of interest and extract a corresponding faster rcnn feature from dataset. Swap the faster rcnn feature of object of interest with the new extracted feature.
                      2. If no of similar objects found are less than NO_OF_CHANGES required, then pick a random object from the dataset and extract feature corresponding to that and swap accordingly.
    '''
    def get_features_including_random(self, img, unrelevant_objects, feats, boxes) :

        original_feature = feats
        features = []
        changes = []

        features.append(original_feature)
        changes.append('original')

        for name in unrelevant_objects :
            bbox = unrelevant_objects[name]['bbox']
            obj_idx = self.get_overlap_box(boxes, bbox)
            if obj_idx == 100000000000 :
                continue

            try :
                similar_objects = self.get_similar_objects(name)
            except :
                print ("Fail in similar objects : ", name )
                continue

            c=0
            for similar_obj in similar_objects :
                try :
                    features_list = self.dataset_mapping[similar_obj]
                except :
                    continue
                c += 1
                pick_file = random.choice(list(features_list.values()))

                sample_file_no = pick_file["file_no"]
                sample_obj_idx = pick_file["obj_idx"]

                pick_feature = torch.tensor(self.features_dataset[sample_file_no]["features"][sample_obj_idx])
                feature_temp = torch.tensor(original_feature)
                feature_temp[obj_idx] = pick_feature

                change = str(name) + '_to_' + str(similar_obj)

                features.append(feature_temp)
                changes.append(change)

                #feature_temp = original_feature

                if c >= self.NO_OF_CHANGES :
                    break

            while (c<self.NO_OF_CHANGES) :
                similar_obj = random.choice(self.list_all_objects)
                try :
                    features_list = self.dataset_mapping[similar_obj]
                except :
                    continue
                c +=1
                pick_file = random.choice(list(features_list.values()))

                sample_file_no = pick_file["file_no"]
                sample_obj_idx = pick_file["obj_idx"]

                pick_feature = torch.tensor(self.features_dataset[sample_file_no]["features"][sample_obj_idx])
                feature_temp = torch.tensor(original_feature)
                feature_temp[obj_idx] = pick_feature

                change = str(name) + '_to_' + str(similar_obj)

                features.append(feature_temp)
                changes.append(change)

        return features, changes



    ## This is used for swapmix training
    '''
    Input : image -> image number
            unrelevant_objs -> dict of unrelevant objects for the question being used in training

    Output : Change faster rcnn features corresponding to unrelevant objects with probabilty 0.5.
             For changing feature -> 0.5 probabilty change with same object and different attribute
                                  -> 0.5 probabilty change with random object

    '''
    def get_feature_swapmix(self, img, unrelevant_objs, feats, boxes) :

        original_feature = feats
        modified_feature = original_feature

        for obj in unrelevant_objs :

            ## Pick if the feature vector for the irrelevant objects needs to be changed with probability 0.25
            to_be_changed = random.randint(0,3)
            if to_be_changed != 0:
                ## Do not change feature vector
                continue

            bbox = unrelevant_objs[obj]['bbox']
            name = unrelevant_objs[obj]['name']
            obj_idx = self.get_overlap_box(boxes, bbox)
            if obj_idx == 100000000000 :
                ## No match found
                continue


            ## Pick if the change needs to be with object perturbation or attribute perturbation with probability 0.25
            obj_pert_change = random.randint(0,3)
            if obj_pert_change == 0 :
                ## Attribute perturbation - Select a different vector with same object name from dataset
                try : 
                    features_list = self.dataset_mapping[name]
                except :
                    ## No match found - veery few objects from entire dataset that fall in this category
                    continue

                pick_file = random.choice(list(features_list.values()))

                sample_file_no = pick_file["file_no"]
                sample_obj_idx = pick_file["obj_idx"]

                pick_feature = self.features_dataset[sample_file_no]["features"][sample_obj_idx]

                feature_temp = modified_feature
                feature_temp[obj_idx] = pick_feature

                modified_feature = feature_temp

            else :
                ## Object perturbation - Select a different object and replace the vector
                ## Here similar_obj means a different object but which has some relation to the object of interest - like car <-> bus
                try :
                    similar_objects = self.get_similar_objects(name)
                    if similar_objects == {} :
                        name = random.choice(list(self.dataset_mapping.keys()))
                        similar_objects = self.get_similar_objects(name)
                    similar_obj = random.choice(list(similar_objects.keys()))
                    features_list = self.dataset_mapping[similar_obj]
                except :
                    continue

                pick_file = random.choice(list(features_list.values()))

                sample_file_no = pick_file["file_no"]
                sample_obj_idx = pick_file["obj_idx"]

                pick_feature = self.features_dataset[sample_file_no]["features"][sample_obj_idx]

                feature_temp = modified_feature
                feature_temp[obj_idx] = pick_feature

                modified_feature = feature_temp
            
        return torch.tensor(modified_feature)
