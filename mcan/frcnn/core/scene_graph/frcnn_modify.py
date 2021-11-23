import h5py
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
    def __init__(self, __C) :
        self.__C = __C

        self.f0 = h5py.File(__C.f0_path, 'r')
        self.f1 = h5py.File(__C.f1_path, 'r')
        self.f2 = h5py.File(__C.f2_path, 'r')
        self.f3 = h5py.File(__C.f3_path, 'r')
        self.f4 = h5py.File(__C.f4_path, 'r')
        self.f5 = h5py.File(__C.f5_path, 'r')
        self.f6 = h5py.File(__C.f6_path, 'r')
        self.f7 = h5py.File(__C.f7_path, 'r')
        self.f8 = h5py.File(__C.f8_path, 'r')
        self.f9 = h5py.File(__C.f9_path, 'r')
        self.f10 = h5py.File(__C.f10_path, 'r')
        self.f11 = h5py.File(__C.f11_path, 'r')
        self.f12 = h5py.File(__C.f12_path, 'r')
        self.f13 = h5py.File(__C.f13_path, 'r')
        self.f14 = h5py.File(__C.f14_path, 'r')
        self.f15 = h5py.File(__C.f15_path, 'r')

        self.info_file = json.load(open(__C.FASTRCNN_INFO_FILE))
        self.val_scene = json.load(open(__C.VAL_SCENE_GRAPH))

        self.obj_matching = json.load(open(__C.OBJ_MATCHING))
        self.dataset_mapping = json.load(open(__C.FASTRCNN_MATCHING))

        self.list_all_objects = list(self.dataset_mapping.keys())
        self.NO_OF_CHANGES = int(__C.NO_OF_CHANGES)

        self.file_map = {0:self.f0, 1:self.f1, 2:self.f2, 3:self.f3, 4:self.f4, 5:self.f5, 6:self.f6, 7:self.f7, 8:self.f8, 9:self.f9, 10:self.f10, 11:self.f11, 12:self.f12, 13:self.f13, 14:self.f14, 15:self.f15}

        self.mean = int(__C.MEAN)
        self.std_dev = int(__C.STD_DEV)

        self.allow_random = __C.ALLOW_RANDOM

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

    ## To be used for masking all the overlapping boxes of relevant features to zero
    def get_overlap_boxes(self, boxes, ref_box):
        """ Given a list of boxes and a reference box, it returns the index from boxes list with maximum overlap with reference box """
        max_iou = 0.0
        obj_idx=100000000000
        obj_list = []
        for i in range(len(boxes)) :
            box = boxes[i]
            iou = self.get_iou(ref_box,box)
            if iou > 0.25 :
                obj_list.append(i)

        return obj_list


    def get_similar_objects(self, name) :
        try :
            similar_objects = self.obj_matching[name]
        except :
            try :
                name1 = singularize(name)
                similar_objects = self.obj_matching[name1]
            except :
                name = name.rstrip('s')
                similar_objects = self.obj_matching[name]
        return similar_objects


    def get_features(self, img, unrelevant_objects) :
        info = self.info_file[img]
        idx = info["idx"]
        file_no = info["file"]
        file_name = self.file_map[file_no]
        boxes = file_name["bboxes"][idx]

        original_feature = file_name["features"][idx]
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
                c += 1
                try :
                    features_list = self.dataset_mapping[similar_obj]
                except :
                    continue
                pick_file = random.choice(list(features_list.keys()))

                sample_file_no = features_list[pick_file]["file_no"]
                sample_idx = features_list[pick_file]["idx"]
                sample_obj_idx = features_list[pick_file]["obj_idx"]

                pick_feature = torch.tensor(self.file_map[sample_file_no]["features"][sample_idx][sample_obj_idx])
                feature_temp = torch.tensor(original_feature)
                feature_temp[obj_idx] = pick_feature

                change = str(name) + '_to_' + str(similar_obj)

                features.append(feature_temp)
                changes.append(change)

                #feature_temp = original_feature

                if c >= self.NO_OF_CHANGES :
                    break
            
        return features, changes



    def get_features_irrelevant_attr(self, img, unrelevant_objects) :
        info = self.info_file[img]
        idx = info["idx"]
        file_no = info["file"]
        file_name = self.file_map[file_no]
        boxes = file_name["bboxes"][idx]

        original_feature = file_name["features"][idx]
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
                sample_idx = pick_file["idx"]
                sample_obj_idx = pick_file["obj_idx"]

                pick_feature = torch.tensor(self.file_map[sample_file_no]["features"][sample_idx][sample_obj_idx])
                
                if cos(feature_of_interest,pick_feature) > 0.7 :
                    continue
                
                feature_temp = torch.tensor(original_feature)
                feature_temp[obj_idx] = pick_feature

                change = str(name) + '_to_' + str(name)

                c +=1
                features.append(feature_temp)
                changes.append(change)

        return features, changes



    ## Matches related objects and then does random selection to select 5 perturbations
    def get_features_including_random(self, img, unrelevant_objects) :
        info = self.info_file[img]
        idx = info["idx"]
        file_no = info["file"]
        file_name = self.file_map[file_no]
        boxes = file_name["bboxes"][idx]

        original_feature = file_name["features"][idx]
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
                pick_file = random.choice(list(features_list.keys()))

                sample_file_no = features_list[pick_file]["file_no"]
                sample_idx = features_list[pick_file]["idx"]
                sample_obj_idx = features_list[pick_file]["obj_idx"]

                pick_feature = torch.tensor(self.file_map[sample_file_no]["features"][sample_idx][sample_obj_idx])
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
                pick_file = random.choice(list(features_list.keys()))

                sample_file_no = features_list[pick_file]["file_no"]
                sample_idx = features_list[pick_file]["idx"]
                sample_obj_idx = features_list[pick_file]["obj_idx"]

                pick_feature = torch.tensor(self.file_map[sample_file_no]["features"][sample_idx][sample_obj_idx])
                feature_temp = torch.tensor(original_feature)
                feature_temp[obj_idx] = pick_feature

                change = str(name) + '_to_' + str(similar_obj)

                features.append(feature_temp)
                changes.append(change)

        return features, changes

    ## This function will be used for masking features of relevant object to zeros
    def get_features_masking(self, img, relevant_objects) :
        info = self.info_file[img]
        idx = info["idx"]
        file_no = info["file"]
        file_name = self.file_map[file_no]
        boxes = file_name["bboxes"][idx]

        original_feature = file_name["features"][idx]
        features = []
        changes = []

        features.append(original_feature)
        changes.append('original')

        modified_feature = original_feature.copy()
        for name in relevant_objects :
            bbox = relevant_objects[name]['bbox']
            obj_list = self.get_overlap_boxes(boxes, bbox)
            if obj_list == [] :
                continue

            zero_feature = torch.zeros(1, 2048)
            for obj_idx in obj_list :
                modified_feature[obj_idx] = zero_feature

        features.append(modified_feature)
        changes.append('masked')

        return features, changes

    ## This is used for training
    def get_feature_single(self, img) :
        info = self.info_file[img]
        idx = info["idx"]
        file_no = info["file"]
        file_name = self.file_map[file_no]
        boxes = file_name["bboxes"][idx]

        feature = file_name["features"][idx]

        return feature


    ## This is used for training with random gaussian noise added
    ## Random noise is added to every vector to see the effect of good visual features on the model
    def get_feature_single_noise(self, img) :
        info = self.info_file[img]
        idx = info["idx"]
        file_no = info["file"]
        file_name = self.file_map[file_no]
        boxes = file_name["bboxes"][idx]

        feature = file_name["features"][idx]
        noise = np.random.normal(self.mean,self.std_dev,size=(100,2048)).astype('float32')

        feature = feature + noise
        feature[feature<0] = 0

        return feature



    ## This is used for swapmix training
    '''
    Input : image -> image number
            unrelevant_objs -> dict of unrelevant objects for the question being used in training

    Output : Change faster rcnn features corresponding to unrelevant objects with probabilty 0.5.
             For changing feature -> 0.5 probabilty change with same object and different attribute
                                  -> 0.5 probabilty change with random object

    '''
    def get_feature_swapmix(self, img, unrelevant_objs) :
        info = self.info_file[img]
        idx = info["idx"]
        file_no = info["file"]
        file_name = self.file_map[file_no]
        boxes = file_name["bboxes"][idx]

        original_feature = file_name["features"][idx]
        modified_feature = original_feature

        for obj in unrelevant_objs :

            ## Pick if the feature vector for the irrelevant objects needs to be changed with probability 0.5
            to_be_changed = random.randint(0,1)
            if to_be_changed != 0:
                ## Do not change feature vector
                continue

            bbox = unrelevant_objs[obj]['bbox']
            name = unrelevant_objs[obj]['name']
            obj_idx = self.get_overlap_box(boxes, bbox)
            if obj_idx == 100000000000 :
                ## No match found
                continue


            ## Pick if the change needs to be with object perturbation or attribute perturbation with probability 0.5
            obj_pert_change = random.randint(0,1)
            if obj_pert_change == 0 :
                ## Attribute perturbation - Select a different vector with same object name from dataset
                try : 
                    features_list = self.dataset_mapping[name]
                except :
                    ## No match found - veery few objects from entire dataset that fall in this category
                    continue

                pick_file = random.choice(list(features_list.values()))

                sample_file_no = pick_file["file_no"]
                sample_idx = pick_file["idx"]
                sample_obj_idx = pick_file["obj_idx"]

                pick_feature = self.file_map[sample_file_no]["features"][sample_idx][sample_obj_idx]

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
                sample_idx = pick_file["idx"]
                sample_obj_idx = pick_file["obj_idx"]

                pick_feature = self.file_map[sample_file_no]["features"][sample_idx][sample_obj_idx]

                feature_temp = modified_feature
                feature_temp[obj_idx] = pick_feature

                modified_feature = feature_temp
            
        return modified_feature
