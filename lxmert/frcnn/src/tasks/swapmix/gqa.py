import os
from swapmix.utils import load_json, np_load
import numpy as np
import torch
import torch.nn as nn
import json
import pdb

class GQADataset_swapmix() :

    def __init__(self, scenegraphs_json, ques_list) :
        self.scenegraphs_json = load_json(scenegraphs_json)
        self.ques_list = ques_list

    def getobjects(self, imageid) :
        ## object mapping to object name
        scenegraphs_json = self.scenegraphs_json
        info_raw = scenegraphs_json[imageid]
        objects_raw = {}
        for obj in info_raw['objects'] :
            objects_raw[obj] = info_raw['objects'][obj]['name']
        return objects_raw

    def get_unrelated_ques(self, imageid, objid) :
        unrelated = {}
        ques_list = self.ques_list
        for ques in ques_list[imageid] :
            if objid not in ques_list[imageid][ques]['annotations'].keys() :
                unrelated[ques] = {}
        return unrelated

    def get_unrelevant_objects(self, info, ques, objects_raw) :
        unrelevant_objects = {}
        annotations = info[ques]['annotations']
        for obj in objects_raw :
            if obj not in annotations :
                unrelevant_objects[obj] = {}

        return unrelevant_objects

    def get_relevant_ques(self, imageid, objid) :
        related = {}
        ques_list = self.ques_list
        for ques in ques_list[imageid] :
            if objid in ques_list[imageid][ques]['annotations'].keys() :
                related[ques] = {}
        return related

    def get_bbox(self, imageid, objid) :
        scenegraphs_json = self.scenegraphs_json
        obj_info = scenegraphs_json[imageid]["objects"][objid]
        x = obj_info["x"]
        y = obj_info["y"]
        w = obj_info["w"]
        h = obj_info["h"]
        ref = [x,y,x+w,y+h]
        return ref


    ''' Always use this function instead of above function to avoid confusion '''
    ''' Input -> image list
        Output -> image -> obj -> {questions where object is relevant, name of the object, bounding box}
    '''
    def extractembeddings_related(self,images_list, mapping) :
        final_embeddings = mapping
        images = images_list
        for image in images :
            final_embeddings[image] = {}
            objects_raw = self.getobjects(image)
            for obj in objects_raw :
                relevant_ques = self.get_relevant_ques(image, obj)
                bbox = self.get_bbox(image, obj)
                name = self.scenegraphs_json[image]['objects'][obj]['name']
                final_embeddings[image][obj] = {}
                final_embeddings[image][obj]['relevant_questions'] = relevant_ques
                final_embeddings[image][obj]['name'] = name
                final_embeddings[image][obj]['bbox'] = bbox

        return final_embeddings
    


    ## This function is used only for swapmix training
    ## It gets unrelevant objects for each question
    def extractembeddings_questions(self,images_list, mapping) :
        final_embeddings = mapping
        images = images_list
        for image in images :
            final_embeddings[image] = {}
            info = self.ques_list[image]
            questions = list(info.keys())
            objects_raw = self.getobjects(image)
            for ques in questions :
                ## unrelevant_objects -> objects which are not relevant to the question
                unrelevant_objects = self.get_unrelevant_objects(info, ques, objects_raw)
                for obj in unrelevant_objects :
                    bbox = self.get_bbox(image, obj)
                    name = self.scenegraphs_json[image]['objects'][obj]['name']
                    unrelevant_objects[obj]['bbox'] = bbox
                    unrelevant_objects[obj]['name'] = name
                final_embeddings[image][ques] = {}
                final_embeddings[image][ques]['unrelevant_objects'] = unrelevant_objects

        return final_embeddings
