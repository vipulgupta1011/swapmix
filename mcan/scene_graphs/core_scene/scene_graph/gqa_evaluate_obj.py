import os
from core_scene.scene_graph.utils import load_json, np_load
import numpy as np
import torch
import torch.nn as nn
import json
import pdb
from pattern3.text.en import singularize
import random

class GQADataset() :
    def __init__(self, scenegraphs_json, vocab_json, embedding_json, obj_matching, attr_matching, ques_list, object_mapping, no_of_changes, allow_random) :
        self.scenegraphs_json = load_json(scenegraphs_json)
        self.vocab_json = load_json(vocab_json)
        self.embedding_json = np_load(embedding_json)
        self.obj_matching = load_json(obj_matching)
        self.attr_matching = load_json(attr_matching)
        self.object_mapping = load_json(object_mapping)
        self.ques_list = ques_list

        self.len_labels = len(self.vocab_json['label2idx'])
        self.len_attr = len(self.vocab_json['attr2idx'])

        self.embed_size = 300
        self.attr_length = 622
        self.no_of_changes = no_of_changes
        self.allow_random = allow_random


    def getname2idx(self, name) :
        try :
            cid = self.vocab_json['label2idx'][name]
        except :
            try :
                name1 = singularize(name)
                cid = self.vocab_json['label2idx'][name1]
            except :
                name = name.rstrip('s')
                cid = self.vocab_json['label2idx'][name]
        return cid

    def getattr2idx(self, attr) :
        try :
            cid = self.vocab_json['attr2idx'][attr]
        except :
            attr1 = singularize(attr)
            cid = self.vocab_json['attr2idx'][attr1]

        return cid

    def box2embedding(self, box) :
        proj = nn.Linear(4, self.embed_size)
        box = torch.from_numpy(box)
        embed = proj(box)
        return embed

    def getembedding(self, cid, is_label=False) :
        embed = np.empty(self.embed_size)
        if is_label :
            embed = self.embedding_json[self.attr_length + cid]
        else :
            embed = self.embedding_json[cid]
        embed = [float(emb) for emb in embed]
        embed = np.asarray(embed)
        return embed

    def getobjects(self, imageid) :
        ## object mapping to object name
        scenegraphs_json = self.scenegraphs_json
        info_raw = scenegraphs_json[imageid]
        objects_raw = {}
        for obj in info_raw['objects'] :
            objects_raw[obj] = info_raw['objects'][obj]['name']
        return objects_raw

    def get_irrelevant_ques(self, imageid, objid) :
        irrelevant = []
        ques_list = self.ques_list
        for ques in ques_list[imageid] :
            if objid not in ques_list[imageid][ques]['annotations'].keys() :
                irrelevant.append(ques)
        return irrelevant

    def scene2embedding(self, imageid, ele) :
        #print (imageid)
        meta = dict()
        embeds = dict()
        scenegraphs_json = self.scenegraphs_json
        vocab_json = self.vocab_json
        meta['imageId'] = imageid
        
        info_raw = scenegraphs_json[imageid]
        meta['height'] = info_raw['height']
        meta['width'] = info_raw['width']

        boxes = {}
        changes = []

        for obj_number in info_raw['objects'] :
            obj = info_raw['objects'][obj_number]
            embeds[obj_number] = {}
            if obj_number == ele :
                continue
            else:
                obj_name = np.zeros(self.len_labels, dtype=np.float32)
                obj_attr = np.zeros(self.len_attr, dtype=np.float32)
                box = np.zeros(4, dtype=np.float32)
                name = obj['name']
                try: 
                    cid = self.getname2idx(name)
                    label_embed = self.getembedding(cid, is_label=True)
                    embeds[obj_number]['obj_embed'] = label_embed
                    cid = self.getname2idx(name)
                    obj_name[cid] = 1

                    embeds[obj_number]['attr_embed'] = []
                    for attr in obj['attributes'] :
                        cid = self.getattr2idx(attr)
                        attr_embed = self.getembedding(cid)
                        embeds[obj_number]['attr_embed'].append(attr_embed)
                        obj_attr[cid] = 1

                    box[0] = float(obj['x'])/meta['width']
                    box[1] = float(obj['y'])/meta['height']
                    box[2] = float(obj['x'] + obj['w'])/meta['width']
                    box[3] = float(obj['y'] + obj['h'])/meta['height']
                    boxes[obj_number] = box
                except :
                    pass

        for obj_number in info_raw['objects'] :
            obj = info_raw['objects'][obj_number]
            if obj_number == ele :
                obj_name = np.zeros(self.len_labels, dtype=np.float32)
                obj_attr = np.zeros(self.len_attr, dtype=np.float32)
                box = np.zeros(4, dtype=np.float32)
                name = obj['name']
                embeds[obj_number]['name'] = name
                try :
                    match_objs = self.obj_matching[name]
                except :
                    try :
                        name1 = singularize(name)
                        match_objs = self.obj_matching[name1]
                    except :
                        try : 
                            name = name.rstrip('s')
                            match_objs = self.obj_matching[name]
                        except :
                            continue
                #print ('obj : ', name)
                #print ('match_objs : ', match_objs)
                embeds[ele]['obj_embed'] = []
                embeds[ele]['attr_embed'] = []
                cid = self.getname2idx(name)
                try :
                    label_embed = self.getembedding(cid, is_label=True)
                    embeds[ele]['obj_embed'].append(label_embed)
                except :
                    label_embed = np.zeros(300)
                    embeds[ele]['obj_embed'].append(label_embed)
                change_text = 'original'
                changes.append(change_text)

                temp_attr = []
                for attr in obj['attributes'] :
                    cid = self.getattr2idx(attr)
                    attr_embed = self.getembedding(cid)
                    temp_attr.append(attr_embed)
                if temp_attr == [] :
                    temp_attr = label_embed

                embeds[ele]['attr_embed'].append(temp_attr)
                ## Getting object mapping which corresponds to where this object is present in entire dataset

                box[0] = float(obj['x'])/meta['width']
                box[1] = float(obj['y'])/meta['height']
                box[2] = float(obj['x'] + obj['w'])/meta['width']
                box[3] = float(obj['y'] + obj['h'])/meta['height']
                boxes[obj_number] = box
                ## Doing only maximum 5 perturbations per object
                count = 0
                for match_obj in match_objs :
                    try :
                        objects_map = self.object_mapping[match_obj]
                        cid = self.getname2idx(match_obj)
                        ## Pick a random attribute pair along with the object 
                        random_obj_map = random.choice(list(objects_map.keys()))
                        attributes = objects_map[random_obj_map]['attributes']

                        label_embed = self.getembedding(cid, is_label=True)
                        embeds[ele]['obj_embed'].append(label_embed)
                        change_text = str(name) + 'to' + str(match_obj)
                        changes.append(change_text)
                        temp_attr = []
                        for attr in attributes :
                            cid = self.getattr2idx(attr)
                            attr_embed = self.getembedding(cid)
                            temp_attr.append(attr_embed)
                        embeds[obj_number]['attr_embed'].append(temp_attr)
                        count += 1

                        if count == self.no_of_changes :
                            break

                    except :
                        pass

                # Selecting random objects to replace object of interest if matching object are not sufficient
                if self.allow_random :
                    while (count < self.no_of_changes) :
                        try :
                            match_obj = random.choice(list(self.object_mapping.keys()))
                            objects_map = self.object_mapping[match_obj]
                            cid = self.getname2idx(match_obj)
                            ## Pick a random attribute pair along with the object 
                            random_obj_map = random.choice(list(objects_map.keys()))
                            attributes = objects_map[random_obj_map]['attributes']

                            label_embed = self.getembedding(cid, is_label=True)
                            embeds[ele]['obj_embed'].append(label_embed)
                            change_text = str(name) + 'to' + str(match_obj)
                            changes.append(change_text)
                            temp_attr = []
                            for attr in attributes :
                                cid = self.getattr2idx(attr)
                                attr_embed = self.getembedding(cid)
                                temp_attr.append(attr_embed)
                            embeds[obj_number]['attr_embed'].append(temp_attr)
                            count += 1

                        except :
                            pass
        return embeds, boxes, changes
        
    def extractembeddings(self,images_list, mapping) :
        final_embeddings = mapping
        images = images_list
        i=0
        print ('GQA embedding extraction in place')
        for image in images :
            final_embeddings[image] = {}
            objects_raw = self.getobjects(image)
            for obj in objects_raw :
                irrelevant_ques = self.get_irrelevant_ques(image, obj)
                embeddings, bboxes, changes = self.scene2embedding(image, obj)
                #embeddings = embeddings.astype(np.double)
                final_embeddings[image][obj] = {}
                final_embeddings[image][obj]['objandattr'] = embeddings
                final_embeddings[image][obj]['bboxes'] = bboxes
                final_embeddings[image][obj]['changes'] = changes
                final_embeddings[image][obj]['relevant_questions'] = irrelevant_ques

           
            i += 1
            if i%100 == 0:
                print (str(i) + ' / ' + str(len(images)))
           
        return final_embeddings


        
