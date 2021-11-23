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
    def __init__(self, scenegraphs_json, vocab_json, embedding_json, obj_matching, attr_matching,object_mapping, ques_list) :
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


    def getobjects(self, imageid) :
        ## object mapping to object name
        scenegraphs_json = self.scenegraphs_json
        info_raw = scenegraphs_json[imageid]
        objects_raw = {}
        for obj in info_raw['objects'] :
            objects_raw[obj] = info_raw['objects'][obj]['name']
        return objects_raw

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

    def get_unrelevant_objects(self, info, ques, objects_raw) :
        unrelevant_objects = {}
        annotations = info[ques]['annotations']
        for obj in objects_raw :
            if obj not in annotations :
                unrelevant_objects[obj] = {}

        return unrelevant_objects

    def getembedding(self, cid, is_label=False) :
        embed = np.empty(self.embed_size)
        if is_label :
            embed = self.embedding_json[self.attr_length + cid]
        else :
            embed = self.embedding_json[cid]
        embed = [float(emb) for emb in embed]
        embed = np.asarray(embed)
        return embed


    def scene2embedding_swapmix(self, imageid, unrelevant_objects) :
        meta = dict()
        embeds = dict()
        scenegraphs_json = self.scenegraphs_json
        vocab_json = self.vocab_json
        meta['imageId'] = imageid
        
        info_raw = scenegraphs_json[imageid]
        meta['height'] = info_raw['height']
        meta['width'] = info_raw['width']

        boxes = {}

        for obj_id in info_raw['objects'] :
            embeds[obj_id] = {}
            obj = info_raw['objects'][obj_id]
            obj_name = np.zeros(self.len_labels, dtype=np.float32)
            obj_attr = np.zeros(self.len_attr, dtype=np.float32)
            box = np.zeros(4, dtype=np.float32)
            name = obj['name']
            embeds[obj_id]['name'] = name
            embeds[obj_id]['attr_embed'] = []

            try : 

                ## Selecting if the embeddings needs to be swapped with a different object with probability 0.5
                embed_change = random.randint(0,1)
                if (obj_id not in unrelevant_objects) or embed_change != 0 :
                    cid = self.getname2idx(name)
                    label_embed = self.getembedding(cid, is_label=True)
                    embeds[obj_id]['obj_embed'] = label_embed
                    obj_name[cid] = 1
                    for attr in obj['attributes'] :
                        cid = self.getattr2idx(attr)
                        attr_embed = self.getembedding(cid)
                        embeds[obj_id]['attr_embed'].append(attr_embed)
                else :
                    ## Selecting if object needs to be swapped or attributes
                    obj_change = random.randint(0,1)
                    if obj_change == 0 :
                        try :
                            match_objs = self.obj_matching[name]
                        except :
                            name1 = singularize(name)
                            match_objs = self.obj_matching[name1]

                        swap_obj = random.choice(list(match_objs.keys()))
                        cid = self.getname2idx(swap_obj)
                        label_embed = self.getembedding(cid, is_label=True)
                        embeds[obj_id]['obj_embed'] = label_embed
                        obj_name[cid] = 1
                        try : 
                            objects_map = self.object_mapping[swap_obj]
                            random_obj_map = random.choice(list(objects_map.keys()))
                            attributes = objects_map[random_obj_map]['attributes']
                        except :
                            attributes = obj['attributes']
                        for attr in attributes :
                            cid = self.getattr2idx(attr)
                            attr_embed = self.getembedding(cid)
                            embeds[obj_id]['attr_embed'].append(attr_embed)


                    else :
                        ## Attributes are swapped with prob 0.5
                        cid = self.getname2idx(name)
                        label_embed = self.getembedding(cid, is_label=True)
                        embeds[obj_id]['obj_embed'] = label_embed
                        obj_name[cid] = 1

                        for attr in obj['attributes'] :
                            attrs_match = self.attr_matching[attr]
                            attr_swap = random.choice(list(attrs_match.keys()))
                            cid = self.getattr2idx(attr_swap)
                            attr_embed = self.getembedding(cid)
                            embeds[obj_id]['attr_embed'].append(attr_embed)

                box[0] = float(obj['x'])/meta['width']
                box[1] = float(obj['y'])/meta['height']
                box[2] = float(obj['x'] + obj['w'])/meta['width']
                box[3] = float(obj['y'] + obj['h'])/meta['height']
                boxes[obj_id] = box

            except:
                continue

        return embeds, boxes
       


    ## This function is used only for swapmix training
    ## It gets unrelevant objects for each question
    ''' This cannot be used as embeddings needs to change at every epoch'''
    def extractembeddings_swapmix(self,images_list, mapping) :
        final_embeddings = mapping
        images = json.load(open(images_list))
        i=0
        for image in images :
            final_embeddings[image] = {}
            info = self.ques_list[image]
            questions = list(info.keys())
            objects_raw = self.getobjects(image)
            for ques in questions :
                final_embeddings[image][ques] = {}
                unrelevant_objects = self.get_unrelevant_objects(info, ques, objects_raw)
                embeddings, bboxes = self.scene2embedding_swapmix(image, unrelevant_objects)
                final_embeddings[image][ques]['objandattr'] = embeddings
                final_embeddings[image][ques]['bboxes'] = bboxes

            i +=1
            if i % 500 == 0 :
                print (str(i) + ' / ' + str(len(images)))

        
        return final_embeddings

    def extractembeddings_swapmix_single(self,image, ques) :

        final_embeddings = {}
        info = self.ques_list[image]
        objects_raw = self.getobjects(image)
        unrelevant_objects = self.get_unrelevant_objects(info, ques, objects_raw)
        embeddings, bboxes = self.scene2embedding_swapmix(image, unrelevant_objects)
        final_embeddings['objandattr'] = embeddings
        final_embeddings['bboxes'] = bboxes

        return final_embeddings
