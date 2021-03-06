import os
import numpy as np
import torch
import torch.nn as nn
import json
import pdb
from inflection import singularize as inf_singularize
from pattern3.text.en import singularize as pat_singularize
import random

def load_json(fname) :
    json_dict = json.load(open(fname))

    return json_dict

def np_load(fname) :
    np_array = np.load(fname)

    return np_array


class GQASceneDataset() :
    def __init__(self, scenegraphs_json, vocab_json, embedding_json) :
        self.scenegraphs_json = load_json(scenegraphs_json)
        self.vocab_json = load_json(vocab_json)
        self.embedding_json = np_load(embedding_json)

        self.len_labels = len(self.vocab_json['label2idx'])
        self.len_attr = len(self.vocab_json['attr2idx'])

        self.embed_size = 300
        self.attr_length = 622


    def getname2idx(self, name) :
        try :
            cid = self.vocab_json['label2idx'][name]
        except :
            try :
                try:
                    name1 = inf_singularize(name)
                    cid = self.vocab_json['label2idx'][name1]
                except:
                    name2 = pat_singularize(name)
                    cid = self.vocab_json['label2idx'][name2]
            except :
                name = name.rstrip('s')
                cid = self.vocab_json['label2idx'][name]
        return cid

    def getattr2idx(self, attr) :
        try :
            cid = self.vocab_json['attr2idx'][attr]
        except :
            try:
                try:
                    attr1 = inf_singularize(attr)
                    cid = self.vocab_json['attr2idx'][attr1]
                except:
                    attr2 = pat_singularize(attr)
                    cid = self.vocab_json['attr2idx'][attr2]
            except :
                name = attr.rstrip('s')
                cid = self.vocab_json['label2idx'][name]

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


    def scene2embedding(self, imageid) :
        #print (imageid)
        meta = dict()
        # embeds = dict()
        zero_embedding = np.asarray([0]*300)
        zero_box = np.asarray([0]*4)
        
        scenegraphs_json = self.scenegraphs_json
        vocab_json = self.vocab_json
        meta['imageId'] = imageid
        
        info_raw = scenegraphs_json[imageid]
        meta['height'] = info_raw['height']
        meta['width'] = info_raw['width']

        objects = []
        objects_name = []
        objects_attr = []
        # boxes = [[0,0,0,0]]*36
        boxes = []
        labels_embeddings = []
        attr_embeddings = []
        boxes_embed = []

        for i,obj_id in enumerate(info_raw['objects'].keys()) :
            # embeds[obj_id] = {}
            obj = info_raw['objects'][obj_id]
            obj_name = np.zeros(self.len_labels, dtype=np.float32)
            obj_attr = np.zeros(self.len_attr, dtype=np.float32)
            box = np.zeros(4, dtype=np.float32)
            name = obj['name']
            obj_attr_embeds = zero_embedding
            # embeds[obj_id]['name'] = name

            try : 

                cid = self.getname2idx(name)
                label_embed = self.getembedding(cid, is_label=True)
                labels_embeddings.append(label_embed)
                obj_name[cid] = 1

                # embeds[obj_id]['attr_embed'] = []
                for attr in obj['attributes'] :
                    if not attr:
                        attr_embed = zero_embedding
                    else:
                        cid = self.getattr2idx(attr)
                        attr_embed = self.getembedding(cid)
                    obj_attr_embeds = np.add(obj_attr_embeds,attr_embed)
                    
                    obj_attr[cid] = 1
                #pdb.set_trace()
                #objects_name.append(obj_name)
                #objects_attr.append(obj_attr)
                if len(obj['attributes']) != 0:
                    obj_attr_embeds = obj_attr_embeds/len(obj['attributes'])
                attr_embeddings.append(obj_attr_embeds)

                box[0] = abs(float(obj['x'])/meta['width'])
                box[1] = abs(float(obj['y'])/meta['height'])
                box[2] = abs(float(obj['x'] + obj['w'])/meta['width'])
                box[3] = abs(float(obj['y'] + obj['h'])/meta['height'])
                boxes.append(box)

            except :
                continue
        
        # print("image id:", imageid)
        # print("lenth of boxes:", len(boxes))
        # print("\n")
        # print("lenth of obj embeds:", len(labels_embeddings))
        # print("\n")
        # print("lenth of attr embeds:", len(attr_embeddings))
        # print("\n")

        
        if len(labels_embeddings) < 36:
            for i in range(36 - len(labels_embeddings)):
                labels_embeddings.append(zero_embedding)
        else:
            labels_embeddings = labels_embeddings[:36] 
        
        if len(attr_embeddings) < 36:
            for i in range(36 - len(attr_embeddings)):
                attr_embeddings.append(zero_embedding)
        else:
            attr_embeddings = attr_embeddings[:36] 

        if len(boxes) < 36:
            for i in range(36 - len(boxes)):
                boxes.append(zero_box)
        else:
            boxes = boxes[:36]
        #embeddings = labels_embeddings + attr_embeddings
        #len_embedding = len(embeddings)
        out = np.zeros((36,300))
        for i in range(36) :
           out[i] = np.add(labels_embeddings[i], attr_embeddings[i])

        return out, boxes
        
    def extractembeddings(self,images_list, mapping) :
        final_embeddings = mapping
        images = json.load(open(images_list))
        i=0
        for image in images :
            embeddings, bboxes = self.scene2embedding(image)
            #embeddings = embeddings.astype(np.double)
            final_embeddings[image] = {}
            final_embeddings[image]['objandattr'] = embeddings
            final_embeddings[image]['bboxes'] = bboxes
           
           # i += 1
           # if i>250 :
           #     break
        return final_embeddings


class GQASceneDataset_swapmix() :
    def __init__(self, scenegraphs_json, vocab_json, embedding_json, obj_matching, attr_matching, object_mapping, ques_list) :
        self.scenegraphs_json = load_json(scenegraphs_json)
        self.vocab_json = load_json(vocab_json)
        self.embedding_json = np_load(embedding_json)
        self.obj_matching = load_json(obj_matching)
        self.attr_matching = load_json(attr_matching)
        self.object_mapping = load_json(object_mapping)
        self.ques_list = load_json(ques_list)

        self.len_labels = len(self.vocab_json['label2idx'])
        self.len_attr = len(self.vocab_json['attr2idx'])

        self.embed_size = 300
        self.attr_length = 622

    def getobjects(self, imageid) :
        scenegraphs_json = self.scenegraphs_json
        info_raw = scenegraphs_json[imageid]
        objects_raw = {}
        for obj in info_raw['objects'] :
            objects_raw[obj] = info_raw['objects'][obj]['name']
        return objects_raw

    def get_unrelevant_objects(self, info, ques, objects_raw) :
        unrelevant_objects = {}
        annotations = info[ques]['annotations']
        for obj in objects_raw :
            if obj not in annotations :
                unrelevant_objects[obj] = {}
        return unrelevant_objects

    def getname2idx(self, name) :
        try :
            cid = self.vocab_json['label2idx'][name]
        except :
            try :
                try:
                    name1 = inf_singularize(name)
                    cid = self.vocab_json['label2idx'][name1]
                except:
                    name2 = pat_singularize(name)
                    cid = self.vocab_json['label2idx'][name2]
            except :
                name = name.rstrip('s')
                cid = self.vocab_json['label2idx'][name]
        return cid

    def getattr2idx(self, attr) :
        try :
            cid = self.vocab_json['attr2idx'][attr]
        except :
            try:
                try:
                    attr1 = inf_singularize(attr)
                    cid = self.vocab_json['attr2idx'][attr1]
                except:
                    attr2 = pat_singularize(attr)
                    cid = self.vocab_json['attr2idx'][attr2]
            except :
                name = attr.rstrip('s')
                cid = self.vocab_json['label2idx'][name]

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


    def scene2embedding_swapmix(self, imageid, unrelevant_objects) :
        #print (imageid)
        meta = dict()
        embeds = dict()
        zero_embedding = np.asarray([0]*300)
        zero_box = np.asarray([0]*4)
        
        scenegraphs_json = self.scenegraphs_json
        vocab_json = self.vocab_json
        meta['imageId'] = imageid
        
        info_raw = scenegraphs_json[imageid]
        meta['height'] = info_raw['height']
        meta['width'] = info_raw['width']

        objects = []
        objects_name = []
        objects_attr = []
        # boxes = [[0,0,0,0]]*36
        boxes = []
        labels_embeddings = []
        attr_embeddings = []
        boxes_embed = []

        for i,obj_id in enumerate(info_raw['objects'].keys()) :
            obj = info_raw['objects'][obj_id]
            obj_name = np.zeros(self.len_labels, dtype=np.float32)
            obj_attr = np.zeros(self.len_attr, dtype=np.float32)
            box = np.zeros(4, dtype=np.float32)
            name = obj['name']
            obj_attr_embeds = zero_embedding

            try : 

                ## Selecting if the embeddings needs to be swapped with a different object with probability 0.5
                embed_change = random.randint(0,1)
                if (obj_id not in unrelevant_objects) or embed_change != 0 :
                    cid = self.getname2idx(name)
                    label_embed = self.getembedding(cid, is_label=True)
                    labels_embeddings.append(label_embed)
                    obj_name[cid] = 1
                    for attr in obj['attributes'] :
                        if not attr:
                            attr_embed = zero_embedding
                        else :
                            cid = self.getattr2idx(attr)
                            attr_embed = self.getembedding(cid)
                        obj_attr_embeds = np.add(obj_attr_embeds,attr_embed)
                    
                    if len(obj['attributes']) != 0: 
                        obj_attr_embeds = obj_attr_embeds/len(obj['attributes'])
                    attr_embeddings.append(obj_attr_embeds)

                else :
                    ## Selecting if object classneeds to be swapped or attributes
                    obj_change = random.randint(0,1)
                    if obj_change == 0 :
                        try :
                            match_objs = self.obj_matching[name]
                        except :
                            name1 = pat_singularize(name)
                            match_objs = self.obj_matching[name1]

                        if match_objs == {} :
                            swap_obj = name
                        else :
                            swap_obj = random.choice(list(match_objs.keys()))
                        cid = self.getname2idx(swap_obj)
                        label_embed = self.getembedding(cid, is_label=True)
                        labels_embeddings.append(label_embed)
                        obj_name[cid] = 1
                        try :            
                            objects_map = self.object_mapping[swap_obj]
                            random_obj_map = random.choice(list(objects_map.keys()))
                            attributes = objects_map[random_obj_map]['attributes']
                        except :
                            attributes = obj['attributes']
                        for attr in attributes :
                            if not attr:
                                attr_embed = zero_embedding
                            else :
                                cid = self.getattr2idx(attr)
                                attr_embed = self.getembedding(cid)
                            obj_attr_embeds = np.add(obj_attr_embeds,attr_embed)
                        
                        if len(obj['attributes']) != 0: 
                            obj_attr_embeds = obj_attr_embeds/len(obj['attributes'])
                        attr_embeddings.append(obj_attr_embeds)


                    else :
                        ## Attributes are swapped with prob 0.5
                        cid = self.getname2idx(name)
                        label_embed = self.getembedding(cid, is_label=True)
                        labels_embeddings.append(label_embed)
                        obj_name[cid] = 1

                        for attr in obj['attributes'] :
                            if not attr:
                                attr_embed = zero_embedding
                            else :
                                attrs_match = self.attr_matching[attr]
                                if attrs_match == {} :
                                    attr_swap = attr
                                else :
                                    attr_swap = random.choice(list(attrs_match.keys()))
                                cid = self.getattr2idx(attr_swap)
                                attr_embed = self.getembedding(cid)
                            obj_attr_embeds = np.add(obj_attr_embeds,attr_embed)
                        
                        if len(obj['attributes']) != 0: 
                            obj_attr_embeds = obj_attr_embeds/len(obj['attributes'])
                        attr_embeddings.append(obj_attr_embeds)

                box[0] = float(obj['x'])/meta['width']
                box[1] = float(obj['y'])/meta['height']
                box[2] = float(obj['x'] + obj['w'])/meta['width']
                box[3] = float(obj['y'] + obj['h'])/meta['height']
                boxes.append(box)

            except :
                continue

        
        if len(labels_embeddings) < 36:
            for i in range(36 - len(labels_embeddings)):
                labels_embeddings.append(zero_embedding)
        else:
            labels_embeddings = labels_embeddings[:36] 
        
        if len(attr_embeddings) < 36:
            for i in range(36 - len(attr_embeddings)):
                attr_embeddings.append(zero_embedding)
        else:
            attr_embeddings = attr_embeddings[:36] 

        if len(boxes) < 36:
            for i in range(36 - len(boxes)):
                boxes.append(zero_box)
        else:
            boxes = boxes[:36]
        #embeddings = labels_embeddings + attr_embeddings
        #len_embedding = len(embeddings)
        out = np.zeros((36,300))
        for i in range(36) :
           out[i] = np.add(labels_embeddings[i], attr_embeddings[i])

        out = np.array(out, np.float32)
        boxes = np.array(boxes, np.float32)
        return out, boxes
        
    def extractembeddings_swapmix_single(self,image, ques) :
        final_embeddings = {}
        info = self.ques_list[image]
        objects_raw = self.getobjects(image)
        unrelevant_objects = self.get_unrelevant_objects(info, ques, objects_raw)
        embeddings, bboxes = self.scene2embedding_swapmix(image, unrelevant_objects)
           
        return embeddings, bboxes
