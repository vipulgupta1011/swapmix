import os
import numpy as np
import torch
import torch.nn as nn
import json
import pdb
from inflection import singularize as inf_singularize
from pattern3.text.en import singularize as pat_singularize
from pattern3.text.en import singularize
import random

def load_json(fname) :
    json_dict = json.load(open(fname))

    return json_dict

def np_load(fname) :
    np_array = np.load(fname)

    return np_array


class GQASceneDataset() :
    def __init__(self, scenegraphs_json, vocab_json, embedding_json, obj_matching, attr_matching, object_mapping, ques_list, no_of_changes, allow_random) :
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

    def getobjects(self, imageid) :
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


    '''' Input  : imageid and object ele
         Output : Feature vectors for perturbations on ele, changes list and bounding box 
         Processing : First create a label and attr embeddings for all objecxts except ele. 
                      The do no_of_changes perturbations on ele and make a new feature vector for each perturbations.
    ''' 

    def scene2embedding(self, imageid, ele) :
        #print (imageid)
        meta = dict()
        labels_embeds = dict()
        attrs_embeds = dict()
        zero_embedding = np.asarray([0]*300)
        zero_box = np.asarray([0]*4)
        
        scenegraphs_json = self.scenegraphs_json
        vocab_json = self.vocab_json
        meta['imageId'] = imageid
        
        info_raw = scenegraphs_json[imageid]
        meta['height'] = info_raw['height']
        meta['width'] = info_raw['width']

        boxes = []
        changes = []
        labels_embeddings = []
        attr_embeddings = []
        ## Make labels and attr embedding list for all objects except ele
        for i, obj_number in enumerate(info_raw['objects'].keys()) :
            obj = info_raw['objects'][obj_number]
            if obj_number == ele :
                continue
            else :
                obj_name = np.zeros(self.len_labels, dtype=np.float32)
                obj_attr = np.zeros(self.len_attr, dtype=np.float32)
                box = np.zeros(4, dtype=np.float32)
                name = obj['name']

                try :
                    cid = self.getname2idx(name)
                    label_embed = self.getembedding(cid, is_label=True)
                    labels_embeddings.append(label_embed)
                    obj_name[cid] = 1

                    obj_attr_embeds = zero_embedding.copy()
                    for attr in obj['attributes'] :
                        if not attr:
                            attr_embed = zero_embedding.copy()
                        else :
                            cid = self.getattr2idx(attr)
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

        
        ## Find a label and attr aembeddings for matching objects and add it to object and attr embeddings:
        for i, obj_number in enumerate(info_raw['objects'].keys()) :
            obj = info_raw['objects'][obj_number]
            if obj_number == ele :
                obj_name = np.zeros(self.len_labels, dtype=np.float32)
                obj_attr = np.zeros(self.len_attr, dtype=np.float32)
                box = np.zeros(4, dtype=np.float32)
                name = obj['name']
                box[0] = float(obj['x'])/meta['width']
                box[1] = float(obj['y'])/meta['height']
                box[2] = float(obj['x'] + obj['w'])/meta['width']
                box[3] = float(obj['y'] + obj['h'])/meta['height']
                boxes.append(box)

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
                
                ##  First feature in the list is normal embeddings without perturbations
                try :
                    labels_embeddings_rest = labels_embeddings.copy()
                    attr_embeddings_rest = attr_embeddings.copy()
                    cid = self.getname2idx(name)
                    label_embed = self.getembedding(cid, is_label=True)
                    labels_embeddings_rest.append(label_embed)
                    obj_attr_embeds = zero_embedding.copy()
                    for attr in obj['attributes'] :
                        if not attr:
                            attr_embed = zero_embedding.copy()
                        else :
                            cid = self.getattr2idx(attr)
                            attr_embed = self.getembedding(cid)
                        obj_attr_embeds = np.add(obj_attr_embeds,attr_embed)
                    if len(obj['attributes']) != 0: 
                        obj_attr_embeds = obj_attr_embeds/len(obj['attributes'])

                    attr_embeddings_rest.append(obj_attr_embeds)
                    
                    change_text = 'original'
                    changes.append(change_text)
                except :
                    continue
                labels_embeds[len(labels_embeds)] = labels_embeddings_rest
                attrs_embeds[len(attrs_embeds)] = attr_embeddings_rest
                

                count = 0
                ## Iterating over match objs and saving embeddings for each match obj
                for match_obj in match_objs :
                    labels_embeddings_rest = labels_embeddings.copy()
                    attr_embeddings_rest = attr_embeddings.copy()
                    try :
                        objects_map = self.object_mapping[match_obj]
                        cid = self.getname2idx(match_obj)
                        ## Pick a random attribute pair along with the object 
                        random_obj_map = random.choice(list(objects_map.keys()))
                        attributes = objects_map[random_obj_map]['attributes']

                        label_embed = self.getembedding(cid, is_label=True)
                        labels_embeddings_rest.append(label_embed)
                        change_text = str(name) + 'to' + str(match_obj)
                        changes.append(change_text)
                        obj_attr_embeds = zero_embedding.copy()
                        for attr in attributes :
                            if not attr:
                                attr_embed = zero_embedding.copy()
                            else :
                                cid = self.getattr2idx(attr)
                                attr_embed = self.getembedding(cid)
                            obj_attr_embeds = np.add(obj_attr_embeds,attr_embed)
                        if len(attributes) != 0 :
                            obj_attr_embeds = obj_attr_embeds/len(attributes)

                        attr_embeddings_rest.append(obj_attr_embeds)
                        
                        labels_embeds[len(labels_embeds)] = labels_embeddings_rest
                        attrs_embeds[len(attrs_embeds)] = attr_embeddings_rest

                        count += 1

                        if count == self.no_of_changes :
                            break

                    except :
                        pass

                # Selecting random objects to replace object of interest if matching object are not sufficient
                if self.allow_random :
                    while (count < self.no_of_changes) :
                        try :
                            labels_embeddings_rest = labels_embeddings.copy()
                            attr_embeddings_rest = attr_embeddings.copy()

                            match_obj = random.choice(list(self.object_mapping.keys()))
                            objects_map = self.object_mapping[match_obj]
                            cid = self.getname2idx(match_obj)
                            ## Pick a random attribute pair along with the object 
                            random_obj_map = random.choice(list(objects_map.keys()))
                            attributes = objects_map[random_obj_map]['attributes']

                            label_embed = self.getembedding(cid, is_label=True)
                            labels_embeddings_rest.append(label_embed)
                            change_text = str(name) + 'to' + str(match_obj)
                            changes.append(change_text)
                            obj_attr_embeds = zero_embedding.copy()
                            for attr in attributes :
                                if not attr:
                                    attr_embed = zero_embedding.copy()
                                else :
                                    cid = self.getattr2idx(attr)
                                    attr_embed = self.getembedding(cid)
                                obj_attr_embeds = np.add(obj_attr_embeds,attr_embed)
                            if len(attributes) != 0 :
                                obj_attr_embeds = obj_attr_embeds/len(attributes)

                            attr_embeddings_rest.append(obj_attr_embeds)
                            
                            labels_embeds[len(labels_embeds)] = labels_embeddings_rest
                            attrs_embeds[len(attrs_embeds)] = attr_embeddings_rest

                            count += 1

                        except :
                            pass

       
        for idx in labels_embeds :
            labels_embeddings = labels_embeds[idx] 
            if len(labels_embeddings) < 36:
                for i in range(36 - len(labels_embeddings)):
                    labels_embeddings.append(zero_embedding)
            else:
                labels_embeddings = labels_embeddings[:36]
            labels_embeds[idx] = labels_embeddings 
        
        for idx in attrs_embeds :
            attrs_embeddings = attrs_embeds[idx] 
            if len(attr_embeddings) < 36:
                for i in range(36 - len(attr_embeddings)):
                    attr_embeddings.append(zero_embedding)
            else:
                attr_embeddings = attr_embeddings[:36]
            attrs_embeds[idx] = attr_embeddings 

        if len(boxes) < 36:
            for i in range(36 - len(boxes)):
                boxes.append(zero_box)
        else:
            boxes = boxes[:36]
        #embeddings = labels_embeddings + attr_embeddings
        #len_embedding = len(embeddings)
        embeddings = {}
        for idx in labels_embeds :
            labels_embeddings = labels_embeds[idx]
            attr_embeddings = attrs_embeds[idx]
            out = np.zeros((36,300))
            for i in range(36) :
               out[i] = np.add(labels_embeddings[i], attr_embeddings[i])

            out = np.array(out, np.float32)
            embeddings[len(embeddings)] = out

        boxes = np.array(boxes, np.float32)

        return embeddings, boxes, changes
        
    def extractembeddings(self,images_list) :
        final_embeddings = {}
        images = images_list
        print ('GQA embedding extraction in place')
        i = 0
        for image in images :
            final_embeddings[image] = {}
            objects_raw = self.getobjects(image)
            for obj in objects_raw :
                ## Extracting questions for which object is irrelevant
                irrelevant_ques = self.get_irrelevant_ques(image, obj)
                embeddings, bboxes, changes = self.scene2embedding(image, obj)
                final_embeddings[image][obj] = {}
                final_embeddings[image][obj]['objandattr'] = embeddings
                final_embeddings[image][obj]['bboxes'] = bboxes
                final_embeddings[image][obj]['changes'] = changes
                final_embeddings[image][obj]['relevant_questions'] = irrelevant_ques

            i += 1
            if i%100 == 0:
                print (str(i) + ' / ' + str(len(images)))
           
        return final_embeddings
