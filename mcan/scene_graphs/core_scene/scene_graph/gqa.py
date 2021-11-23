import os
from core_scene.scene_graph.utils import load_json, np_load
import numpy as np
import torch
import torch.nn as nn
import json
import pdb
from pattern3.text.en import singularize

class GQADataset() :
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


    def scene2embedding(self, imageid) :
        #print (imageid)
        meta = dict()
        embeds = dict()
        scenegraphs_json = self.scenegraphs_json
        vocab_json = self.vocab_json
        meta['imageId'] = imageid
        
        info_raw = scenegraphs_json[imageid]
        meta['height'] = info_raw['height']
        meta['width'] = info_raw['width']

        objects = []
        objects_name = []
        objects_attr = []
        boxes = []
        labels_embeddings = []
        attr_embeddings = []
        boxes_embed = []

        for obj_id in info_raw['objects'] :
            embeds[obj_id] = {}
            obj = info_raw['objects'][obj_id]
            obj_name = np.zeros(self.len_labels, dtype=np.float32)
            obj_attr = np.zeros(self.len_attr, dtype=np.float32)
            box = np.zeros(4, dtype=np.float32)
            name = obj['name']
            embeds[obj_id]['name'] = name

            try : 

                cid = self.getname2idx(name)
                label_embed = self.getembedding(cid, is_label=True)
                embeds[obj_id]['obj_embed'] = label_embed
                #labels_embeddings.append(label_embed)
                obj_name[cid] = 1

                embeds[obj_id]['attr_embed'] = []
                for attr in obj['attributes'] :
                    cid = self.getattr2idx(attr)
                    attr_embed = self.getembedding(cid)
                    embeds[obj_id]['attr_embed'].append(attr_embed)
                    #attr_embeddings.append(attr_embed)
                    obj_attr[cid] = 1
                #pdb.set_trace()
                #objects_name.append(obj_name)
                #objects_attr.append(obj_attr)

                box[0] = float(obj['x'])/meta['width']
                box[1] = float(obj['y'])/meta['height']
                box[2] = float(obj['x'] + obj['w'])/meta['width']
                box[3] = float(obj['y'] + obj['h'])/meta['height']
                boxes.append(box)

            except :
                continue


        #embeddings = labels_embeddings + attr_embeddings
        #len_embedding = len(embeddings)
        #out = np.zeros((len_embedding,300))
        #for i in range(len_embedding) :
        #    out[i] = embeddings[i]

        return embeds, boxes
        
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

