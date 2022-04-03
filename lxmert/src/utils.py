# coding=utf-8
# Copyleft 2019 Project LXRT

import sys
import csv
import base64
import time
import pdb
import numpy as np

from tasks.dataloader.scene_data_init import GQASceneDataset, load_json

csv.field_size_limit(sys.maxsize)
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]


def load_obj_tsv(fname, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data



''' To be used only with val_mapping_new as it was created with this function call  '''
def load_obj_tsv_selective(fname, images, topk=None):
    """Load object features from tsv file.

    :param fname: The path to the tsv file.
    :param topk: Only load features for top K images (lines) in the tsv file.
        Will load all the features if topk is either -1 or None.
    :return: A list of image object features where each feature is a dict.
        See FILENAMES above for the keys in the feature dict.
    """
    data = []
    start_time = time.time()
    print("Start to load Faster-RCNN detected objects from %s" % fname)
    with open(fname) as f:
        reader = csv.DictReader(f, FIELDNAMES, delimiter="\t")
        for i, item in enumerate(reader):

            if item['img_id'] not in images :
                continue
            for key in ['img_h', 'img_w', 'num_boxes']:
                item[key] = int(item[key])
            
            boxes = item['num_boxes']
            decode_config = [
                ('objects_id', (boxes, ), np.int64),
                ('objects_conf', (boxes, ), np.float32),
                ('attrs_id', (boxes, ), np.int64),
                ('attrs_conf', (boxes, ), np.float32),
                ('boxes', (boxes, 4), np.float32),
                ('features', (boxes, -1), np.float32),
            ]
            for key, shape, dtype in decode_config:
                item[key] = np.frombuffer(base64.b64decode(item[key]), dtype=dtype)
                item[key] = item[key].reshape(shape)
                item[key].setflags(write=False)

            data.append(item)
            if topk is not None and len(data) == topk:
                break
    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


def load_scene_graphs(scenegraphs_json_path, vocab_json, embedding_json, topk=None):
    """
    An example of scene graph

    {
        "2407890": {
            "width": 640,
            "height": 480,
            "location": "living room",
            "weather": none,
            "objects": {
                "271881": {
                    "name": "chair",
                    "x": 220,
                    "y": 310,
                    "w": 50,
                    "h": 80,
                    "attributes": ["brown", "wooden", "small"],
                    "relations": {
                        "32452": {
                            "name": "on",
                            "object": "275312"
                        },
                        "32452": {
                            "name": "near",
                            "object": "279472"
                        }                    
                    }
                }
            }
        }
    }
    """

    data = []
    start_time = time.time()
    print("Start to load scene graphs from %s" % scenegraphs_json_path)

    GQA_with_scene_graph = GQASceneDataset(scenegraphs_json_path, vocab_json, embedding_json)
    scenegraphs_json = GQA_with_scene_graph.scenegraphs_json

    for imageid in scenegraphs_json.keys():
        item = dict()
        embeds, boxes = GQA_with_scene_graph.scene2embedding(imageid) 
        item["img_id"] = imageid

        info_raw = scenegraphs_json[imageid]
        item["img_h"] = int(info_raw['height'])
        item["img_w"] = int(info_raw['width'])

        objects_id = []
        attrs_id = []
        for obj_id in info_raw['objects']:
            obj = info_raw['objects'][obj_id]
            name = obj['name']
            try:
                obj_cid = GQA_with_scene_graph.getname2idx(name)
            except:
                obj_cid = 0
            objects_id.append(int(obj_cid))
            for attr in obj['attributes'] :
                try:
                    attr_cid = GQA_with_scene_graph.getattr2idx(attr)
                except:
                    attr_cid = 0
                attrs_id.append(attr_cid)

        # objects_id is the one-hot encoder
        # need to make the object_id of different images to be the same size
        # Truncated and filled (Threshold = 36)
        if len(objects_id) < 36:
            for i in range(36 - len(objects_id)):
                objects_id.append(0)
        else:
            objects_id = objects_id[:36]    

        if len(attrs_id) < 36:
            for i in range(36 - len(attrs_id)):
                attrs_id.append(0)
        else:
            attrs_id = attrs_id[:36]


        item["objects_id"] = objects_id
        item["objects_conf"] = [1]*36
        item["attrs_id"] = attrs_id
        item["attrs_conf"] = [1]*36
        item["num_boxes"] = len(boxes)
        item["boxes"] = boxes
        item["features"] = embeds

        # num_boxes = item['num_boxes']
        # decode_config = [
        #     ('objects_id',  (num_boxes, ), np.int64),
        #     ('objects_conf', (num_boxes, ), np.float32),
        #     ('attrs_id', (num_boxes, ), np.int64),
        #     ('attrs_conf', (num_boxes, ), np.float32),
        #     ('boxes', (num_boxes, 4), np.float32),
        #     ('features', (num_boxes, -1), np.float32), # is the embeddings
        # ]
        num_boxes = item['num_boxes']
        decode_config = [
            ('objects_id', np.int64),
            ('objects_conf',  np.float32),
            ('attrs_id',  np.int64),
            ('attrs_conf',  np.float32),
            ('boxes', np.float32),
            ('features', np.float32), # is the embeddings
        ]

    

        for key, datatype in decode_config:
            # print("%s data type:" % key, type(item[key]))
            # print("%s shape:" % key, len(item[key]))
            # if not key == 'features':
            #     print("%s:" % key, item[key])
            # else:
            #     for i in range(len(item[key])):
            #         print("%s[%d]:" % (key, i), len(item[key][i]))
                
            item[key] = np.array(item[key], dtype = datatype)
            # item[key] = item[key].reshape(shape)
            item[key].setflags(write=False)
        
        # exit(0)
        data.append(item)
        if topk is not None and len(data) == topk:
            break

    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), scenegraphs_json_path, elapsed_time))
    return data


def load_scene_graphs_selective(scenegraphs_json_path, vocab_json, embedding_json, images, topk=None):
    data = []
    start_time = time.time()
    print("Start to load scene graphs from %s" % scenegraphs_json_path)

    GQA_with_scene_graph = GQASceneDataset(scenegraphs_json_path, vocab_json, embedding_json)
    scenegraphs_json = GQA_with_scene_graph.scenegraphs_json

    for imageid in scenegraphs_json.keys():
        item = dict()
        embeds, boxes = GQA_with_scene_graph.scene2embedding(imageid) 
        item["img_id"] = imageid
        if item["img_id"] not in images :
            continue

        info_raw = scenegraphs_json[imageid]
        item["img_h"] = int(info_raw['height'])
        item["img_w"] = int(info_raw['width'])

        objects_id = []
        attrs_id = []
        for obj_id in info_raw['objects']:
            obj = info_raw['objects'][obj_id]
            name = obj['name']
            try:
                obj_cid = GQA_with_scene_graph.getname2idx(name)
            except:
                obj_cid = 0
            objects_id.append(int(obj_cid))
            for attr in obj['attributes'] :
                try:
                    attr_cid = GQA_with_scene_graph.getattr2idx(attr)
                except:
                    attr_cid = 0
                attrs_id.append(attr_cid)

        # objects_id is the one-hot encoder
        # need to make the object_id of different images to be the same size
        # Truncated and filled (Threshold = 36)
        if len(objects_id) < 36:
            for i in range(36 - len(objects_id)):
                objects_id.append(0)
        else:
            objects_id = objects_id[:36]    

        if len(attrs_id) < 36:
            for i in range(36 - len(attrs_id)):
                attrs_id.append(0)
        else:
            attrs_id = attrs_id[:36]


        item["objects_id"] = objects_id
        item["objects_conf"] = [1]*36
        item["attrs_id"] = attrs_id
        item["attrs_conf"] = [1]*36
        item["num_boxes"] = len(boxes)
        item["boxes"] = boxes
        item["features"] = embeds

        # num_boxes = item['num_boxes']
        # decode_config = [
        #     ('objects_id',  (num_boxes, ), np.int64),
        #     ('objects_conf', (num_boxes, ), np.float32),
        #     ('attrs_id', (num_boxes, ), np.int64),
        #     ('attrs_conf', (num_boxes, ), np.float32),
        #     ('boxes', (num_boxes, 4), np.float32),
        #     ('features', (num_boxes, -1), np.float32), # is the embeddings
        # ]
        num_boxes = item['num_boxes']
        decode_config = [
            ('objects_id', np.int64),
            ('objects_conf',  np.float32),
            ('attrs_id',  np.int64),
            ('attrs_conf',  np.float32),
            ('boxes', np.float32),
            ('features', np.float32), # is the embeddings
        ]

    

        for key, datatype in decode_config:
            # print("%s data type:" % key, type(item[key]))
            # print("%s shape:" % key, len(item[key]))
            # if not key == 'features':
            #     print("%s:" % key, item[key])
            # else:
            #     for i in range(len(item[key])):
            #         print("%s[%d]:" % (key, i), len(item[key][i]))
                
            item[key] = np.array(item[key], dtype = datatype)
            # item[key] = item[key].reshape(shape)
            item[key].setflags(write=False)
        
        # exit(0)
        data.append(item)
        if topk is not None and len(data) == topk:
            break

    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), scenegraphs_json_path, elapsed_time))
    return data
