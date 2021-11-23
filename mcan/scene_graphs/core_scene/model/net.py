# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core_scene.model.net_utils import FC, MLP, LayerNorm
from core_scene.model.mca import MCA_ED
from core_scene.data.data_utils import proc_img_feat

import torch.nn as nn
import torch.nn.functional as F
import torch
import pdb


# ------------------------------
# ---- Flatten the sequence ----
# ------------------------------

class AttFlat(nn.Module):
    def __init__(self, __C):
        super(AttFlat, self).__init__()
        self.__C = __C

        self.mlp = MLP(
            in_size=__C.HIDDEN_SIZE,
            mid_size=__C.FLAT_MLP_SIZE,
            out_size=__C.FLAT_GLIMPSES,
            dropout_r=__C.DROPOUT_R,
            use_relu=True
        )

        self.linear_merge = nn.Linear(
            __C.HIDDEN_SIZE * __C.FLAT_GLIMPSES,
            __C.FLAT_OUT_SIZE
        )

    def forward(self, x, x_mask):
        ## x -> [64, 14, 512]
        att = self.mlp(x)
        ## att -> [64, 14, 1]
        att = att.masked_fill(
            x_mask.squeeze(1).squeeze(1).unsqueeze(2),
            -1e9
        )
        att = F.softmax(att, dim=1)
        ## att -> [64, 14, 1]

        att_list = []
        for i in range(self.__C.FLAT_GLIMPSES):
            att_list.append(
                torch.sum(att[:, :, i: i + 1] * x, dim=1)
            )

        ## att_list -> [64, 512]
        x_atted = torch.cat(att_list, dim=1)
        ## x_atted -> [64, 512]
        x_atted = self.linear_merge(x_atted)
        ## x_atted -> [64, 1024]

        return x_atted


# -------------------------
# ---- Main MCAN Model ----
# -------------------------

class Net(nn.Module):
    def __init__(self, __C, pretrained_emb, token_size, answer_size):
        super(Net, self).__init__()
        self.embedding = nn.Embedding(
            num_embeddings=token_size,
            embedding_dim=__C.WORD_EMBED_SIZE
        )

        self.IMG_FEAT_PAD_SIZE = __C.IMG_FEAT_PAD_SIZE

        self.bbox_encodings = nn.Linear(4, __C.IMG_FEAT_SIZE)
        # Loading the GloVe embedding weights
        if __C.USE_GLOVE:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_emb))

        self.lstm = nn.LSTM(
            input_size=__C.WORD_EMBED_SIZE,
            hidden_size=__C.HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.img_feat_linear = nn.Linear(
            __C.IMG_FEAT_SIZE,
            __C.HIDDEN_SIZE
        )

        self.backbone = MCA_ED(__C)

        self.attflat_img = AttFlat(__C)
        self.attflat_lang = AttFlat(__C)

        self.proj_norm = LayerNorm(__C.FLAT_OUT_SIZE)
        self.proj = nn.Linear(__C.FLAT_OUT_SIZE, answer_size)


    def forward(self, img_feat_obj, img_feat_attr, ques_ix, img_feat_bbox):

        ## encode box coordinates and add them to img features
        bbox_output = self.bbox_encodings(img_feat_bbox)
        #img_feat = torch.cat([bbox_output,img_feat],dim=1)

        #pdb.set_trace()
        #num_samples = bbox_output.size()[0]
        #out_size = img_feat.size()[1] + bbox_output.size()[1]
        #out_dim = img_feat.size()[2]

        #concat_output = torch.zeros(num_samples,out_size,out_dim)
        #for j in range(100) :
        #    index = j*3 + 2
        #    img_feat_index = j*2 + 2
        #    concat_output[:,index-2:index] = img_feat[:,img_feat_index-2:img_feat_index]
        #    concat_output[:,index] = bbox_output[:,j]

        ''' Taking average of obj , attr and box embeddings '''
        img_feat = img_feat_obj + img_feat_attr + bbox_output
        img_feat = img_feat/3
        #img_feat = proc_img_feat(img_feat,self.IMG_FEAT_PAD_SIZE)
        # Make mask - Masking for 0's
        ## ques_ix -> [64, 14]  img_feat -> [64, 100, 2048]
        lang_feat_mask = self.make_mask(ques_ix.unsqueeze(2))
        img_feat_mask = self.make_mask(img_feat)
        ## lang_feat_mask -> [64, 1, 1, 14]  img_feat_mask -> [64, 1, 1, 100]

        # Pre-process Language Feature - converting ques to glove embedding (300) and then using lstm to 512 size
        lang_feat = self.embedding(ques_ix)
        lang_feat, _ = self.lstm(lang_feat)

        # Pre-process Image Feature - convertes 2048 to 512

        img_feat = self.img_feat_linear(img_feat)

        # Backbone Framework
        #pdb.set_trace()
        lang_feat, img_feat = self.backbone(
            lang_feat,
            img_feat,
            lang_feat_mask,
            img_feat_mask
        )
        #pdb.set_trace()
        ## lang_feat -> [64, 14, 512]  
        lang_feat = self.attflat_lang(
            lang_feat,
            lang_feat_mask
        )
        ## lang_feat -> [64, 1024]  

        ## img_feat -> [64, 100, 512]  
        img_feat = self.attflat_img(
            img_feat,
            img_feat_mask
        )
        ## img_feat -> [64, 1024]  

        proj_feat = lang_feat + img_feat
        proj_feat = self.proj_norm(proj_feat)
        proj_feat = torch.sigmoid(self.proj(proj_feat))
        ## proj_feat -> [64, 3129]

        return proj_feat


    # Masking
    def make_mask(self, feature):
        return (torch.sum(
            torch.abs(feature),
            dim=-1
        ) == 0).unsqueeze(1).unsqueeze(2)
