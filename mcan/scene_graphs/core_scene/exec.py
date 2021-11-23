# --------------------------------------------------------
# mcan-vqa (Deep Modular Co-Attention Networks)
# Licensed under The MIT License [see LICENSE for details]
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------

from core_scene.data.load_data import DataSet
from core_scene.model.net import Net
from core_scene.model.optim import get_optim, adjust_lr
from core_scene.data.data_utils import shuffle_list

import os, json, torch, datetime, pickle, copy, shutil, time
import numpy as np
import torch.nn as nn
import torch.utils.data as Data
import pdb
from collections import OrderedDict

class Execution:
    def __init__(self, __C):
        self.__C = __C

        print('Loading training set ........')
        self.dataset = DataSet(__C)

        self.dataset_eval = None
        if __C.EVAL_EVERY_EPOCH:
            __C_eval = copy.deepcopy(__C)
            setattr(__C_eval, 'RUN_MODE', 'val')

            print('Loading validation set for per-epoch evaluation ........')
            self.dataset_eval = DataSet(__C_eval)


    def train(self, dataset, dataset_eval=None):

        # Obtain needed information
        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        # Define the MCAN model
        net = Net(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.train()

        # Define the multi-gpu training if needed
        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        # Define the binary cross entropy loss
        # loss_fn = torch.nn.BCELoss(size_average=False).cuda()
        loss_fn = torch.nn.BCELoss(reduction='sum').cuda()

        # Load checkpoint if resume training
        if self.__C.RESUME:
            print(' ========== Resume training')

            if self.__C.CKPT_PATH is not None:
                print('Warning: you are now using CKPT_PATH args, '
                      'CKPT_VERSION and CKPT_EPOCH will not work')

                path = self.__C.CKPT_PATH
            else:
                path = self.__C.CKPTS_PATH + \
                       'ckpt_' + self.__C.CKPT_VERSION + \
                       '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

            # Load the network parameters
            print('Loading ckpt {}'.format(path))
            ckpt = torch.load(path)
            print('Finish!')
            net.load_state_dict(ckpt['state_dict'])

            # Load the optimizer paramters
            optim = get_optim(self.__C, net, data_size, ckpt['lr_base'])
            optim._step = int(data_size / self.__C.BATCH_SIZE * self.__C.CKPT_EPOCH)
            optim.optimizer.load_state_dict(ckpt['optimizer'])

            start_epoch = self.__C.CKPT_EPOCH

        else:
            if ('ckpt_' + self.__C.VERSION) in os.listdir(self.__C.CKPTS_PATH):
                shutil.rmtree(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            os.mkdir(self.__C.CKPTS_PATH + 'ckpt_' + self.__C.VERSION)

            optim = get_optim(self.__C, net, data_size)
            start_epoch = 0

        loss_sum = 0
        named_params = list(net.named_parameters())
        grad_norm = np.zeros(len(named_params))

        # Define multi-thread dataloader
        if self.__C.SHUFFLE_MODE in ['external']:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=False,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )
        else:
            dataloader = Data.DataLoader(
                dataset,
                batch_size=self.__C.BATCH_SIZE,
                shuffle=True,
                num_workers=self.__C.NUM_WORKERS,
                pin_memory=self.__C.PIN_MEM,
                drop_last=True
            )

        # Training script
        for epoch in range(start_epoch, self.__C.MAX_EPOCH):

            # Save log information
            logfile = open(
                self.__C.LOG_PATH +
                'log_run_' + self.__C.VERSION + '.txt',
                'a+'
            )
            #if epoch == 1 :
            #    logfile.write(self.__C)
            #    logfile.write('\n')

            logfile.write(
                'nowTime: ' +
                datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') +
                '\n'
            )
            logfile.close()

            # Learning Rate Decay
            if epoch in self.__C.LR_DECAY_LIST:
                adjust_lr(optim, self.__C.LR_DECAY_R)

            # Externally shuffle
            ''' need to undo this '''
            #if self.__C.SHUFFLE_MODE == 'external':
            #    shuffle_list(dataset.ans_list)

            time_start = time.time()
            # Iteration
            #### Modification required in dataloader
            for step, (
                    img_feat_obj_iter,
                    img_feat_attr_iter,
                    ques_ix_iter,
                    ans_iter,
                    img_feat_bbox
            ) in enumerate(dataloader):
            #img_feat_iter, ques_ix_iter, ans_iter, img_feat_bbox = dataset.loading(1)

            #for k in range(1) :
                optim.zero_grad()

                img_feat_obj_iter = torch.tensor(img_feat_obj_iter, dtype=torch.float32)
                img_feat_attr_iter = torch.tensor(img_feat_attr_iter, dtype=torch.float32)
                img_feat_attr_iter = img_feat_attr_iter.cuda()
                img_feat_obj_iter = img_feat_obj_iter.cuda()
                ques_ix_iter = ques_ix_iter.cuda()
                img_feat_bbox = img_feat_bbox.cuda()
                ans_iter = ans_iter.cuda()
                for accu_step in range(self.__C.GRAD_ACCU_STEPS):

                    sub_img_feat_obj_iter = \
                        img_feat_obj_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_img_feat_attr_iter = \
                        img_feat_attr_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                      (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ques_ix_iter = \
                        ques_ix_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                     (accu_step + 1) * self.__C.SUB_BATCH_SIZE]
                    sub_ans_iter = \
                        ans_iter[accu_step * self.__C.SUB_BATCH_SIZE:
                                 (accu_step + 1) * self.__C.SUB_BATCH_SIZE]

                    
                    #pdb.set_trace()
                    #net = net.float()
                    pred = net(
                        sub_img_feat_obj_iter,
                        sub_img_feat_attr_iter,
                        sub_ques_ix_iter,
                        img_feat_bbox
                    )
                    #pdb.set_trace()

                    loss = loss_fn(pred, sub_ans_iter)
                    # only mean-reduction needs be divided by grad_accu_steps
                    # removing this line wouldn't change our results because the speciality of Adam optimizer,
                    # but would be necessary if you use SGD optimizer.
                    # loss /= self.__C.GRAD_ACCU_STEPS
                    loss.backward()
                    loss_sum += loss.cpu().data.numpy() * self.__C.GRAD_ACCU_STEPS

                    if self.__C.VERBOSE:
                        if dataset_eval is not None:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['val']
                        else:
                            mode_str = self.__C.SPLIT['train'] + '->' + self.__C.SPLIT['test']

                        print("\r[version %s][epoch %2d][step %4d/%4d][%s] loss: %.4f, lr: %.2e" % (
                            self.__C.VERSION,
                            epoch + 1,
                            step,
                            int(data_size / self.__C.BATCH_SIZE),
                            mode_str,
                            loss.cpu().data.numpy() / self.__C.SUB_BATCH_SIZE,
                            optim._rate
                        ), end='          ')

                # Gradient norm clipping
                if self.__C.GRAD_NORM_CLIP > 0:
                    nn.utils.clip_grad_norm_(
                        net.parameters(),
                        self.__C.GRAD_NORM_CLIP
                    )

                # Save the gradient information
                for name in range(len(named_params)):
                    norm_v = torch.norm(named_params[name][1].grad).cpu().data.numpy() \
                        if named_params[name][1].grad is not None else 0
                    grad_norm[name] += norm_v * self.__C.GRAD_ACCU_STEPS
                    # print('Param %-3s Name %-80s Grad_Norm %-20s'%
                    #       (str(grad_wt),
                    #        params[grad_wt][0],
                    #        str(norm_v)))

                optim.step()

            time_end = time.time()
            print('Finished in {}s'.format(int(time_end-time_start)))

            # print('')
            epoch_finish = epoch + 1

            # Save checkpoint

            if (epoch % 5 == 0) :

                state = {
                    'state_dict': net.state_dict(),
                    'optimizer': optim.optimizer.state_dict(),
                    'lr_base': optim.lr_base
                }
                torch.save(
                    state,
                    self.__C.CKPTS_PATH +
                    'ckpt_' + self.__C.VERSION +
                    '/epoch' + str(epoch_finish) +
                    '.pkl'
                )

                # Logging
                logfile = open(
                    self.__C.LOG_PATH +
                    'log_run_' + self.__C.VERSION + '.txt',
                    'a+'
                )
                logfile.write(
                    'epoch = ' + str(epoch_finish) +
                    '  loss = ' + str(loss_sum / data_size) +
                    '\n' +
                    'lr = ' + str(optim._rate) +
                    '\n\n'
                )
                logfile.close()

            # Eval after every epoch
            if dataset_eval is not None:
                self.eval(
                    dataset_eval,
                    state_dict=net.state_dict(),
                    valid=True
                )

            # if self.__C.VERBOSE:
            #     logfile = open(
            #         self.__C.LOG_PATH +
            #         'log_run_' + self.__C.VERSION + '.txt',
            #         'a+'
            #     )
            #     for name in range(len(named_params)):
            #         logfile.write(
            #             'Param %-3s Name %-80s Grad_Norm %-25s\n' % (
            #                 str(name),
            #                 named_params[name][0],
            #                 str(grad_norm[name] / data_size * self.__C.BATCH_SIZE)
            #             )
            #         )
            #     logfile.write('\n')
            #     logfile.close()

            loss_sum = 0
            grad_norm = np.zeros(len(named_params))


    # Evaluation
    def eval(self, dataset, state_dict=None, valid=False):

        # Load parameters
        if self.__C.CKPT_PATH is not None:
            print('Warning: you are now using CKPT_PATH args, '
                  'CKPT_VERSION and CKPT_EPOCH will not work')

            path = self.__C.CKPT_PATH
        else:
            path = self.__C.CKPTS_PATH + \
                   'ckpt_' + self.__C.CKPT_VERSION + \
                   '/epoch' + str(self.__C.CKPT_EPOCH) + '.pkl'

        val_ckpt_flag = False
        if state_dict is None:
            val_ckpt_flag = True
            print('Loading ckpt {}'.format(path))
            state_dict = torch.load(path)['state_dict']
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]
                new_state_dict[name] = v
            state_dict = new_state_dict
            print('Finish!')

        # Store the prediction list
        #qid_list = [ques['question_id'] for ques in dataset.ques_list]
        ans_ix_list = []
        ans_index_list = []
        pred_list = []

        data_size = dataset.data_size
        token_size = dataset.token_size
        ans_size = dataset.ans_size
        pretrained_emb = dataset.pretrained_emb

        #pdb.set_trace()
        net = Net(
            self.__C,
            pretrained_emb,
            token_size,
            ans_size
        )
        net.cuda()
        net.eval()

        if self.__C.N_GPU > 1:
            net = nn.DataParallel(net, device_ids=self.__C.DEVICES)

        #pdb.set_trace()
        net.load_state_dict(state_dict)
        #net.load_state_dict({k.replace('module.',''):v for k,v in torch.load(path).items()})

        ## This needs to be changed
        #dataloader = Data.DataLoader(
        #    dataset,
        #    batch_size=self.__C.EVAL_BATCH_SIZE,
        #    #batch_size=1,
        #    shuffle=False,
        #    num_workers=self.__C.NUM_WORKERS,
        #    pin_memory=True
        #)

        #ques_list = dataset.ques_list
        #scene_embeddings = dataset.iid_to_img_feat

       # for step, (
       #         img_feat_iter,
       #         ques_ix_iter,
       #         ans_iter,
       #         img_feat_bbox,
       # ) in enumerate(dataloader):
       #     print("\rEvaluation: [step %4d/%4d]" % (
       #         step,
       #         int(data_size / self.__C.EVAL_BATCH_SIZE),
       #     ), end='          ')

        num_images = len(dataset.list_images)
        output_map = {}
        result = {}
        print ('num_images : ', num_images)
        for k in range(num_images) :
            print (str(k) + ' / ' + str(num_images))

            img_feat_iter_mapping, ques_dict, ans_dict, img_feat_bbox_mapping, changes_mapping, relevant_questions_mapping, img = dataset.extract(k)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            result[img] = {}
            for obj in img_feat_iter_mapping :
                #pdb.set_trace()
                result[img][obj] = {}
                result[img][obj]['changes'] = changes_mapping[obj]

                img_feat_iters = img_feat_iter_mapping[obj]
                c = 0
                ques_dict_obj = {}
                ques_dict_obj = { relevant_key: ques_dict[relevant_key]  for relevant_key in relevant_questions_mapping[obj]}
                num_ques = len(ques_dict_obj)
                result[img][obj]['relevant_questions'] = relevant_questions_mapping[obj]
                
                if num_ques == 0:
                    continue

                incorrect_questions = {}
                ans_list = {}
                for ques in ques_dict_obj :
                    ans_list[ques] = np.argmax(ans_dict[ques])

                result[img][obj]['answers'] = ans_list
                result[img][obj]["pred"] = {}

                ques_ix_iter = np.zeros((num_ques,14))
                question_ids = list(ques_dict_obj.keys())
                for j in range(num_ques) :
                    ques_ix_iter[j] = ques_dict_obj[question_ids[j]]

                #print (changes_mapping[obj])
                for img_feat_iter in img_feat_iters :

                    ques_ix_iter = torch.tensor(ques_ix_iter).to(device).long()
                    img_feat_iter = torch.tensor(img_feat_iter, dtype=torch.float32)
                    img_feat_iter = torch.unsqueeze(img_feat_iter,0)
                    img_feat_iter = torch.cat(num_ques * [img_feat_iter])
                    img_feat_iter = img_feat_iter.cuda()
                    ## img_feat_iter : torch.Size([32, 100, 2048])
                    ques_ix_iter = ques_ix_iter.cuda()

                    ## ques_ix_iter : torch.Size([32, 14]) tensor([179,   3, 707,   7,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0])
                    img_feat_bbox = torch.from_numpy(img_feat_bbox_mapping[obj])
                    img_feat_bbox = torch.unsqueeze(img_feat_bbox,0)
                    img_feat_bbox = torch.cat(num_ques * [img_feat_bbox])
                    img_feat_bbox = img_feat_bbox.cuda()

                    n_batches = int(np.ceil(num_ques/32))

                    result[img][obj]["pred"][c] = []
                    for i in range(n_batches) :
                        img_feat_iter_batch = img_feat_iter[32*i:32*(i+1)]
                        ques_ix_iter_batch = ques_ix_iter[32*i:32*(i+1)]
                        img_feat_bbox_batch = img_feat_bbox[32*i:32*(i+1)]
                        pred = net(
                            img_feat_iter_batch,
                            ques_ix_iter_batch,
                            img_feat_bbox_batch
                        )
                        #print ('pred : ', pred)
                        pred_np = pred.cpu().data.numpy()
                        pred_argmax = np.argmax(pred_np, axis=1)

                        result[img][obj]["pred"][c].append(pred_argmax)

                    c +=1

            #except :
            #    pass
        with open(self.__C.OUTPUT_JSON, 'w+') as f :
            json.dump(result, f, indent=4, sort_keys=True, separators=(', ', ': '), ensure_ascii=False, cls=NumpyEncoder)


    def run(self, run_mode):
        if run_mode == 'train':
            self.empty_log(self.__C.VERSION)
            #self.train(self.dataset, self.dataset_eval)
            self.train(self.dataset)

        elif run_mode == 'val':
            #self.eval(self.dataset, valid=True)
            self.eval(self.dataset, valid=False)

        elif run_mode == 'test':
            self.eval(self.dataset)

        else:
            exit(-1)


    def empty_log(self, version):
        print('Initializing log file ........')
        if (os.path.exists(self.__C.LOG_PATH + 'log_run_' + version + '.txt')):
            os.remove(self.__C.LOG_PATH + 'log_run_' + version + '.txt')
        print('Finished!')
        print('')




