from __future__ import print_function, division
from typing import Optional

import os
import time
import math
import GPUtil
import numpy as np
from yacs.config import CfgNode

import torch
from connectomics.model import *
from affseg.dataset import get_CV_dataset
from connectomics.data.augmentation import TestAugmentor
from connectomics.data.dataset import build_dataloader
from connectomics.data.utils import build_blending_matrix, writeh5
from connectomics.data.utils import get_padsize, array_unpad
from .mira_utils import cv_io
from synaptor import io, chunk_bboxes



class Predictor(object):
    r"""Trainer class for supervised learning.

    Args:
        cfg (yacs.config.CfgNode): YACS configuration options.
        device (torch.device): model running device. GPUs are recommended for model training and inference.
        mode (str): running mode of the trainer (``'train'`` or ``'test'``). Default: ``'train'``
        rank (int, optional): node rank for distributed training. Default: `None`
        checkpoint (str, optional): the checkpoint file to be loaded. Default: `None`
    """

    def __init__(self,
                 cfg: CfgNode,
                 device: torch.device,
                 mode: str = 'train',
                 rank: Optional[int] = None,
                 checkpoint: Optional[str] = None):

        assert mode in [ 'test']
        self.cfg = cfg
        self.device = device
        self.output_dir = cfg.DATASET.OUTPUT_PATH
        self.mode = mode
        self.rank = rank
        self.is_main_process = rank is None or rank == 0
        self.inference_singly = (mode == 'test') and cfg.INFERENCE.DO_SINGLY

        self.model = build_model(self.cfg, self.device, rank)

        self.update_checkpoint(checkpoint)
        # build test-time augmentor and update output filename
        self.augmentor = TestAugmentor.build_from_cfg(cfg, activation=True)
        self.dataset, self.dataloader = None, None


    def update_checkpoint(self, checkpoint: Optional[str] = None):
        r"""Update the model with the specified checkpoint file path.
        """
        if checkpoint is None:
            return

        # load pre-trained model
        print('Load pretrained checkpoint: ', checkpoint)
        checkpoint = torch.load(checkpoint)
        print('checkpoints: ', checkpoint.keys())

        # update model weights
        if 'state_dict' in checkpoint.keys():
            pretrained_dict = checkpoint['state_dict']
            pretrained_dict = update_state_dict(
                self.cfg, pretrained_dict, mode=self.mode)
            model_dict = self.model.module.state_dict()  # nn.DataParallel
            # 1. filter out unnecessary keys by name
            pretrained_dict = {k: v for k,
                               v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict (if size match)
            for param_tensor in pretrained_dict:
                if model_dict[param_tensor].size() == pretrained_dict[param_tensor].size():
                    model_dict[param_tensor] = pretrained_dict[param_tensor]
            # 3. load the new state dict
            self.model.module.load_state_dict(model_dict)  # nn.DataParallel

        if self.mode == 'train' and not self.cfg.SOLVER.ITERATION_RESTART:
            if hasattr(self, 'optimizer') and 'optimizer' in checkpoint.keys():
                self.optimizer.load_state_dict(checkpoint['optimizer'])

            if hasattr(self, 'lr_scheduler') and 'lr_scheduler' in checkpoint.keys():
                self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

            if hasattr(self, 'start_iter') and 'iteration' in checkpoint.keys():
                self.start_iter = checkpoint['iteration']


    def affine_test(self):
        r"""Inference function of the trainer class.
        """
        self.model.eval() if self.cfg.INFERENCE.DO_EVAL else self.model.train()
        output_scale = self.cfg.INFERENCE.OUTPUT_SCALE
        spatial_size = list(np.ceil(
            np.array(self.cfg.MODEL.OUTPUT_SIZE) *
            np.array(output_scale)).astype(int))
        channel_size = self.cfg.MODEL.OUT_PLANES
        if self.cfg.MODEL.ARCHITECTURE == 'twoway_net':
            channel_size = channel_size * 2

        sz = tuple([channel_size] + spatial_size)
        ww = build_blending_matrix(spatial_size, self.cfg.INFERENCE.BLENDING)

        output_size = [tuple(np.ceil(np.array(x) * np.array(output_scale)).astype(int))
                       for x in self.dataloader._dataset.volume_size]
        result = [np.stack([np.zeros(x, dtype=np.float32)
                            for _ in range(channel_size)]) for x in output_size]
        weight = [np.zeros(x, dtype=np.float32) for x in output_size]
        print("Total number of batches: ", len(self.dataloader))

        start = time.perf_counter()
        with torch.no_grad():
            for i, sample in enumerate(self.dataloader):
                print('progress: %d/%d batches, total time %.2fs' %
                      (i+1, len(self.dataloader), time.perf_counter()-start))

                pos, volume = sample.pos, sample.out_input
                volume = volume.to(self.device, non_blocking=True)
                output = self.augmentor(self.model, volume)
                if torch.cuda.is_available() and i % 50 == 0:
                    GPUtil.showUtilization(all=True)

                for idx in range(output.shape[0]):
                    st = pos[idx]
                    st = (np.array(st) *
                          np.array([1]+output_scale)).astype(int).tolist()
                    out_block = output[idx]
                    if result[st[0]].ndim - out_block.ndim == 1:  # 2d model
                        out_block = out_block[:, np.newaxis, :]

                    result[st[0]][:, st[1]:st[1]+sz[1], st[2]:st[2]+sz[2],
                                  st[3]:st[3]+sz[3]] += out_block * ww[np.newaxis, :]
                    weight[st[0]][st[1]:st[1]+sz[1], st[2]:st[2]+sz[2],
                                  st[3]:st[3]+sz[3]] += ww

        end = time.perf_counter()
        print("Prediction time: %.2fs" % (end-start))

        for vol_id in range(len(result)):
            if result[vol_id].ndim > weight[vol_id].ndim:
                weight[vol_id] = np.expand_dims(weight[vol_id], axis=0)
            result[vol_id] /= weight[vol_id]  # in-place to save memory
            result[vol_id] *= 255

            ##### affine invert transform #########
            if self.cfg.INFERENCE.affine_reverse:
                result[vol_id] = self.dataset.maybe_affine(result[vol_id], order=1, invert=True)
            ##### affine invert transform #########

            result[vol_id] = result[vol_id].astype(np.uint8)

            if self.cfg.INFERENCE.UNPAD:

                if self.dataset is None:
                    pad_size = (np.array(self.cfg.DATASET.PAD_SIZE) *
                                np.array(output_scale)).astype(int).tolist()
                else:
                    pad_size = (np.array(self.dataset.pad_size) * np.array(output_scale)).astype(int).tolist()
                if self.cfg.DATASET.DO_CHUNK_TITLE != 0 or self.cfg.DATASET.DO_STACK_TITLE != 0:
                    # In chunk-based inference using TileDataset, padding is applied
                    # before resizing, while in normal inference using VolumeDataset,
                    # padding is after resizing. Thus we adjust pad_size accordingly.
                    pad_size = (np.array(self.cfg.DATASET.DATA_SCALE) *
                                np.array(pad_size)).astype(int).tolist()
                pad_size = get_padsize(pad_size)
                result[vol_id] = array_unpad(result[vol_id], pad_size)

        if self.output_dir is None:
            return result
        else:
            print('Final prediction shapes are:')
            for k in range(len(result)):
                print(result[k].shape)
            if self.cfg.INFERENCE.CV_SAVE_NAME is not None:
                ### write cloud volume
                self.cvio.writecv(self.dataset.coord, result[0])
                print('Prediction saved as CV')




    def save_zeros(self, volume_size):
        output_scale = self.cfg.INFERENCE.OUTPUT_SCALE
        spatial_size = list(np.ceil(
            np.array(self.cfg.MODEL.OUTPUT_SIZE) *
            np.array(output_scale)).astype(int))
        channel_size = self.cfg.MODEL.OUT_PLANES
        if self.cfg.MODEL.ARCHITECTURE == 'twoway_net':
            channel_size = channel_size * 2
        output_size = [tuple(np.ceil(np.array(x) * np.array(output_scale)).astype(int))
                       for x in volume_size]
        result = [np.stack([np.zeros(x, dtype=np.uint8)
                            for _ in range(channel_size)]) for x in output_size]
        if self.cfg.INFERENCE.CV_SAVE_NAME is not None:
            ### write cloud volume
            self.cvio.writecv(self.dataset.coord, result[0])
        else:
            writeh5(os.path.join(self.output_dir, self.test_filename), result,
                    ['vol%d' % (x) for x in range(len(result))])
            print('Prediction saved as: ', self.test_filename)



    def run_CV_PC(self, mode: str):
        self.dataset = get_CV_dataset(self.cfg, self.augmentor, mode)
        chunk_begin = self.cfg.DATASET.VOLUME_START[::-1]
        chunk_end = self.cfg.DATASET.VOLUME_END[::-1]
        bboxes = chunk_bboxes(np.array(chunk_end)-np.array(chunk_begin), self.cfg.DATASET.DATA_CHUNK_SIZE[::-1], offset=chunk_begin, mip=self.cfg.DATASET.MIP)
        # inference mode
        vol_size = self.cfg.DATASET.VOLUME_SIZE[::-1]
        mip_factor = 2 ** self.cfg.DATASET.MIP
        vol_size = (vol_size[0] // mip_factor,
                    vol_size[1] // mip_factor,
                    vol_size[2])
        self.cvio = cv_io.CVIO(self.cfg.INFERENCE.CV_SAVE_NAME, self.cfg.INFERENCE.CV_RESOLUTION,
                               vol_size, self.cfg.INFERENCE.TARGET, chunk_size=self.cfg.DATASET.CV_CHUNK_SIZE[::-1], mode='save', )
        import random
        random.shuffle(bboxes)
        for box in bboxes:
            self.run_CV_box(box)
        print('all boxes are done')
        return

    def run_CV_box(self, bbox):

        x1, y1, z1 = bbox.min()
        x2, y2, z2 = bbox.max()
        coord = [z1, z2, y1, y2, x1, x2]
        if self.cvio.exists(coord):
            return
        flag = self.dataset.loadchunk(np.array(coord))
        if not flag:

            return
        self.dataloader = build_dataloader(self.cfg, self.augmentor, 'test',
                                           dataset=self.dataset.dataset)
        self.dataloader = iter(self.dataloader)

        self.affine_test()






