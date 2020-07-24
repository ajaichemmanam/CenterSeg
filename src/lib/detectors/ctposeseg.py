from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import numpy as np
from progress.bar import Bar
import time
import torch
from pycocotools import mask as mask_utils
try:
    from external.nms import soft_nms, soft_nms_39
except:
    print('NMS not imported! If you need it,'
          ' do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import ctposeseg_decode
from models.utils import flip_tensor, flip_lr_off, flip_lr
from utils.image import get_affine_transform
from utils.post_process import ctposeseg_post_process
from utils.debugger import Debugger

from .base_detector import BaseDetector


class CtPoseSegDetector(BaseDetector):
    def __init__(self, opt):
        super(CtPoseSegDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            output = self.model(images)[-1]
            output['hm'] = output['hm'].sigmoid_()
            if self.opt.hm_hp and not self.opt.mse_loss:
                output['hm_hp'] = output['hm_hp'].sigmoid_()

            seg_feat = output['seg_feat']
            conv_weight = output['conv_weight']
            reg = output['reg'] if self.opt.reg_offset else None
            hm_hp = output['hm_hp'] if self.opt.hm_hp else None
            hp_offset = output['hp_offset'] if self.opt.reg_hp_offset else None
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time = time.time()
            assert not self.opt.flip_test, "not support flip_test"
            dets, masks = ctposeseg_decode(
                output['hm'], output['wh'],  output['hps'], seg_feat, conv_weight, reg=reg, cat_spec_wh=self.opt.cat_spec_wh, hm_hp=hm_hp, hp_offset=hp_offset, K=self.opt.K)

        if return_time:
            return output, (dets, masks), forward_time
        else:
            return output, (dets, masks)

    def post_process(self, det_seg, meta, scale=1):
        assert scale == 1, "not support scale != 1"
        dets, seg = det_seg
        dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
        seg = seg.detach().cpu().numpy()
        dets = ctposeseg_post_process(
            dets.copy(), seg.copy(), [meta['c']], [meta['s']],
            meta['out_height'], meta['out_width'], *meta['img_size'], self.opt.num_classes)
        return dets[0]

    def merge_outputs(self, detections):
        return detections[0]

    def debug(self, debugger, images, dets, output, scale=1):
        dets = dets.detach().cpu().numpy().copy()
        dets[:, :, :4] *= self.opt.down_ratio
        dets[:, :, 5:39] *= self.opt.down_ratio
        img = images[0].detach().cpu().numpy().transpose(1, 2, 0)
        img = np.clip(((
            img * self.std + self.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
        debugger.add_blend_img(img, pred, 'pred_hm')
        if self.opt.hm_hp:
            pred = debugger.gen_colormap_hp(
                output['hm_hp'][0].detach().cpu().numpy())
            debugger.add_blend_img(img, pred, 'pred_hmhp')

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctposeseg')
        for j in range(1, self.num_classes + 1):
            for i in range(len(results[j]['boxs'])):
                bbox = results[j]['boxs'][i]
                mask = mask_utils.decode(results[j]['pred_mask'][i])
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(
                        bbox[:4], j - 1, bbox[4], img_id='ctposeseg')
                    debugger.add_coco_seg(mask, img_id='ctposeseg')
                    debugger.add_coco_hp(bbox[5:39], img_id='ctposeseg')
        debugger.show_all_imgs(pause=self.pause)
