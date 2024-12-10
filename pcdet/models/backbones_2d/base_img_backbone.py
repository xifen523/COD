import numpy as np
import torch
import torch.nn as nn


import contextlib
import ast
from . import detr

from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    ResNetLayer, RTDETRDecoder, Segment)
from ultralytics.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights, intersect_dicts,
                                           make_divisible, model_info, scale_img, time_sync)


class BaseIMGBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        if self.model_cfg.get('LAYER_NUMS', None) is not None:
            assert len(self.model_cfg.LAYER_NUMS) == len(self.model_cfg.LAYER_STRIDES) == len(self.model_cfg.NUM_FILTERS)
            layer_nums = self.model_cfg.LAYER_NUMS
            layer_strides = self.model_cfg.LAYER_STRIDES
            num_filters = self.model_cfg.NUM_FILTERS
        else:
            layer_nums = layer_strides = num_filters = []

        if self.model_cfg.get('UPSAMPLE_STRIDES', None) is not None:
            assert len(self.model_cfg.UPSAMPLE_STRIDES) == len(self.model_cfg.NUM_UPSAMPLE_FILTERS)
            num_upsample_filters = self.model_cfg.NUM_UPSAMPLE_FILTERS
            upsample_strides = self.model_cfg.UPSAMPLE_STRIDES
        else:
            upsample_strides = num_upsample_filters = []

        num_levels = len(layer_nums)
        c_in_list = [input_channels, *num_filters[:-1]]
        self.blocks = nn.ModuleList()
        self.deblocks = nn.ModuleList()
        for idx in range(num_levels):
            cur_layers = [
                nn.ZeroPad2d(1),
                nn.Conv2d(
                    c_in_list[idx], num_filters[idx], kernel_size=3,
                    stride=layer_strides[idx], padding=0, bias=False
                ),
                nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                nn.ReLU()
            ]
            for k in range(layer_nums[idx]):
                cur_layers.extend([
                    nn.Conv2d(num_filters[idx], num_filters[idx], kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(num_filters[idx], eps=1e-3, momentum=0.01),
                    nn.ReLU()
                ])
            self.blocks.append(nn.Sequential(*cur_layers))
            if len(upsample_strides) > 0:
                stride = upsample_strides[idx]
                if stride > 1 or (stride == 1 and not self.model_cfg.get('USE_CONV_FOR_NO_STRIDE', False)):
                    self.deblocks.append(nn.Sequential(
                        nn.ConvTranspose2d(
                            num_filters[idx], num_upsample_filters[idx],
                            upsample_strides[idx],
                            stride=upsample_strides[idx], bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))
                else:
                    stride = np.round(1 / stride).astype(np.int)
                    self.deblocks.append(nn.Sequential(
                        nn.Conv2d(
                            num_filters[idx], num_upsample_filters[idx],
                            stride,
                            stride=stride, bias=False
                        ),
                        nn.BatchNorm2d(num_upsample_filters[idx], eps=1e-3, momentum=0.01),
                        nn.ReLU()
                    ))

        c_in = sum(num_upsample_filters)
        if len(upsample_strides) > num_levels:
            self.deblocks.append(nn.Sequential(
                nn.ConvTranspose2d(c_in, c_in, upsample_strides[-1], stride=upsample_strides[-1], bias=False),
                nn.BatchNorm2d(c_in, eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ))

        self.num_bev_features = c_in

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        ups = []
        ret_dict = {}
        x = spatial_features
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)

            stride = int(spatial_features.shape[2] / x.shape[2])
            ret_dict['spatial_features_%dx' % stride] = x
            if len(self.deblocks) > 0:
                ups.append(self.deblocks[i](x))
            else:
                ups.append(x)

        if len(ups) > 1:
            x = torch.cat(ups, dim=1)
        elif len(ups) == 1:
            x = ups[0]

        if len(self.deblocks) > len(self.blocks):
            x = self.deblocks[-1](x)

        data_dict['spatial_features_2d'] = x

        return data_dict


class BaseIMGBackboneDETR(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        ch = [3]   # input_channels(3)
        nc = self.model_cfg.get('NUM_CLASS', 3)
        max_channels = 1024
        if self.model_cfg.get('NET', None) is not None:
            net = self.model_cfg.get('NET', None)
            self.blocks = nn.ModuleList()
            self.is_save_result(net)
            for index_net, net_block in enumerate(net):
                net_info = net[net_block]
                n = net_info.get("NUM_LAYER", 1)                # num_layer
                f = net_info.get("INPUT", -1)                   # input_layer
                args = net_info.get("ARGS")                    # block_args
                m = net_info.get("NAME")                        # block_name
                m = getattr(torch.nn, m[3:]) if 'nn.' in m else detr.__all__[m]  # get module

                for j, a in enumerate(args):
                    if isinstance(a, str):
                        with contextlib.suppress(ValueError):
                            args[j] = locals()[a] if a in locals() else ast.literal_eval(a)
                

                if m in (detr.__all__['HGStem'], detr.__all__['HGBlock']):
                    c1, cm, c2 = ch[f], args[0], args[1]
                    args = [c1, cm, c2, *args[2:]]
                    if m is detr.__all__['HGBlock']:
                        args.insert(4, n)  # number of repeats
                        n = 1
                
                elif m in (detr.__all__['DWConv'], detr.__all__['Conv'], detr.__all__['RepC3']):
                    c1, c2 = ch[f], args[0]
                    if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                        c2 = make_divisible(min(c2, max_channels), 8)
                    args = [c1, c2, *args[1:]]
                    if m in (detr.__all__['RepC3'], detr.__all__['C1']):
                        args.insert(2, n)  # number of repeats
                        n = 1

                elif m is detr.__all__['AIFI']:
                    args = [ch[f], *args]

                elif m is detr.__all__['Concat']:
                    c2 = sum(ch[x] for x in f)

                elif m is detr.__all__['RTDETRDecoder']:  # special case, channels arg must be passed in index 1
                    args.insert(1, [ch[x] for x in f])

                else:
                    c2 = ch[f]

                m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
                m_.i, m_.f = index_net, f
                self.blocks.append(m_)
                if index_net == 0:
                    ch = []
                ch.append(c2)

        self.num_bev_features = ch[-1]
        self.backbone_features = ch[-1]

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                img
        Returns:
        """
        x = data_dict['images']

        bs = len(data_dict['images'])
        num_bboxes = data_dict['gt_boxes2d'].shape[1]
        batch_idx = np.arange(bs).repeat(num_bboxes)
        batch_idx = torch.from_numpy(batch_idx).to(x.device, dtype=torch.long)
        bboxes_mask = ~(data_dict['gt_boxes2d'].sum(2) == 0)
        batch_idx = batch_idx[bboxes_mask.view(-1)]

        bboxes = data_dict['gt_boxes2d'][bboxes_mask]
        cls = data_dict['gt_boxes'][:,:,-1]
        cls = cls[bboxes_mask]

        gt_groups = [(batch_idx == i).sum().item() for i in range(bs)]
        targets = {
            'cls': cls.to(x.device, dtype=torch.long).view(-1),
            'bboxes': bboxes,
            'batch_idx': batch_idx.view(-1),
            'gt_groups': gt_groups,
            'final_box_dicts': data_dict["final_box_dicts"],
            'gt_boxes': data_dict['gt_boxes'],
            'trans_lidar_to_cam': data_dict['trans_lidar_to_cam'],
            'trans_cam_to_img': data_dict['trans_cam_to_img'],
            'gt_boxes2d': data_dict['gt_boxes2d'],
            'gt_boxes2d_raw': data_dict['gt_boxes2d_raw'],
            'calib': data_dict['calib'],
            'ori_shape': data_dict["ori_shape"],
            'noise_calib': data_dict.get('noise_calib', None),
            } 
        
        # pred_boxes = data_dict["final_box_dicts"][0]["pred_boxes"]
        # new_column = torch.ones((pred_boxes.shape[0], 1), device=pred_boxes.device)
        # pred_pos = torch.cat((pred_boxes[:,:3], new_column), dim=1)
        # img_hmo = (pred_pos @  data_dict["trans_lidar_to_cam"][0].T) @ data_dict["trans_cam_to_img"][0].T
        # img_hmo[:,[0,1]] = img_hmo[:,[0,1]] / img_hmo[:,[2]]
        data_dict["targets"] = targets


        y = []
        for i in range(len(self.blocks)-1):
            m = self.blocks[i]
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            x = m(x)
            y.append(x if m.i in self.save else None)  # save output

        data_dict["backbone_features_img"] = x
        data_dict['backbone_features_img_list'] = y

        head = self.blocks[-1]   # last block  RTDETRDecoder
        x = [y[j] for j in head.f]
        preds = head(x,targets)  

        data_dict["preds_img"] = preds

        return data_dict
    
    def is_save_result(self, net):
        self.save = []
        for i, block_name in enumerate(net):
            val = net[block_name]
            f = val.get("INPUT", -1)
            self.save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        sorted(self.save)
