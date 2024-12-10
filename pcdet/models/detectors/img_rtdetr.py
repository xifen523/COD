import torch
import numpy as np

from .detector3d_template import Detector3DTemplate
from ultralytics.engine.results import Results
from ultralytics.utils import ops


class ImgRTdetr(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)

        self.nc = model_cfg["IMG_BACKBONE_2D"]["NUM_CLASS"]
        self.criterion = self.init_criterion()

        self.module_topology = [
            'vfe', 'backbone_3d', 'map_to_bev_module', 'pfe',
            'backbone_2d', 'dense_head',  'point_head', 'roi_head', 'img_backbone_2d'
        ]
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list[:-1]: # 最后一层detr的图像检测先不计算
            batch_dict = cur_module(batch_dict)

        batch_dict = self.module_list[-1](batch_dict) # detr 的前向传播

        if self.training:
            loss_lidar, tb_dict_lidar, disp_dict_lidar = self.get_training_loss()
            loss_img, tb_dict_img, disp_dict_img = self.get_training_loss_img(batch_dict)

            ret_dict = {
                'loss': loss_lidar+loss_img
            }

            tb_dict = {
                **tb_dict_lidar,
                **tb_dict_img
            }

            disp_dict = {
                **disp_dict_lidar,
                **disp_dict_img
            }
            
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            results = self.detr_post_processing(batch_dict)

            combined_pred_dicts = [{**d1, **d2} for d1, d2 in zip(pred_dicts, results)]
            return combined_pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}

        loss_rpn, tb_dict = self.dense_head.get_loss()
        tb_dict = {
            'loss_rpn': loss_rpn.item(),
            **tb_dict
        }

        loss = loss_rpn
        return loss, tb_dict, disp_dict
    
    def get_training_loss_img(self,data_dict):
        disp_dict = {}

        preds = data_dict["preds_img"]
        targets = data_dict["targets"]
        dec_bboxes, dec_scores, enc_bboxes, enc_scores, dn_meta = preds if self.training else preds[1]
        if dn_meta is None:
            dn_bboxes, dn_scores = None, None
        else:
            dn_bboxes, dec_bboxes = torch.split(dec_bboxes, dn_meta['dn_num_split'], dim=2)
            dn_scores, dec_scores = torch.split(dec_scores, dn_meta['dn_num_split'], dim=2)

        dec_bboxes = torch.cat([enc_bboxes.unsqueeze(0), dec_bboxes])  # (7, bs, 300, 4)
        dec_scores = torch.cat([enc_scores.unsqueeze(0), dec_scores])

        
        loss = self.criterion((dec_bboxes, dec_scores),
                                targets,
                                dn_bboxes=dn_bboxes,
                                dn_scores=dn_scores,
                                dn_meta=dn_meta)
        
        tb_dict = {
            'loss_giou': loss["loss_giou"].item(),
            'loss_class': loss["loss_class"].item(),
            'loss_bbox': loss["loss_bbox"].item()
        }

        return sum(loss.values()), tb_dict, disp_dict
    
    def init_criterion(self):
        """Initialize the loss criterion for the RTDETRDetectionModel."""
        from ultralytics.models.utils.loss import RTDETRDetectionLoss

        return RTDETRDetectionLoss(nc=self.nc, use_vfl=True)


    def post_processing(self, batch_dict):
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
    
    def detr_post_processing(self, batch_dict):
        """
        Postprocess the raw predictions from the model to generate bounding boxes and confidence scores.

        The method filters detections based on confidence and class if specified in `self.args`.

        Args:
            preds (torch.Tensor): Raw predictions from the model.
            img (torch.Tensor): Processed input images.
            orig_imgs (list or torch.Tensor): Original, unprocessed images.

        Returns:
            (list[Results]): A list of Results objects containing the post-processed bounding boxes, confidence scores,
                and class labels.
        """
        conf = self.model_cfg.POST_PROCESSING.CONF
        imgsz = self.model_cfg.POST_PROCESSING.SIZE
        y, x = batch_dict['preds_img']
        bs, query_num, nd = y.shape

        max_init_num = query_num - 300   # 300 个初始的query用于图片检测   max_init_num表示电云初始化的query数量

        results = []
        for idx in range(bs):
            bboxes, scores = y[idx].split((4, nd - 4), dim=-1)
            valid_num = len(batch_dict['final_box_dicts'][idx]['pred_labels']) #真正有效点云初始的个数
            orig_imgs_size = batch_dict['ori_shape'] 
            pred_boxes = []
            pred_scores = []
            pred_labels = []

            pred_boxes_3dinit = []
            pred_scores_3dinit = []
            pred_labels_3dinit = []
            
            for i, bbox in enumerate(bboxes):  # (300, 4)
                score, cls = scores[i].max(-1, keepdim=True)  # (300, 1)
                if i < max_init_num: # 保存点云初始化的query
                    if i < valid_num: # 保存点云有效query
                        bbox = self.yolo2annos(bbox, orig_imgs_size[idx], imgsz)
                        pred_boxes_3dinit.append(bbox)
                        pred_scores_3dinit.append(score)
                        pred_labels_3dinit.append(cls)
                elif score.squeeze(-1) > conf:  # 保存高置信度的query
                    bbox = self.yolo2annos(bbox, orig_imgs_size[idx], imgsz)
                    pred_boxes.append(bbox)
                    pred_scores.append(score)
                    pred_labels.append(cls)
                else:
                    continue

            final_2dpred_dict = {
                "2d_pred_boxes": torch.stack(pred_boxes) if len(pred_boxes) > 0 else torch.empty(0),
                "2d_pred_scores": torch.stack(pred_scores) if len(pred_scores) > 0 else torch.empty(0),
                "2d_pred_labels": torch.stack(pred_labels) if len(pred_labels) > 0 else torch.empty(0),
                "2d_pred_boxes_3dinit": torch.stack(pred_boxes_3dinit) if len(pred_boxes_3dinit) > 0 else torch.empty(0),
                "2d_pred_scores_3dinit": torch.stack(pred_scores_3dinit) if len(pred_scores_3dinit) > 0 else torch.empty(0),
                "2d_pred_labels_3dinit": torch.stack(pred_labels_3dinit) if len(pred_labels_3dinit) > 0 else torch.empty(0),
            }
            results.append(final_2dpred_dict)
        return results
    
    
    def yolo2annos(self, x, raw_shape, imgsz):
        """
        功能：将yolo的标注转为原始图像的标注
        """

        assert x.shape[-1] == 4, f"input shape last dimension expected 4 but input shape is {x.shape}"
        y = torch.empty_like(x) if isinstance(x, torch.Tensor) else np.empty_like(x)  # faster than clone/copy
        dw = x[..., 2] / 2  # half-width
        dh = x[..., 3] / 2  # half-height
        y[..., 0] = x[..., 0] - dw  # top left x
        y[..., 1] = x[..., 1] - dh  # top left y
        y[..., 2] = x[..., 0] + dw  # bottom right x
        y[..., 3] = x[..., 1] + dh  # bottom right y

        oh, ow = raw_shape  # 原始图像的高度和宽度
        width_scale = ow / imgsz
        height_scale = oh / imgsz
        y *= imgsz

        y[..., [0,2]] *= width_scale
        y[..., [1,3]] *= height_scale 

        return y