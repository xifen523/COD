import pickle
import os
import numpy as np
import numba
from tqdm import tqdm
from prettytable import PrettyTable
from scipy.optimize import linear_sum_assignment
from pcdet.utils import box_utils, calibration_kitti

from pcdet.datasets.kitti.kitti_object_eval_python.rotate_iou import rotate_iou_gpu_eval


def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # ONLY support overlap in CAMERA, not lider.
    N, K = boxes.shape[0], qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                # iw = (min(boxes[i, 1] + boxes[i, 4], qboxes[j, 1] +
                #         qboxes[j, 4]) - max(boxes[i, 1], qboxes[j, 1]))
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


def image_box_overlap(boxes, query_boxes, criterion=-1):
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (
                            (boxes[n, 2] - boxes[n, 0]) *
                            (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih)
                    elif criterion == 0:
                        ua = ((boxes[n, 2] - boxes[n, 0]) *
                              (boxes[n, 3] - boxes[n, 1]))
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps

class CPMatrix(object):
    def __init__(self, gt_annos, dt_annos, img_pre=None, calib_path=None, iou_3d=0.5, iou_2d=0.8):
        self.gt_annos = self._read_data(gt_annos)  # 获取gt信息
        self.dt_annos = self._read_data(dt_annos)  # 获取dt信息
        if img_pre is not None:
            self.img_pre = self._read_data(img_pre)   # 获取img的dt信息
        else:
            self.img_pre = None

        self.iou_3d = iou_3d
        self.iou_2d = iou_2d

        self.gt_num = {"car": 0, "pedestrian": 0, "cyclist": 0}
        self.valid_num = {"car": 0, "pedestrian": 0, "cyclist": 0}

        self.calib_path = calib_path
        
    def _read_data(self, path):
        with open(path, 'rb') as f:
            data = pickle.load(f)
        return data
    
    def _write_data(self, path, data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    def _parse_bbox(self,info):
        car = []
        pedestrian = []
        cyclist = []
        for i, cls in enumerate(info["name"]):
            if cls.lower() == "car":
                bbox_3d = np.concatenate((info["location"][i], info["dimensions"][i], np.array([info["rotation_y"][i]]), info["bbox"][i]))
                car.append(bbox_3d)
            elif cls.lower() == "pedestrian" :
                bbox_3d = np.concatenate((info["location"][i], info["dimensions"][i], np.array([info["rotation_y"][i]]), info["bbox"][i]))
                pedestrian.append(bbox_3d)
            elif cls.lower() == "cyclist" :
                bbox_3d = np.concatenate((info["location"][i], info["dimensions"][i], np.array([info["rotation_y"][i]]), info["bbox"][i]))
                cyclist.append(bbox_3d)
            else:
                pass
            
        
        car_boxes = np.array(car)
        pedestrian_boxes = np.array(pedestrian)
        cyclist_boxes = np.array(cyclist)


        return car_boxes, pedestrian_boxes, cyclist_boxes
    
    def count_frame(self, gt_annos, dt_annos, cls):
        
        if len(gt_annos) > 0 and len(dt_annos) > 0:

            car_iou = d3_box_overlap(gt_annos[:,:7], dt_annos[:,:7]).astype(np.float64)
            # 将car_iou中小于 iou_3d 设置为0
            car_iou[car_iou < self.iou_3d] = 0
            cost_iou = 1 - car_iou

            # 应用匈牙利算法
            row_indices, col_indices = linear_sum_assignment(cost_iou)

            overlap_part = image_box_overlap(gt_annos[:,7:], dt_annos[:,7:])

            for i, j in zip(row_indices, col_indices):
                if overlap_part[i, j] > self.iou_2d:
                    self.valid_num[cls] += 1

    def get_matrix(self, img_dt = None):
        gt_annos = self.gt_annos
        dt_annos = self.dt_annos

        if img_dt is not None:
            dt_annos = img_dt


        assert len(gt_annos) == len(dt_annos), "gt_annos and dt_annos must have same length" 


        for i in tqdm(range(len(gt_annos))):

            car_gt_boxes, pedestrian_gt_boxes, cyclist_gt_boxes = self._parse_bbox(gt_annos[i]) 
            car_dt_boxes, pedestrian_dt_boxes, cyclist_dt_boxes = self._parse_bbox(dt_annos[i])

            self.gt_num["car"] += len(car_gt_boxes)
            self.gt_num["pedestrian"] += len(pedestrian_gt_boxes)
            self.gt_num["cyclist"] += len(cyclist_gt_boxes)

            self.count_frame(car_gt_boxes, car_dt_boxes, "car")
            self.count_frame(pedestrian_gt_boxes, pedestrian_dt_boxes, "pedestrian")
            self.count_frame(cyclist_gt_boxes, cyclist_dt_boxes, "cyclist")

            
        
        print(self.gt_num)

        cp = {"car": 0, "pedestrian": 0, "cyclist": 0}
        for cls in ["car", "pedestrian", "cyclist"]:
            cp[cls] = (self.valid_num[cls] / self.gt_num[cls]) * 100
        
                      
        return cp
    
    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_path, f'{idx}.txt')
        assert  os.path.exists(calib_file), "calib file not exist"
        return calibration_kitti.Calibration(calib_file)
    
    def get_calib_noise(self, idx):
        # noise_path ="/home/dell/workspace/motion/cd/data/kitti/training/calib_noise"
        noise_path ="/home/dell/workspace/motion/cd/data/V2X_I/training/calib_noise"
        calib_file = os.path.join(noise_path, f'{idx}.txt')
        assert  os.path.exists(calib_file), "calib file not exist"
        return calibration_kitti.Calibration(calib_file)
    
    def get_img_matrix(self):
        gt_annos = self.gt_annos
        dt_annos = self.dt_annos
        img_dt_annos = self.img_pre

        assert len(gt_annos) == len(dt_annos), "gt_annos and dt_annos must have same length"
        assert len(dt_annos) == len(img_dt_annos), "gt_annos and img_dt_annos must have same length"

        merge_result = []
        merge_result2 = []
        for i in tqdm(range(len(dt_annos))):
            frame_idx = dt_annos[i]["frame_id"]
            assert dt_annos[i]["frame_id"] == img_dt_annos[i]["frame_id"], "frame_id must be same"

            image_shape = img_dt_annos[i]['image_shape']

            loc = dt_annos[i]["location"] 
            dims = dt_annos[i]["dimensions"] 
            rots = dt_annos[i]["rotation_y"] 
            pred_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
            
            calib = self.get_calib(frame_idx)
            calib_noise = self.get_calib_noise(frame_idx)
            pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera, calib, image_shape=image_shape
            )
            
            pred_boxes = box_utils.boxes3d_kitti_camera_to_lidar(pred_boxes_camera, calib)
            pred_boxes_camera_noise = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib_noise)
            pred_boxes_img_noise = box_utils.boxes3d_kitti_camera_to_imageboxes(
                pred_boxes_camera_noise, calib_noise, image_shape=image_shape
            )


            img_pre_boxes = img_dt_annos[i]["bbox"]
            overlap_part = image_box_overlap(pred_boxes_img, img_pre_boxes)
            overlap_part[overlap_part<0.8]=0
            cost_iou = 1 - overlap_part
            row_indices, col_indices = linear_sum_assignment(cost_iou)

            overlap_part2 = image_box_overlap(pred_boxes_img_noise, img_pre_boxes)
            overlap_part2[overlap_part2<0.8]=0
            cost_iou2 = 1 - overlap_part2
            row_indices2, col_indices2 = linear_sum_assignment(cost_iou2)


            frame_info = {"frame_id": frame_idx}
            frame_info2 = {"frame_id": frame_idx}

            # 将dt_annos[i]中的key 更新到frame_info中
            for key in dt_annos[i].keys():
                if key in ["frame_id"]:
                    continue
                if key not in ["bbox"]:
                    frame_info[key] = dt_annos[i][key][row_indices]
                    frame_info2[key] = dt_annos[i][key][row_indices2]
                else:
                    frame_info[key] = img_dt_annos[i][key][col_indices]
                    frame_info2[key] = img_dt_annos[i][key][col_indices2]

            merge_result.append(frame_info)
            merge_result2.append(frame_info2)
        
        
        
        cp1 = self.get_matrix(merge_result)
        cp2 = self.get_matrix(merge_result2)

        return (cp1, cp2)


        
def print_result(cp):
    table = PrettyTable(['名称', 'car', 'pedestrian', 'cyclist'])
    if len(cp) == 2:
        table.add_row(["cal", cp[0]["car"], cp[0]["pedestrian"], cp[0]["cyclist"]])
        table.add_row(["noise", cp[1]["car"], cp[1]["pedestrian"], cp[1]["cyclist"]])
    elif len(cp) == 3 and not isinstance(cp,dict):
        table.add_row(["pre", cp[0]["car"], cp[0]["pedestrian"], cp[0]["cyclist"]])   
        table.add_row(["cal", cp[1]["car"], cp[1]["pedestrian"], cp[1]["cyclist"]])
        table.add_row(["noise", cp[2]["car"], cp[2]["pedestrian"], cp[2]["cyclist"]])
    else:
        table.add_row(["pre", cp["car"], cp["pedestrian"], cp["cyclist"]])

    print(table)
        

    
if __name__ == "__main__":
    # gt_annos = "/home/dell/workspace/motion/cd/data/kitti/kitti_gt_annos.pkl"
    # # dt_annos = "/home/dell/workspace/motion/cd/output/cfgs/kitti_models/pointpillar_img/default/eval/epoch_80/val/default/result_pred_noise_bbox.pkl"   # pre noise cp   cp_precision_1
    # # dt_annos = "/home/dell/workspace/motion/cd/output/cfgs/kitti_models/pointpillar_img/default/eval/epoch_80/val/default/result_pred_bbox.pkl"         # pre cp cp_precision_1
    # dt_annos = "/home/dell/workspace/motion/cd/output/cfgs/kitti_models/pointpillar/default/eval/epoch_80/val/default/result.pkl"                         # cal cp cp_precision_2
    # calib_path = "/home/dell/workspace/motion/cd/data/kitti/training/calib"
    # # calib_path_noise = "/home/dell/workspace/motion/cd/data/kitti/training/calib_noise"
    # img_dt_annos = "/home/dell/workspace/ultralytics-main/runs/detect/val6/labels/dt_bbox.pkl"
    # cp = CPMatrix(gt_annos, dt_annos, img_pre=img_dt_annos, calib_path=calib_path, iou_2d=0.5)
    # # cp_precision_1 = cp.get_matrix()
    # cp_precision_2 = cp.get_img_matrix()
    # # cp_precision = (cp_precision_1, cp_precision_2[0], cp_precision_2[1])
    # # print(cp_precision)
    # print_result(cp_precision_2)
    # # 同时计算cp_precision_1和cp_precision_2 会出bug  到时候再修复







    gt_annos = "/home/dell/workspace/motion/cd/data/V2X_I/v2x_i_gt_annos.pkl"
    # dt_annos = "/home/dell/workspace/motion/cd/output/cfgs/v2x_i_models/centerpoint_img_paper/default/eval/epoch_80/val/default/result_pred_bbox.pkl"
    dt_annos = "/home/dell/workspace/motion/cd/output/cfgs/v2x_i_models/centerpoint_img_paper/default/eval/epoch_80/val/default/result.pkl"              # pre noise cp   cp_precision_1
    # dt_annos = "/home/dell/workspace/motion/cd/output/cfgs/v2x_i_models/pillarnet/default/eval/epoch_80/val/default/result_cal_bbox.pkl"                         # cal cp cp_precision_2
    calib_path = "/home/dell/workspace/motion/cd/data/V2X_I/training/calib"
    img_dt_annos = "/home/dell/workspace/ultralytics-main/runs/detect/train422/labels/dt_bbox_v2x_i.pkl"
    cp = CPMatrix(gt_annos, dt_annos, img_pre=img_dt_annos, calib_path=calib_path, iou_2d=0.5)
    # cp_precision_2 = cp.get_img_matrix()
    # print_result(cp_precision_2)
    cp_precision_1 = cp.get_matrix()
    print_result(cp_precision_1)