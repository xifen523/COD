 
from __future__ import print_function
 
import os
import sys
import cv2
import os.path
from PIL import Image
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
from kitti_object import *
 
 
def visualization(gt_boxes_lidar):
    # import mayavi.mlab as mlab
    dataset = kitti_object(os.path.join(ROOT_DIR, '/mnt/disk/data/kitti'))   # linux 路径
    data_idx = 10               # 选择第几张图像
 
    # 1-加载标签数据
    objects = dataset.get_label_objects(data_idx)
    print("There are %d objects.", len(objects))
 
    # 2-加载图像
    img = dataset.get_image(data_idx)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_height, img_width, img_channel = img.shape
 
    # 3-加载点云数据
    pc_velo = dataset.get_lidar(data_idx)[:,0:3] # (x, y, z)
 
    # 4-加载标定参数
    calib = dataset.get_calibration(data_idx)
 
    # 5-可视化原始图像
    print(' ------------ show raw image -------- ')
    Image.fromarray(img).show()
    
    # 6-在图像中画2D框
    print(' ------------ show image with 2D bounding box -------- ')
    show_image_with_boxes(img, objects, calib, False)
 
    # 7-在图像中画3D框
    print(' ------------ show image with 3D bounding box ------- ')
    show_image_with_boxes(img, objects, calib, True)
    
    # 8-将点云数据投影到图像
    print(' ----------- LiDAR points projected to image plane -- ')
    show_lidar_on_image(pc_velo, img, calib, img_width, img_height)

    img_pos = calib.project_velo_to_image(gt_boxes_lidar[:,:3])
    print(img_pos)

 
    # 9-画BEV图
    print('------------------ BEV of LiDAR points -----------------------------')
    show_lidar_topview(pc_velo, objects, calib)
 
     # 10-在BEV图中画2D框
    print('--------------- BEV of LiDAR points with bobes ---------------------')
    img1 = cv2.imread('save_output/BEV.png')     
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    show_lidar_topview_with_boxes(img1, objects, calib)
    
    
if __name__=='__main__':
    

    gt_boxes_lidar = np.array( [[     5.4909,     -4.4136,    -0.92955,        3.35,        1.65,        1.57,     -0.1508],
                                [     12.089,      2.4069,    -0.86853,        3.95,         1.7,        1.43,     -3.3308],
                                [       23.8,     -8.3122,     -0.4844,        1.09,        0.72,        1.96,     -3.3208],
                                [     16.791,     -5.8323,    -0.84645,        3.24,         1.6,        1.51,     -0.1308],
                                [      22.34,     -6.8517,    -0.80923,         4.1,        1.74,        1.45,     -0.1808],
                                [      23.93,     0.39954,    -0.81101,        3.79,        1.68,        1.54,     -3.3508],
                                [      29.36,    -0.61993,    -0.77003,        3.35,        1.52,        1.49,     -3.3608],
                                [     28.821,     -7.8595,    -0.84216,        4.37,        1.65,        1.53,     -0.1708],
                                [      43.14,     -4.4774,    -0.65178,        3.48,        1.45,        1.64,     -3.5908]])
    
    visualization(gt_boxes_lidar)


    trans_lidar_to_cam = np.array([[ 2.3477e-04,  1.0449e-02,  9.9995e-01,  0.0000e+00],
                                    [-9.9994e-01,  1.0565e-02,  1.2437e-04,  0.0000e+00],
                                    [-1.0563e-02, -9.9989e-01,  1.0451e-02,  0.0000e+00],
                                    [-2.7968e-03, -7.5109e-02, -2.7213e-01,  1.0000e+00]]).T
    trans_cam_to_img = np.array([[7.2154e+02, 0.0000e+00, 0.0000e+00],
                                [0.0000e+00, 7.2154e+02, 0.0000e+00],
                                [6.0956e+02, 1.7285e+02, 1.0000e+00],
                                [4.4857e+01, 2.1638e-01, 2.7459e-03]]).T