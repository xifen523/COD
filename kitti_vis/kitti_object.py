 
 
from __future__ import print_function
 
import os
import sys
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'mayavi'))
import kitti_util as utils
 
 
'''
在图像中画2D框、3D框
'''
def show_image_with_boxes(img, objects, calib, show3d=True):
    img1 = np.copy(img) # for 2d bbox
    img2 = np.copy(img) # for 3d bbox

    index = 0
    for obj in objects:
        index = index + 1
        if obj.type=='DontCare':continue
        cv2.rectangle(img1, (int(obj.xmin),int(obj.ymin)), (int(obj.xmax),int(obj.ymax)), (0,255,0), 2) # 画2D框
        # # 绘制标签ID
        label = f'img_ID: {index}'
        cv2.putText(img1, label, (int(obj.xmin), int(obj.ymin)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) # 获取图像3D框(8*2)、相机坐标系3D框(8*3)
        img2 = utils.draw_projected_box3d(img2, box3d_pts_2d, id=index) # 在图像上画3D框
    if show3d:
        Image.fromarray(img2).save('save_output/image_with_3Dboxes.png')
        Image.fromarray(img2).show()
    else:
        Image.fromarray(img1).save('save_output/image_with_2Dboxes.png')
        Image.fromarray(img1).show()
 
 
'''
可视化BEV鸟瞰图
'''
def show_lidar_topview(pc_velo, objects, calib):
      # 1-设置鸟瞰图范围
    side_range = (-30, 30)  # 左右距离
    fwd_range = (0, 80)  # 后前距离
    
    x_points = pc_velo[:, 0]
    y_points = pc_velo[:, 1]
    z_points = pc_velo[:, 2]
    
    # 2-获得区域内的点
    f_filt = np.logical_and(x_points > fwd_range[0], x_points < fwd_range[1])
    s_filt = np.logical_and(y_points > side_range[0], y_points < side_range[1])
    filter = np.logical_and(f_filt, s_filt)
    indices = np.argwhere(filter).flatten() 
    x_points = x_points[indices]
    y_points = y_points[indices]
    z_points = z_points[indices]
    
    # 定义了鸟瞰图中每个像素代表的距离
    res = 0.1   
    # 3-1将点云坐标系 转到 BEV坐标系
    x_img = (-y_points / res).astype(np.int32)
    y_img = (-x_points / res).astype(np.int32)
    # 3-2调整坐标原点
    x_img -= int(np.floor(side_range[0]) / res)
    y_img += int(np.floor(fwd_range[1]) / res)
    print(x_img.min(), x_img.max(), y_img.min(), y_img.max()) 
    
    # 4-填充像素值, 将点云数据的高度信息（Z坐标）映射到像素值
    height_range = (-3, 1.0)
    pixel_value = np.clip(a=z_points, a_max=height_range[1], a_min=height_range[0])
     
 
    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)
    
    pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    
    # 创建图像数组
    x_max = 1 + int((side_range[1] - side_range[0]) / res)
    y_max = 1 + int((fwd_range[1] - fwd_range[0]) / res)
    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    
    im2 = Image.fromarray(im)
    im2.save('save_output/BEV.png')
    im2.show()
 
 
'''
将点云数据3D框投影到BEV
'''
def show_lidar_topview_with_boxes(img, objects, calib):
    def bbox3d(obj):
        box3d_pts_2d, box3d_pts_3d = utils.compute_box_3d(obj, calib.P) # 获取3D框-图像、3D框-相机坐标系
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d) # 将相机坐标系的框 转到 激光雷达坐标系
        return box3d_pts_3d_velo # 返回nx3的点
 
    boxes3d = [bbox3d(obj) for obj in objects if obj.type == "Car"]
    gt = np.array(boxes3d)
    im2 = utils.draw_box3d_label_on_bev(img, gt, scores=None, thickness=1) # 获取激光雷达坐标系的3D点，选择x, y两维，画到BEV平面坐标系上
    im2 = Image.fromarray(im2)
    im2.save('save_output/BEV with boxes.png')
    im2.show()
 
 
'''
将点云数据投影到图像
'''
def show_lidar_on_image(pc_velo, img, calib, img_width, img_height):
    ''' Project LiDAR points to image '''
    imgfov_pc_velo, pts_2d, fov_inds = get_lidar_in_image_fov(pc_velo,
        calib, 0, 0, img_width, img_height, True)
    imgfov_pts_2d = pts_2d[fov_inds,:]
    imgfov_pc_rect = calib.project_velo_to_rect(imgfov_pc_velo)
 
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:,:3]*255
 
    for i in range(imgfov_pts_2d.shape[0]):
        depth = imgfov_pc_rect[i,2]
        color = cmap[int(640.0/depth),:]
        cv2.circle(img, (int(np.round(imgfov_pts_2d[i,0])),
            int(np.round(imgfov_pts_2d[i,1]))),
            2, color=tuple(color), thickness=-1)
    Image.fromarray(img).save('save_output/lidar_on_image.png')
    Image.fromarray(img).show() 
    return img
 
 
'''
将点云数据投影到相机坐标系
'''
def get_lidar_in_image_fov(pc_velo, calib, xmin, ymin, xmax, ymax,
                           return_more=False, clip_distance=2.0):
    ''' Filter lidar points, keep those in image FOV '''
    pts_2d = calib.project_velo_to_image(pc_velo)
    fov_inds = (pts_2d[:,0]<xmax) & (pts_2d[:,0]>=xmin) & \
        (pts_2d[:,1]<ymax) & (pts_2d[:,1]>=ymin)
    fov_inds = fov_inds & (pc_velo[:,0]>clip_distance)
    imgfov_pc_velo = pc_velo[fov_inds,:]
    if return_more:
        return imgfov_pc_velo, pts_2d, fov_inds
    else:
        return imgfov_pc_velo
    
 
'''
解析标签
'''
class kitti_object(object):
    '''Load and parse object data into a usable format.'''
    
    def __init__(self, root_dir, split='training'):
        '''root_dir contains training and testing folders'''
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(root_dir, split)
 
        if split == 'training':
            self.num_samples = 7481
        elif split == 'testing':
            self.num_samples = 7518
        else:
            print('Unknown split: %s' % (split))
            exit(-1)
 
        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.lidar_dir = os.path.join(self.split_dir, 'velodyne')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
 
    def __len__(self):
        return self.num_samples
 
    def get_image(self, idx):
        assert(idx<self.num_samples) 
        img_filename = os.path.join(self.image_dir, '%06d.png'%(idx))
        return utils.load_image(img_filename)
 
    def get_lidar(self, idx): 
        assert(idx<self.num_samples) 
        lidar_filename = os.path.join(self.lidar_dir, '%06d.bin'%(idx))
        return utils.load_velo_scan(lidar_filename)
 
    def get_calibration(self, idx):
        assert(idx<self.num_samples) 
        calib_filename = os.path.join(self.calib_dir, '%06d.txt'%(idx))
        return utils.Calibration(calib_filename)
 
    def get_label_objects(self, idx):
        assert(idx<self.num_samples and self.split=='training') 
        label_filename = os.path.join(self.label_dir, '%06d.txt'%(idx))
        return utils.read_label(label_filename)
    
    def get_depth_map(self, idx):
        pass
 
    def get_top_down(self, idx):
        pass