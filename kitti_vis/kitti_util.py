from __future__ import print_function
 
import numpy as np
import cv2
from PIL import Image
import os
 
# 设置BEV鸟瞰图参数
side_range = (-30, 30)  # 左右距离
fwd_range = (0, 80)  # 后前距离
res = 0.1  # 分辨率0.05m
 
def compute_box_3d(obj, P):
    '''
    计算对象的3D边界框在图像平面上的投影
    输入: obj代表一个物体标签信息,  P代表相机的投影矩阵-内参。
    输出: 返回两个值, corners_3d表示3D边界框在 相机坐标系 的8个角点的坐标-3D坐标。
                                     corners_2d表示3D边界框在 图像上 的8个角点的坐标-2D坐标。
    '''
    # 计算一个绕Y轴旋转的旋转矩阵R，用于将3D坐标从世界坐标系转换到相机坐标系。obj.ry是对象的偏航角
    R = roty(obj.ry)    
 
    # 物体实际的长、宽、高
    l = obj.l;
    w = obj.w;
    h = obj.h;
    
    # 存储了3D边界框的8个角点相对于对象中心的坐标。这些坐标定义了3D边界框的形状。
    x_corners = [l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2];
    y_corners = [0,0,0,0,-h,-h,-h,-h];
    z_corners = [w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2];
    
    # 1、将3D边界框的角点坐标从对象坐标系转换到相机坐标系。它使用了旋转矩阵R
    corners_3d = np.dot(R, np.vstack([x_corners,y_corners,z_corners]))
    # 3D边界框的坐标进行平移
    corners_3d[0,:] = corners_3d[0,:] + obj.t[0];
    corners_3d[1,:] = corners_3d[1,:] + obj.t[1];
    corners_3d[2,:] = corners_3d[2,:] + obj.t[2];
 
    # 2、检查对象是否在相机前方，因为只有在相机前方的对象才会被绘制。
    # 如果对象的Z坐标（深度）小于0.1，就意味着对象在相机后方，那么corners_2d将被设置为None，函数将返回None。
    if np.any(corners_3d[2,:]<0.1):
        corners_2d = None
        return corners_2d, np.transpose(corners_3d)
    
    # 3、将相机坐标系下的3D边界框的角点，投影到图像平面上，得到它们在图像上的2D坐标。
    corners_2d = project_to_image(np.transpose(corners_3d), P);
    return corners_2d, np.transpose(corners_3d)
 
 
def project_to_image(pts_3d, P):
    '''
    将相机坐标系下的3D边界框的角点, 投影到图像平面上, 得到它们在图像上的2D坐标
    输入: pts_3d是一个nx3的矩阵, 包含了待投影的3D坐标点(每行一个点), P是相机的投影矩阵, 通常是一个3x4的矩阵。
    输出: 返回一个nx2的矩阵, 包含了投影到图像平面上的2D坐标点。
      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)  => normalize projected_pts_2d(2xn)
      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)   => normalize projected_pts_2d(nx2)
    '''
    n = pts_3d.shape[0] # 获取3D点的数量
    pts_3d_extend = np.hstack((pts_3d, np.ones((n,1)))) # 将每个3D点的坐标扩展为齐次坐标形式（4D），通过在每个点的末尾添加1，创建了一个nx4的矩阵。
    pts_2d = np.dot(pts_3d_extend, np.transpose(P)) # 将扩展的3D坐标点矩阵与投影矩阵P相乘，得到一个nx3的矩阵，其中每一行包含了3D点在图像平面上的投影坐标。每个点的坐标表示为[x, y, z]。
    pts_2d[:,0] /= pts_2d[:,2] # 将投影坐标中的x坐标除以z坐标，从而获得2D图像上的x坐标。
    pts_2d[:,1] /= pts_2d[:,2] # 将投影坐标中的y坐标除以z坐标，从而获得2D图像上的y坐标。
    return pts_2d[:,0:2] # 返回一个nx2的矩阵,其中包含了每个3D点在2D图像上的坐标。
 
 
 
def draw_projected_box3d(image, qs, color=(0,60,255), thickness=2, id=None):
    '''
    qs: 包含8个3D边界框角点坐标的数组, 形状为(8, 2)。图像坐标下的3D框, 8个顶点坐标。
    '''
    ''' Draw 3d bounding box in image
        qs: (8,2) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    '''
    qs = qs.astype(np.int32) # 将输入的顶点坐标转换为整数类型，以便在图像上绘制。
 
    # 这个循环迭代4次，每次处理一个边界框的一条边。
    for k in range(0,4):
       # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
 
       # 定义了要绘制的边的起始点和结束点的索引。在这个循环中，它用于绘制边界框的前四条边。
       i,j=k,(k+1)%4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
 
        # 定义了要绘制的边的起始点和结束点的索引。在这个循环中，它用于绘制边界框的后四条边，与前四条边平行
       i,j=k+4,(k+1)%4 + 4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)
 
        # 定义了要绘制的边的起始点和结束点的索引。在这个循环中，它用于绘制连接前四条边和后四条边的边界框的边。
       i,j=k,k+4
       cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness)

    label = f'3d_ID: {id}'
    cv2.putText(image, label, (int(qs[0,0]), int(qs[0,1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
    return image

def draw_box3d_label_on_bev(image, boxes3d, thickness=1, scores=None):
    # if scores is not None and scores.shape[0] >0:
    img = image.copy() 
    num = len(boxes3d)
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        if (x0<30 and x1<30 and x2<30 and x3<30):
            u0, v0 = lidar_to_top_coords(x0, y0)
            u1, v1 = lidar_to_top_coords(x1, y1)
            u2, v2 = lidar_to_top_coords(x2, y2)
            u3, v3 = lidar_to_top_coords(x3, y3)
            color = (0, 255, 0) # green
            cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
            cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)
        elif (x0<50 and x1<50 and x2<50 and x3<50):
            color = (255, 0, 0) # red
            u0, v0 = lidar_to_top_coords(x0, y0)
            u1, v1 = lidar_to_top_coords(x1, y1)
            u2, v2 = lidar_to_top_coords(x2, y2)
            u3, v3 = lidar_to_top_coords(x3, y3)
            cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
            cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)
        else:
            color = (0, 0, 255) # blue
            u0, v0 = lidar_to_top_coords(x0, y0)
            u1, v1 = lidar_to_top_coords(x1, y1)
            u2, v2 = lidar_to_top_coords(x2, y2)
            u3, v3 = lidar_to_top_coords(x3, y3)
            cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
            cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
            cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
            cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)       
 
    return img
 
def draw_box3d_predict_on_bev(image, boxes3d, thickness=1, scores=None):
     # if scores is not None and scores.shape[0] >0:
    img = image.copy() 
    num = len(boxes3d)
    for n in range(num):
        b = boxes3d[n]
        x0 = b[0, 0]
        y0 = b[0, 1]
        x1 = b[1, 0]
        y1 = b[1, 1]
        x2 = b[2, 0]
        y2 = b[2, 1]
        x3 = b[3, 0]
        y3 = b[3, 1]
        color = (255, 255, 255) # white
        u0, v0 = lidar_to_top_coords(x0, y0)
        u1, v1 = lidar_to_top_coords(x1, y1)
        u2, v2 = lidar_to_top_coords(x2, y2)
        u3, v3 = lidar_to_top_coords(x3, y3)
        cv2.line(img, (u0, v0), (u1, v1), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u1, v1), (u2, v2), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u2, v2), (u3, v3), color, thickness, cv2.LINE_AA)
        cv2.line(img, (u3, v3), (u0, v0), color, thickness, cv2.LINE_AA)
    return img
 
def lidar_to_top_coords(x, y, z=None):
    if 0:
        return x, y
    else:
        # print("TOP_X_MAX-TOP_X_MIN:",TOP_X_MAX,TOP_X_MIN)
        xx = (-y / res).astype(np.int32)
        yy = (-x / res).astype(np.int32)
        # 调整坐标原点
        xx -= int(np.floor(side_range[0]) / res)
        yy += int(np.floor(fwd_range[1]) / res)
        return xx, yy
 
 
# 解析标签数据
class Object3d(object):
    ''' 3d object label '''
    def __init__(self, label_file_line):
        data = label_file_line.split(' ')
        data[1:] = [float(x) for x in data[1:]]
 
        # extract label, truncation, occlusion
        self.type = data[0] # 'Car', 'Pedestrian', ...
        self.truncation = data[1] # truncated pixel ratio [0..1]
        self.occlusion = int(data[2]) # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3] # object observation angle [-pi..pi]
 
        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4] # left
        self.ymin = data[5] # top
        self.xmax = data[6] # right
        self.ymax = data[7] # bottom
        self.box2d = np.array([self.xmin,self.ymin,self.xmax,self.ymax])
        
        # extract 3d bounding box information
        self.h = data[8] # box height
        self.w = data[9] # box width
        self.l = data[10] # box length (in meters)
        self.t = (data[11],data[12],data[13]) # location (x,y,z) in camera coord.
        self.ry = data[14] # yaw angle (around Y-axis in camera coordinates) [-pi..pi]
 
    def print_object(self):
        print('Type, truncation, occlusion, alpha: %s, %d, %d, %f' % \
            (self.type, self.truncation, self.occlusion, self.alpha))
        print('2d bbox (x0,y0,x1,y1): %f, %f, %f, %f' % \
            (self.xmin, self.ymin, self.xmax, self.ymax))
        print('3d bbox h,w,l: %f, %f, %f' % \
            (self.h, self.w, self.l))
        print('3d bbox location, ry: (%f, %f, %f), %f' % \
            (self.t[0],self.t[1],self.t[2],self.ry))
 
 
class Calibration(object):
    ''' Calibration matrices and utils
        3d XYZ in <label>.txt are in rect camera coord.
        2d box xy are in image2 coord
        Points in <lidar>.bin are in Velodyne coord.
        y_image2 = P^2_rect * x_rect
        y_image2 = P^2_rect * R0_rect * Tr_velo_to_cam * x_velo
        x_ref = Tr_velo_to_cam * x_velo
        x_rect = R0_rect * x_ref
        P^2_rect = [f^2_u,  0,      c^2_u,  -f^2_u b^2_x;
                    0,      f^2_v,  c^2_v,  -f^2_v b^2_y;
                    0,      0,      1,      0]
                 = K * [1|t]
        image2 coord:
         ----> x-axis (u)
        |
        |
        v y-axis (v)
        velodyne coord:
        front x, left y, up z
        rect/ref camera coord:
        right x, down y, front z
        Ref (KITTI paper): http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
        TODO(rqi): do matrix multiplication only once for each projection.
    '''
    def __init__(self, calib_filepath, from_video=False):
        if from_video:
            calibs = self.read_calib_from_video(calib_filepath)
        else:
            calibs = self.read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord
        self.P = calibs['P2'] 
        self.P = np.reshape(self.P, [3,4])
        # Rigid transform from Velodyne coord to reference camera coord
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3,4])
        self.C2V = inverse_rigid_trans(self.V2C)
        # Rotation from reference camera coord to rect camera coord
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0,[3,3])
 
        # Camera intrinsics and extrinsics
        self.c_u = self.P[0,2]
        self.c_v = self.P[1,2]
        self.f_u = self.P[0,0]
        self.f_v = self.P[1,1]
        self.b_x = self.P[0,3]/(-self.f_u) # relative 
        self.b_y = self.P[1,3]/(-self.f_v)
 
    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.'''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line)==0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
 
        return data
    
    def read_calib_from_video(self, calib_root_dir):
        ''' Read calibration for camera 2 from video calib files.
            there are calib_cam_to_cam and calib_velo_to_cam under the calib_root_dir
        '''
        data = {}
        cam2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_cam_to_cam.txt'))
        velo2cam = self.read_calib_file(os.path.join(calib_root_dir, 'calib_velo_to_cam.txt'))
        Tr_velo_to_cam = np.zeros((3,4))
        Tr_velo_to_cam[0:3,0:3] = np.reshape(velo2cam['R'], [3,3])
        Tr_velo_to_cam[:,3] = velo2cam['T']
        data['Tr_velo_to_cam'] = np.reshape(Tr_velo_to_cam, [12])
        data['R0_rect'] = cam2cam['R_rect_00']
        data['P2'] = cam2cam['P_rect_02']
        return data
 
    def cart2hom(self, pts_3d):
        ''' Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        '''
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n,1))))
        return pts_3d_hom
 
    # =========================== 
    # ------- 3d to 3d ---------- 
    # =========================== 
    def  project_velo_to_ref(self, pts_3d_velo):
        pts_3d_velo = self.cart2hom(pts_3d_velo) # nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C))
 
    def project_ref_to_velo(self, pts_3d_ref):
        pts_3d_ref = self.cart2hom(pts_3d_ref) # nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V))
 
    def project_rect_to_ref(self, pts_3d_rect):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
    
    def project_ref_to_rect(self, pts_3d_ref):
        ''' Input and Output are nx3 points '''
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))
 
    def project_rect_to_velo(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        ''' 
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)
 
    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)
    
    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)
 
        img_pts = np.matmul(corners3d_hom, self.P.T)  # (N, 8, 3)
 
        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)
 
        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)
 
        return boxes, boxes_corner
 
 
    # =========================== 
    # ------- 3d to 2d ---------- 
    # =========================== 
    def project_rect_to_image(self, pts_3d_rect):
        ''' Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.cart2hom(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P)) # nx3
        pts_2d[:,0] /= pts_2d[:,2]
        pts_2d[:,1] /= pts_2d[:,2]
        return pts_2d[:,0:2]
    
    def project_velo_to_image(self, pts_3d_velo):
        ''' Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        '''
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)
 
    # =========================== 
    # ------- 2d to 3d ---------- 
    # =========================== 
    def project_image_to_rect(self, uv_depth):
        ''' Input: nx3 first two channels are uv, 3rd channel
                   is depth in rect camera coord.
            Output: nx3 points in rect camera coord.
        '''
        n = uv_depth.shape[0]
        x = ((uv_depth[:,0]-self.c_u)*uv_depth[:,2])/self.f_u + self.b_x
        y = ((uv_depth[:,1]-self.c_v)*uv_depth[:,2])/self.f_v + self.b_y
        pts_3d_rect = np.zeros((n,3))
        pts_3d_rect[:,0] = x
        pts_3d_rect[:,1] = y
        pts_3d_rect[:,2] = uv_depth[:,2]
        return pts_3d_rect
 
    def project_image_to_velo(self, uv_depth):
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)
 
 
def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])
 
 
def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])
 
 
def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
 
 
def transform_from_rot_trans(R, t):
    ''' Transforation matrix from rotation matrix and translation vector. '''
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
 
 
def inverse_rigid_trans(Tr):
    ''' Inverse a rigid body transform matrix (3x4 as [R|t])
        [R'|-R't; 0|1]
    '''
    inv_Tr = np.zeros_like(Tr) # 3x4
    inv_Tr[0:3,0:3] = np.transpose(Tr[0:3,0:3])
    inv_Tr[0:3,3] = np.dot(-np.transpose(Tr[0:3,0:3]), Tr[0:3,3])
    return inv_Tr
 
def read_label(label_filename):
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [Object3d(line) for line in lines]
    return objects
 
def load_image(img_filename):
    return cv2.imread(img_filename)
 
def load_velo_scan(velo_filename):
    scan = np.fromfile(velo_filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))
    return scan
 