import numpy as np
import os

class GenCalibNoisy:
    def __init__(self, calib_path):
        self.calib_path = calib_path

    
    # 生成x轴的干扰噪声
    def _gen_x_noise(self, x_dis):
        Rx = np.array([[1, 0, 0],
               [0, np.cos(x_dis), -np.sin(x_dis)],
               [0, np.sin(x_dis), np.cos(x_dis)]])
        
        return Rx
    
    # 生成y轴的干扰噪声
    def _gen_y_noise(self, y_dis):
        Ry = np.array([[np.cos(y_dis), 0, np.sin(y_dis)],
               [0, 1, 0],
               [-np.sin(y_dis), 0, np.cos(y_dis)]])
        
        return Ry
    
    def _gen_z_noise(self, z_dis):
        Rz = np.array([[np.cos(z_dis), -np.sin(z_dis), 0],
               [np.sin(z_dis), np.cos(z_dis), 0],
               [0, 0, 1]])
        
        return Rz
    
    def _gen_noise_matrix(self, x_dis, y_dis, z_dis):
        Rx = self._gen_x_noise(x_dis)
        Ry = self._gen_y_noise(y_dis)
        Rz = self._gen_z_noise(z_dis)
        
        R = np.dot(Rz, np.dot(Ry, Rx))
        
        return R
    
    def _gen_noise(self, mean_deg, std_deg):
        '''
        input:
            mean_deg: 噪声的均值  单位为角度制
            std_deg:  噪声的标准差  单位为角度制

        return:
            R: 生成的噪声矩阵
        '''
        # 将角度转换为弧度
        mean_rad = np.deg2rad(mean_deg)

        # 转换为弧度
        std_rad = np.deg2rad(std_deg)
        
        x_dis = np.random.normal(mean_deg, std_deg)
        y_dis = np.random.normal(mean_deg, std_deg)
        z_dis = np.random.normal(mean_deg, std_deg)

        R = self._gen_noise_matrix(x_dis, y_dis, z_dis)

        return R
    
    def gen_rotation_noise_calib(self, mean_deg=0, std_deg=5, out_path="output"):
        '''
        input:
            mean_deg: 噪声的均值  单位为角度制
            std_deg:  噪声的标准差  单位为角度制
        
        功能：
            将calib的标注文件加入噪声
        '''
        
        # 获取self.calib_path目录下的所有txt文件
        calib_list = os.listdir(self.calib_path)
        for calib_file in calib_list:
            calib_file_path = os.path.join(self.calib_path, calib_file)
            
            with open(calib_file_path, 'r') as f:
                lines = f.readlines()

            obj = lines[5].strip().split(" ")[1:]
            Tr_velo_to_cam = np.array(obj, dtype=np.float32).reshape(3, 4)
            
            # 计算新的旋转矩阵
            R = self._gen_noise(mean_deg, std_deg)
            new_R = np.dot(R, Tr_velo_to_cam[:, :3])

            # 保留原始平移向量
            new_t = Tr_velo_to_cam[:, 3]

            # 构造新的Tr_velo_to_cam矩阵
            new_Tr_velo_to_cam = np.hstack((new_R, new_t.reshape(-1, 1)))

            # 更新lines[5],使得其为新的Tr_velo_to_cam矩阵
            # lines[5] = "Tr_velo_to_cam: 6.927964000000e-03 -9.999722000000e-01 -2.757829000000e-03 -2.457729000000e-02 -1.162982000000e-03 2.749836000000e-03 -9.999955000000e-01 -6.127237000000e-02 9.999753000000e-01 6.931141000000e-03 -1.143899000000e-03 -3.321029000000e-01"
            lines[5] = "Tr_velo_to_cam: " + " ".join(map(str, new_Tr_velo_to_cam.reshape(-1))) + "\n"

            if not os.path.exists(out_path):
                os.mkdir(out_path)
            save_path = os.path.join(out_path, calib_file)
            with open(save_path, 'w') as f:
                f.writelines(lines)

    def gen_translation_noise_calib(self, mean_m=0, std_m=0.05, out_path="output"):
        '''
        input:
            mean_m: 噪声的均值  单位为米
            std_m:  噪声的标准差  单位为米
        
        功能：
            将calib的标注文件加入平移噪声

        '''
        # 获取self.calib_path目录下的所有txt文件
        calib_list = os.listdir(self.calib_path)
        for calib_file in calib_list:
            calib_file_path = os.path.join(self.calib_path, calib_file)
            
            with open(calib_file_path, 'r') as f:
                lines = f.readlines()
            
            obj = lines[5].strip().split(" ")[1:]
            Tr_velo_to_cam = np.array(obj, dtype=np.float32).reshape(3, 4)
            
            # 计算新的平移向量
            noisy_t = np.array([np.random.normal(mean_m, std_m), np.random.normal(mean_m, std_m), np.random.normal(mean_m, std_m)])
            new_t = Tr_velo_to_cam[:, 3] + noisy_t

            new_Tr_velo_to_cam = np.hstack((Tr_velo_to_cam[:, :3], new_t.reshape(-1, 1)))
            
            # 更新lines[5],使得其为新的Tr_velo_to_cam矩阵
            lines[5] = "Tr_velo_to_cam: " + " ".join(map(str, new_Tr_velo_to_cam.reshape(-1))) + "\n"
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            
            save_path = os.path.join(out_path, calib_file)
            with open(save_path, 'w') as f:
                f.writelines(lines)

    def gen_noise_calib(self, mean_deg=5, std_deg=1, mean_m=0.05, std_m=0.01, out_path="output"):
        '''
        input:
            mean_deg: 噪声的均值  单位为角度制
            std_deg:  噪声的标准差  单位为角度制
            mean_m: 噪声的均值  单位为米
            std_m:  噪声的标准差  单位为米
            out_path: 保存路径
            
        功能：
            将calib的标注文件加入平移噪声和旋转噪声
            
        '''

        # 获取self.calib_path目录下的所有txt文件
        calib_list = os.listdir(self.calib_path)
        for calib_file in calib_list:
            calib_file_path = os.path.join(self.calib_path, calib_file)
            
            with open(calib_file_path, 'r') as f:
                lines = f.readlines()

            obj = lines[5].strip().split(" ")[1:]
            Tr_velo_to_cam = np.array(obj, dtype=np.float32).reshape(3, 4)
            
            # 计算新的旋转矩阵
            R = self._gen_noise(mean_deg, std_deg)
            new_R = np.dot(R, Tr_velo_to_cam[:, :3])
            
            # 计算新的平移向量
            new_t = Tr_velo_to_cam[:, 3] + np.random.normal(mean_m, std_m)
            new_Tr_velo_to_cam = np.hstack((new_R, new_t.reshape(-1, 1)))

            # 更新lines[5],使得其为新的Tr_velo_to_cam矩阵
            lines[5] = "Tr_velo_to_cam: " + " ".join(map(str, new_Tr_velo_to_cam.reshape(-1))) + "\n"
            if not os.path.exists(out_path):
                os.mkdir(out_path)
            
            save_path = os.path.join(out_path, calib_file)
            with open(save_path, 'w') as f:
                f.writelines(lines)

        
            

if __name__ == '__main__':
    # gen = GenCalibNoisy(r'/home/dell/workspace/motion/cd/data/V2X_I/training/calib')
    gen = GenCalibNoisy(r'/home/dell/workspace/motion/cd/data/kitti/training/calib')
    # gen.gen_rotation_noise_calib(mean_deg=0, std_deg=5, out_path=r"E:\calib_noise_rot") # 
    # gen.gen_translation_noise_calib(mean_m=0.0, std_m=0.05, out_path=r"E:\calib_noise_trans")

    ## 消融实验生成的六组噪声数据
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.005, mean_m=0.0, std_m=0.0001, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise1")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.000, mean_m=0.0, std_m=0.0001, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise2")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.005, mean_m=0.0, std_m=0.0000, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise3")

    # gen.gen_noise_calib(mean_deg=0, std_deg=0.01, mean_m=0.0, std_m=0.0002, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise4")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.00, mean_m=0.0, std_m=0.0002, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise5")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.01, mean_m=0.0, std_m=0.0000, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise6")

    # 消融实验补充，平移噪声加大
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.01, mean_m=0.0, std_m=0.0005, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise7")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.00, mean_m=0.0, std_m=0.0005, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise8")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.01, mean_m=0.0, std_m=0.0000, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise9")

# # 消融实验补充，平移噪声加大
#     gen.gen_noise_calib(mean_deg=0, std_deg=0.02, mean_m=0.0, std_m=0.001, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise10")
#     gen.gen_noise_calib(mean_deg=0, std_deg=0.00, mean_m=0.0, std_m=0.001, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise11")
#     gen.gen_noise_calib(mean_deg=0, std_deg=0.02, mean_m=0.0, std_m=0.0000, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise12")
#     print("done")
    
    # # 消融实验补充，平移噪声加大
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.015, mean_m=0.0, std_m=0.0008, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise13")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.00, mean_m=0.0, std_m=0.0008, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise14")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.015, mean_m=0.0, std_m=0.0000, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise15")
    # print("done")

    # 消融实验补充，平移噪声加大
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.012, mean_m=0.0, std_m=0.0007, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise16")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.000, mean_m=0.0, std_m=0.0007, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise17")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.012, mean_m=0.0, std_m=0.0000, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise18")
    # print("done")

     # 消融实验补充，平移噪声加大
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.01, mean_m=0.0, std_m=0.001, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise19")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.000, mean_m=0.0, std_m=0.001, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise20")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.01, mean_m=0.0, std_m=0.0000, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise21")
    # print("done")

    #   # 消融实验补充，平移噪声加大
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.01, mean_m=0.0, std_m=0.002, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise22")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.000, mean_m=0.0, std_m=0.002, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise23")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.01, mean_m=0.0, std_m=0.0000, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise24")
    # print("done")

     # 消融实验补充，平移噪声加大
    gen.gen_noise_calib(mean_deg=0, std_deg=0.00, mean_m=0.0, std_m=0.05, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise25")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.000, mean_m=0.0, std_m=0.002, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise23")
    # gen.gen_noise_calib(mean_deg=0, std_deg=0.01, mean_m=0.0, std_m=0.0000, out_path=r"/home/dell/workspace/motion/cd/data/kitti/training/calib_noise24")
    print("done")