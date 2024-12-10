import cv2
import pickle
import numpy as np
def read_pkl(file):
    with open(dir_root + '/' + file, 'rb') as f:
        return pickle.load(f)

# 不同颜色的框增加标注功能
def draw_bbox(img, bbox, color=(0, 255, 0), marker=None):
    for i in range(bbox.shape[0]):
        cv2.rectangle(img, (int(bbox[i][0]), int(bbox[i][1])), (int(bbox[i][2]), int(bbox[i][3])), color, 2)
    
    return img

if __name__ == '__main__':
    img_path = '/home/dell/workspace/motion/cd/data/V2X_I/training/image_2/003445.jpg'
    # dir_root = 'output/cfgs/v2x_i_models/centerpoint_img_paper/default/eval/epoch_80/val/default'
    dir_root = 'output/cfgs/v2x_i_models/pointpillar_img/default/eval/epoch_80/val/default'
    pickle_file = ["result_cal_bbox.pkl", "result_noise_cal_bbox.pkl", "result_noise_pre_bbox.pkl", "result_pre_bbox.pkl"]

    img = cv2.imread(img_path)
    data = read_pkl(pickle_file[0])
    id_dict = {}
    for i in range(len(data)):
        id_dict[data[i]["frame_id"]] = i

    print(id_dict)
    
    index = id_dict["003445"]
    
    cal_bbox = read_pkl(pickle_file[0])[index]["bbox"]
    noise_cal_bbox = read_pkl(pickle_file[1])[index]["bbox"]
    noise_pre_bbox = read_pkl(pickle_file[2])[index]["bbox"]
    pre_bbox = read_pkl(pickle_file[3])[index]["bbox"]
    # pre_bbox_detr = read_pkl(pickle_file[3])[index]["bbox"]

    # 绘制矩形框 bbox 是x1,y1,x2,y2 格式 (n,4)的numpy数组 float32
    img = draw_bbox(img, cal_bbox, (0, 255, 0))
    img = draw_bbox(img, pre_bbox, (255, 255, 0))

    img = draw_bbox(img, noise_cal_bbox, (0, 0, 255))
    img = draw_bbox(img, noise_pre_bbox, (255, 0, 0))
    # 显示图片
    # cv2.imshow("img", img)
    # cv2.waitKey(0)

    #裁减图片
    crop = np.array([[400, 600, 1600, 1000]])
    # 绘制一下裁减区域
    # img_crop = draw_bbox(img,crop, (255, 255, 0))
    img_crop = img[crop[0][1]:crop[0][3], crop[0][0]:crop[0][2]]
    cv2.imshow("img_crop", img_crop)
    cv2.waitKey(0)
    # 保存裁减图片
    cv2.imwrite("img_crop3.jpg", img_crop)

    cv2.destroyAllWindows()