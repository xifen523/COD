import pickle as pkl
import pandas as pd
import os

def read_pkl(path):
    with open(path, 'rb') as f:
        data = pkl.load(f)
    return data

def read_txt(path):
    with open(path, 'r') as f:
        data = f.readlines()
    return data


def parse_eval(path):
    with open(path, 'r') as f:
        data = f.readlines()

    info_line = []
    flag_info = False
    for i, info in enumerate(data):
        if "INFO  Car AP@0.70, 0.70, 0.70:" in info:
            info_line.append(i)
            flag_info = True
        if flag_info:
            if not "AP" in info:
                flag_info = False
            else:
                info_line.append(i)
    
    valid_info = data[info_line[0]:info_line[-1]+1]
    valid_info[0] =  valid_info[0][32:]

    return valid_info
    


def save_data(path, data):
    """保存数据到excel文件"""


    data = [i.split() for i in data]
    # 每5行增加空行
    # 在每五行之后插入一行空白行
    for i in range(1, len(data) // 5 + 1):
        data.insert(i * 6 - 1, [None, None, None])
    df = pd.DataFrame(data)

    
    #将path路径的.txt改为.xlsx
    excel_file = path.replace('.txt', '.xlsx')

    # 将DataFrame保存到Excel文件
    df.to_excel(excel_file, index=False)
    print(f'Data has been saved to {excel_file}.')

if __name__ == '__main__':
    # path = r'/home/dell/workspace/motion/cd/output/cfgs/kitti_models/pillarnet_img/default/eval/epoch_80/val/default/result.pkl'
    # data = read_pkl(path)
    # print(data)

    # path = r'output/cfgs/kitti_models/centerpoint_img_paper2/default/eval/epoch_80/val/default/log_eval_20240429-151734.txt'
    # data = parse_eval(path)
    # save_data(path, data)

    for filename in os.listdir('output/cfgs/kitti_models/centerpoint_img_paper2/default/eval/epoch_80/val/default/'):
        if filename.startswith('log_eval_20240429') and filename.endswith('.txt'):
            path = os.path.join('output/cfgs/kitti_models/centerpoint_img_paper2/default/eval/epoch_80/val/default/', filename)
            data = parse_eval(path)
            save_data(path, data)
            print(f'{filename} has been processed.')