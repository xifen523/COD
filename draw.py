import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
plt.rcParams['font.family'] = 'Heiti TC' 


import matplotlib.pyplot as plt
from matplotlib import rcParams
import numpy as np

from matplotlib import pyplot as plt
import matplotlib
a=sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

for i in a:
    print(i)

# config = {
#     "font.family":'serif',
#     "font.size": 18,
#     "mathtext.fontset":'stix',
#     "font.serif": ['SimSun'],
# }
# rcParams.update(config)

# x = np.random.random((10,))


# plt.plot(x,label='随机数')
# plt.title('中文：宋体 \n 英文：$\mathrm{Times \; New \; Roman}$ \n 公式： $\\alpha_i + \\beta_i = \\gamma^k$')
# plt.xlabel('横坐标')
# plt.ylabel('纵坐标')
# plt.legend()
# plt.yticks(fontproperties='Times New Roman', size=18)
# plt.xticks(fontproperties='Times New Roman', size=18)
# plt.show()



def tensorboard_smoothing(x,smooth=0.99):
    x = x.copy()
    weight = smooth
    for i in range(1,len(x)):
        x[i] = (x[i-1] * weight + x[i]) / (weight + 1)
        weight = (weight + 1) * smooth
    return x

def sample_value(x,y,num=100):
    window_size = len(x) / num

    x_sample = [i for i in range(num) ]
    # x_sample = [np.mean(x[int(i*window_size):int((i+1)*window_size)]) for i in range(num)]
    y_sample = [np.mean(y[int(i*window_size):int((i+1)*window_size)]) for i in range(num)]
    return x_sample,y_sample

def get_map_x(x,y,num=80):
    
    ratio = 1.0 / (len(x) - 1) * 80
    new_x = [i*ratio for i in range(len(x))]
    new_y = y.tolist()

    return new_x, new_y


fig, ax1 = plt.subplots(1, 1)    # a figure with a 2x1 grid of Axes
# len_mean = pd.read_csv("/home/dell/workspace/motion/cd/output/cfgs/kitti_models/centerpoint_img_paper2/default/tensorboard/run-.-tag-meta_data_learning_rate.csv")
# len_mean = pd.read_csv("/home/dell/workspace/motion/cd/output/cfgs/kitti_models/centerpoint_img_paper2/default/tensorboard/run-.-tag-train_loss.csv")
len_mean = pd.read_csv("/home/dell/下载/centerpoint__loss.csv")

x = len_mean['Step']
y = tensorboard_smoothing(len_mean['Value'], smooth=0.6)

x_sample, y_sample = sample_value(x, y, 80)
new_x,new_y = get_map_x(x, y, num=80)

ax1.plot(new_x, new_y, color="#3399FF")
# ax1.plot(x_sample, y_sample, color="#3399FF")
# ax1.scatter(x_sample, y_sample, c='r', s=10)
# ax1.plot(len_mean['Step'], tensorboard_smoothing(len_mean['Value'], smooth=0.6), color="#3399FF")
#ax1.set_xticks(np.arange(0, 24, step=2))
ax1.set_xlabel("训练轮次")
# ax1.set_ylabel("Average Episode Length(steps)", color="#3399FF")
ax1.set_ylabel("损失")
# ax1.set_title("centerpoint train/loss")
plt.show()
if  not os.path.exists("./figures"):
    os.makedirs("./figures", exist_ok=True)

fig.savefig(fname='./figures/centerpoint train_loss '+'.pdf', format='pdf')
