import os


path = '/home/dell/workspace/motion/cd/data/V2X_I/ImageSets/val.txt'
root_split_path = '/home/dell/workspace/motion/cd/data/V2X_I/training/image_2'
# 读取path中的内容
with open(path, 'r') as f:
    lines = f.readlines()

new_lines = []
cut = 0
for index in lines:
    idx = index.strip()
    img_file = os.path.join(root_split_path, ('%s.jpg' % idx))
    if os.path.exists(img_file):
        new_lines.append(index)
    else:
        print(f">>> {idx} not exist")
        cut += 1

print(f"cut {cut} images")

with open(path, 'w') as f:
    f.writelines(new_lines)