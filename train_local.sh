cd tools/

# python train.py --cfg_file ./cfgs/kitti_models/centerpoint_img_test.yaml

# python train.py --cfg_file ./cfgs/kitti_models/pillarnet.yaml
# python train.py --cfg_file ./cfgs/kitti_models/second.yaml
# python train.py --cfg_file ./cfgs/kitti_models/pointpillar.yaml

# python train.py --cfg_file ./cfgs/kitti_models/pointpillar_img.yaml
# python train.py --cfg_file ./cfgs/kitti_models/second_img.yaml
# python train.py --cfg_file ./cfgs/kitti_models/second_img2.yaml

# noise
# python train.py --cfg_file ./cfgs/kitti_models/pillarnet_img_noise.yaml
# python train.py --cfg_file ./cfgs/kitti_models/centerpoint_img_paper.yaml
# python train.py --cfg_file ./cfgs/kitti_models/centerpoint_img_paper2.yaml








# v2x
# python train.py --cfg_file ./cfgs/v2x_i_models/centerpoint_paper.yaml
# python train.py --cfg_file ./cfgs/v2x_i_models/centerpoint_img_paper.yaml
# python train.py --cfg_file ./cfgs/v2x_i_models/pointpillar_img.yaml
# python train.py --cfg_file ./cfgs/v2x_i_models/pillarnet_img.yaml
# python train.py --cfg_file ./cfgs/v2x_i_models/second_img.yaml
# python train.py --cfg_file ./cfgs/v2x_i_models/second.yaml
# python train.py --cfg_file ./cfgs/v2x_i_models/pointpillar.yaml
# python train.py --cfg_file ./cfgs/v2x_i_models/pillarnet.yaml

# python train.py --cfg_file ./cfgs/kitti_models/pillarnet_img_noise_free.yaml



# Exp added
# kitti
python train.py --cfg_file ./cfgs/kitti_models/voxelnext.yaml

python train.py --cfg_file ./cfgs/kitti_models/voxelnext_img.yaml



# GPU loading test
# python train.py --cfg_file ./cfgs/kitti_models/centerpoint_img_test-copy.yaml
# python train.py --cfg_file ./cfgs/kitti_models/pillarnet_img_noise_free-copy.yaml
# python train.py --cfg_file ./cfgs/kitti_models/centerpoint-copy.yaml
