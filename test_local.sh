cd tools/

# python test.py --cfg_file ./cfgs/kitti_models/centerpoint_img.yaml --batch_size 4 --ckpt ../output/cfgs/kitti_models/centerpoint_img1/default/ckpt/checkpoint_epoch_60.pth
# python test.py --cfg_file ./cfgs/kitti_models/pillarnet.yaml --batch_size 4 --ckpt ../output/cfgs/kitti_models/pillarnet/default/ckpt/checkpoint_epoch_80.pth
# python test.py --cfg_file ./cfgs/v2x_i_models/second.yaml --batch_size 4 --ckpt ../output/cfgs/v2x_i_models/second/default/ckpt/checkpoint_epoch_80.pth
# python test.py --cfg_file ./cfgs/kitti_models/pillarnet_img.yaml --batch_size 4 --ckpt ../output/cfgs/kitti_models/pillarnet_img/default/ckpt/checkpoint_epoch_80.pth

# ablation
python test.py --cfg_file ./cfgs/kitti_models/centerpoint_img_paper2.yaml --batch_size 4 --ckpt ../output/cfgs/kitti_models/centerpoint_img_paper2/default/ckpt/checkpoint_epoch_80.pth
# python test.py --cfg_file ./cfgs/kitti_models/pillarnet_img_noise_free.yaml --batch_size 4 --ckpt ../output/cfgs/kitti_models/pillarnet_img_noise_free/default/ckpt/checkpoint_epoch_80.pth
