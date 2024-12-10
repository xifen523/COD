
import yaml

def cfg_from_yaml_file(cfg_file):
    with open(cfg_file, 'r') as f:
        try:
            new_config = yaml.safe_load(f, Loader=yaml.FullLoader)
        except:
            new_config = yaml.safe_load(f)

    return new_config


if __name__ == "__main__":
    file = "/home/lk/workbase/python/OpenPCDet/tools/cfgs/kitti_models/img_rtdetr.yaml"
    cfg_from_yaml_file(file)