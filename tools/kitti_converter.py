import numpy as np
from pathlib import Path
from tqdm import tqdm

from util import box_np_ops
from .kitti_data_utils import get_kitti_image_info


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    for info in tqdm(infos, desc="calculate gt points"):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])


def create_kitti_info_file(data_path,
                           pkl_prefix='kitti',
                           save_path=None,
                           relative_path=True):
    """Create info file of KITTI dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str): Path to save the info file.
        relative_path (bool): Whether to use relative path.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    kitti_infos_train = get_kitti_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        image_ids=train_img_ids,
        relative_path=relative_path)
    _calculate_num_points_in_gt(data_path, kitti_infos_train, relative_path)