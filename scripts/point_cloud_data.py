import laspy
import numpy as np
import open3d
import matplotlib.pyplot as plt
import subprocess

from config import config
from scripts import utils

def get_data(path):
    file = laspy.open(path, mode='r')
    return laspy.convert(file.read(),
                         file_version=config.LASPY_FILE_VERSION,
                         point_format_id=config.LASPY_POINT_FORMAT)

def append_las_to_array(points_acc, las_points):
    # As used by e.g. KITTI format
    # From https://github.com/PRBonn/lidar-bonnetal/issues/78
    x = las_points.X
    y = las_points.Y
    z = las_points.Z
    i = las_points.intensity
    arr = np.zeros(x.shape[0] + y.shape[0] + z.shape[0] + i.shape[0], dtype=np.float32)
    arr[::4] = x
    arr[1::4] = y
    arr[2::4] = z
    arr[3::4] = i
    points_acc = np.append(points_acc, arr)
    return points_acc