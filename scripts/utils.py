import os
import sys
import time
import glob
import math
import laspy
import ezdxf
import numpy as np
from datetime import datetime, timedelta

from config import config

def get_video_run_data(video_run, cam):
    globs = glob.glob(f'{config.PATH_VIDEO}{video_run}/{cam}/*.jpg')
    filenames = [os.path.basename(path).split('.')[0] for path in globs]
    return dict(zip(filenames, globs))

def get_lidar_run_data(frames_metadata):
    print('retrieving camera and LAS data.. ')
    # Select lidar point cloud tiles surrounding camera at this frame
    las_data = {}
    for cameras in frames_metadata.values():
        # LiDAR data is the same for all cameras in setup,
        # so we choose the middle camera for gathering LiDAR data
        camera = cameras[config.CAMS[1]]
        for frame_metadata in camera.values():
            gps_position = [int(x) for x in frame_metadata['position']]
            filenames_las = extract_lidar_filenames_from_metadata(gps_position)

            for filename_las, path_las in filenames_las.items():
                las_reader = laspy.open(path_las, mode='r')
                if filename_las not in las_data:
                    las_data[filename_las] = las_reader.read()

    # print('done')
    # print('#surrounding tiles:', len(picked_las_data))
    # # print('#total tiles:', len(filenames_las), 'tiles')
    # print('list of surrounding tiles:')
    # pprint(list(picked_las_data.keys()))
    return las_data

def gps_to_utc(timestamp):
    gps_timestamp = timestamp
    gps_epoch_as_gps = datetime(1980, 1, 6)
    # by definition
    gps_time_as_gps = gps_epoch_as_gps + timedelta(seconds=gps_timestamp)
    gps_time_as_tai = gps_time_as_gps + timedelta(seconds=19)  # constant offset
    tai_epoch_as_tai = datetime(1970, 1, 1, 0, 0, 10)
    # by definition
    tai_timestamp = (gps_time_as_tai - tai_epoch_as_tai).total_seconds()
    return datetime.utcfromtimestamp(tai_timestamp)  # "right" timezone is in effect!

# GPS second-of-week
def gps_sow_to_utc(date, sow):
    str_sow, str_sow_microseconds = str(sow).split('.')
    sow_microseconds = int(str_sow_microseconds)
    time_struct = time.gmtime(int(str_sow))
    datetime_obj = datetime.fromtimestamp(time.mktime(time_struct))
    new_datetime_obj = datetime_obj.replace(year=date.year, month=date.month, day=date.day)
    return new_datetime_obj + timedelta(microseconds=sow_microseconds)
    # return time.strftime("%b %d %Y %H:%M:%S", time.gmtime(gps))

# Now we use col and row to map the LiDAR data onto images. The first
# function we use converts HSV to RGB. Please refere to the wikipedia
# article https://en.wikipedia.org/wiki/HSL_and_HSV for more information.
def hsv_to_rgb(h, s, v):
    if s == 0.0:
        return v, v, v

    i = int(h * 6.0)
    f = (h * 6.0) - i
    p = v * (1.0 - s)
    q = v * (1.0 - s * f)
    t = v * (1.0 - s * (1.0 - f))
    i = i % 6

    if i == 0:
        return v, t, p
    if i == 1:
        return q, v, p
    if i == 2:
        return p, v, t
    if i == 3:
        return p, q, v
    if i == 4:
        return t, p, v
    if i == 5:
        return v, p, q

# TODO match corresponding laz files with camera data, using timestamp? Also see:
#  https://gis.stackexchange.com/questions/203392/finding-the-coordinate-system-info-from-a-las-lidar-file-with-python
# Load corresponding point cloud from image based on timestamp/position of image
def extract_lidar_filenames_from_metadata(gps_position, perimeter=30):
    # Get the list of files in LAZ directory
    paths_las = glob.glob(f'{config.PATH_LAZ}*/*.laz')
    filenames_las = [filename.split('\\')[2].split('.')[0] for filename in paths_las]
    tiles = dict(zip(filenames_las, paths_las))
    easting, northing, _ = gps_position

    picked_tiles = {}
    for tile, path in tiles.items():
        _, X, Y = tile.split('_')
        # _, X, Y, _ = re.split('_|[.]', tile)
        X = int(X.split('+')[1])
        Y = int(Y.split('+')[1])

        if X in range(easting - perimeter, easting + perimeter) \
                and Y in range(northing - perimeter, northing + perimeter):
            x, y = str(X).zfill(10), str(Y).zfill(10)
            filename_tile = f'Tile_X+{x}_Y+{y}'
            picked_tiles[filename_tile] = path
    return picked_tiles

def Rx(theta):
    return np.matrix([[1, 0, 0],
                      [0, math.cos(theta), -math.sin(theta)],
                      [0, math.sin(theta), math.cos(theta)]])


def Ry(theta):
    return np.matrix([[math.cos(theta), 0, math.sin(theta)],
                      [0, 1, 0],
                      [-math.sin(theta), 0, math.cos(theta)]])


def Rz(theta):
    return np.matrix([[math.cos(theta), -math.sin(theta), 0],
                      [math.sin(theta), math.cos(theta), 0],
                      [0, 0, 1]])

def euler_to_rotation_matrix(pitch, roll, yaw):
    pitch = math.radians(pitch)
    roll = math.radians(roll)
    yaw = math.radians(yaw)
    # R = RzRyRx
    return Rx(pitch) @ Ry(roll) @ Rz(yaw)


# From https://automaticaddison.com/how-to-convert-euler-angles-to-quaternions-using-python/
def get_quaternion_from_euler(roll, pitch, yaw):
    """
    Convert an Euler angle to a quaternion.

    Input
      :param roll: The roll (rotation around x-axis) angle in radians.
      :param pitch: The pitch (rotation around y-axis) angle in radians.
      :param yaw: The yaw (rotation around z-axis) angle in radians.

    Output
      :return qx, qy, qz, qw: The orientation in quaternion (x,y,z,w) format
    """
    qx = np.sin(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) - np.cos(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)
    qy = np.cos(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2)
    qz = np.cos(roll / 2) * np.cos(pitch / 2) * np.sin(yaw / 2) - np.sin(roll / 2) * np.sin(pitch / 2) * np.cos(yaw / 2)
    qw = np.cos(roll / 2) * np.cos(pitch / 2) * np.cos(yaw / 2) + np.sin(roll / 2) * np.sin(pitch / 2) * np.sin(yaw / 2)

    return qx, qy, qz, qw

def read_dwg_file(path):
    print('Reading DXF file..', end='')
    try:
        annotations = ezdxf.readfile(path)
    except IOError:
        print(f'Not a DXF file or a generic I/O error.')
        sys.exit(1)
    except ezdxf.DXFStructureError:
        print(f'Invalid or corrupted DXF file.')
        sys.exit(2)
    print('done')
    print('DXF version', annotations.dxfversion)
    print('AutoCAD release name', annotations.acad_release)
    return annotations