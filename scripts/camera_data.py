import csv
import numpy as np
import xml.etree.ElementTree as xml
import pandas as pd

from config import config
from scripts import utils

def print_frame_metadata(frame_id, metadata):
    print(f'camera #{ frame_id }:')
    print(f'timestamp: { metadata["timestamp"] }')
    print(f'position: { metadata["position"] }')
    print(f'rotation: { metadata["rotation"] }')
    print()

def prepare_frames(video_run_names, camera_names):
    frames = {}
    i = 0
    for video_run_name in video_run_names:
        frames[video_run_name] = {}
        ranges = config.VIDEO_RUNS_CONFIG[video_run_name]
        for camera_name in camera_names:
            frames[video_run_name][camera_name] = {}
            # Load relevant video and las data
            video_run_data = utils.get_video_run_data(video_run_name, camera_name)
            # Annotations do not cover entire video run, so start at frame x
            for key in range(ranges['min'], ranges['max']):
                # frame_id = f'image_{str(key).zfill(4)}'
                frame_id = f'frame_{str(key).zfill(5)}'
                # Some runs do not contain all consecutive values between min
                # and max range, so check if frame id exists for this run and
                # camera
                if frame_id in video_run_data:
                    frames[video_run_name][camera_name][frame_id] = \
                        video_run_data[frame_id]
                    i+=1
    print('total #frames to process:', i)
    print()
    return frames

def get_video_metadata(video_run_names, camera_names):
    video_metadata = {}

    for video_run_name in video_run_names:
        video_metadata[video_run_name] = {}
        flags = config.VIDEO_RUNS_CONFIG[video_run_name]['ignore_frames']
        for camera_name in camera_names:
            video_metadata[video_run_name][camera_name] = {}
            path = config.get_video_metadata_path(video_run_name, camera_name)
            csv_file = open(path, mode='r')
            csv_reader = csv.reader(csv_file, delimiter=',')
            # csv_reader = pd.read_csv(csv_file)

            # for r in range(csv_reader.shape[0]):
            for row in csv_reader:
                # row = np.array(csv_reader)[r, :]
                frame_id = row[0]
                easting = row[2]
                northing = row[3]
                elevation = row[4]
                roll = row[5]
                pitch = row[6]
                heading = row[7]
                time_utc = utils.gps_sow_to_utc(config.DATE_DATA, row[1])
                time_stamp = time_utc.timestamp()
                # time_stamp = gwpy.time.to_gps(time_utc)
                video_metadata[video_run_name][camera_name][frame_id] = {
                    'timestamp': float(time_stamp),
                    'position': (float(easting), float(northing), float(elevation)),
                    'rotation': (float(roll), float(pitch), float(heading)),
                    'ignore_frame': True if frame_id in flags else False,
                }
    return video_metadata

def get_calibration_data(name, camera_calibration_data, show_info):
    # Luc: ‘CameraToBase’ offsets in this file are not applicable, these are already
    # applied in the external orientation parameters (see \externalOrientation)
    opencv_camera_model, camera_to_base = camera_calibration_data
    center, focal_length, radial_distortion, tangential_distortion = opencv_camera_model

    resolution = opencv_camera_model.attrib
    cx, cy = center.attrib['x'], center.attrib['y']
    fx, fy = focal_length.attrib['x'], focal_length.attrib['y']
    radial_distortion = radial_distortion.attrib
    tangential_distortion = tangential_distortion.attrib
    intrinsics = {
        'id': name,
        'resolution': (resolution['width'], resolution['height']),
        'center': (cx, cy),
        'focal': (fx, fy),
        'distortion': {**radial_distortion, **tangential_distortion}
    }

    if show_info:
        print(f'{ name } general properties:')
        print(f'    resolution: {resolution}')
        print(f'    center: {cx} {cy}')
        print(f'    focal length: {fx} {fy}')
        print()
    return intrinsics

def get_camera_intrinsics(camera_names):
    def get_camera_calibration(camera_name):
        calibration_file = xml.parse(config.CAMERA_CALIBRATION_FILE).getroot()
        camera_calibration = calibration_file[list.index(config.CAMS, camera_name)]
        return camera_calibration

    return [get_calibration_data(cam, get_camera_calibration(cam), show_info=True)
            for cam in camera_names]

def create_camera_matrix(intrinsics):
    # Camera matrix K
    f = intrinsics['focal']
    c = intrinsics['center']
    m = [[f[0], 0.0, c[0]],
         [0.0, f[1], c[1]],
         [0.0, 0.0, 1.0]]
    return np.array(m, dtype=np.float64)


def get_distortion(intrinsics):
    # k3 values are not used
    d = intrinsics['distortion']
    distortion = [d['k1'], d['k2'], d['p1'], d['p2']]
    return np.array(distortion, dtype=np.float64)