import sys
import os
import re
import cv2
import json
import math
import ezdxf
import laspy
import shutil
import numpy as np
import open3d

from config import config
from scripts import utils, kitti_utils, camera_data, overlay_data_camera, point_cloud_data

def create_bbox_2d(pts):
    number_of_points = np.size(pts, axis=0)
    pts = pts.reshape(number_of_points, 2)
    u_min, v_min = np.amin(pts, axis=0)
    u_max, v_max = np.amax(pts, axis=0)

    return u_min, v_min, u_max, v_max

# From https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python
def angle_between(v1, v2):
    x1, y1, _ = v1
    x2, y2, _ = v2
    dot = x1*x2 + y1*y2          # dot product between [x1, y1] and [x2, y2]
    det = x1*y2 - y1*x2          # determinant
    return math.atan2(det, dot)  # atan2(y, x) or atan2(sin, cos)

def extrude_cube(mesh_dxf, center, width, depth, height, angle):
    w, d = width * 0.5, depth * 0.5
    cube_vertices = [
        (-w, -d, 0),
        (w, -d, 0),
        (w, d, 0),
        (-w, d, 0),
        (-w, -d, height),
        (w, -d, height),
        (w, d, height),
        (-w, d, height),
    ]
    cube_faces = [
        [0, 1, 2, 3],
        [4, 5, 6, 7],
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [3, 2, 6, 7],
        [0, 3, 7, 4]
    ]
    x, y, z = center.xyz
    rmat = ezdxf.math.Matrix44().xyz_rotate(0, 0, angle)
    tmat = ezdxf.math.Matrix44().translate(x, y, z)

    # do not subdivide cube, 0 is the default value
    mesh_dxf.dxf.subdivision_levels = 0
    with mesh_dxf.edit_data() as mesh_dxf_data:
        mesh_dxf_data.vertices = cube_vertices
        mesh_dxf_data.faces = cube_faces
    mesh_dxf.transform(rmat)
    mesh_dxf.transform(tmat)


def find_track_angle(point, polylines):
    # First find closest track point
    # Track lines have many points, so this is close to the shortest orthogonal distance)
    angle = 0
    closest_track_point = ezdxf.math.Vec3((0, 0, 0))
    for polyline in polylines:
        track_point = ezdxf.math.closest_point(point, polyline.points())
        rvec = point - track_point

        if ezdxf.math.Vec3(track_point).distance(point) < closest_track_point.distance(point):
            angle = rvec.angle
            closest_track_point = track_point

    return angle

def extrude_cube_from_shape(mesh_dxf, points, height):
    # hardcoded vertices and faces for now
    p = points
    cube_vertices = [
        p[0],
        p[1],
        p[2],
        p[3],
        (p[0][0], p[0][1], p[0][2] + height),
        (p[1][0], p[1][1], p[1][2] + height),
        (p[2][0], p[2][1], p[2][2] + height),
        (p[3][0], p[3][1], p[3][2] + height),
    ]
    cube_faces = [
        [0, 1, 2, 3],  # bottom plane
        [4, 5, 6, 7],  # top plane
        [0, 1, 5, 4],
        [1, 2, 6, 5],
        [3, 2, 6, 7],
        [0, 3, 7, 4]
    ]

    # do not subdivide cube, 0 is the default value
    mesh_dxf.dxf.subdivision_levels = 0
    with mesh_dxf.edit_data() as mesh_dxf_data:
        mesh_dxf_data.vertices = cube_vertices
        mesh_dxf_data.faces = cube_faces

def extract_dwg_data(dwg, only_extract_objects=True, only_extract_rails=False):
    # MESH requires DXF R2000 or later
    # (https://ezdxf.readthedocs.io/en/master/layouts/layouts.html?highlight=modelspace#ezdxf.layouts.BaseLayout.add_mesh)
    dwg_new = ezdxf.new('R2007')
    dwg_new.appids.add(config.DXF_APPID)
    msp = dwg.modelspace()
    msp_new = dwg_new.modelspace()

    rails = msp.query('*[layer=="SU-TR-Rails-Z"]').query('POLYLINE')
    signals = msp.query('*[layer=="TL-TL-Signal light-Z"]').query('CIRCLE')
    signs = msp.query('*[layer=="SU-TR-Markers-Z"]').query('CIRCLE')
    # consists of Polylines and LWPolylines
    cabinets = msp.query('*[layer=="SU-UT-Cabinet-Z"]')

    if only_extract_rails:
        objects = ezdxf.query.new() \
            .extend(rails)
    elif only_extract_objects:
        objects = ezdxf.query.new() \
            .extend(signals) \
            .extend(signs) \
            .extend(cabinets)
    else:
        objects = ezdxf.query.new() \
            .extend(signals) \
            .extend(signs) \
            .extend(cabinets) \
            .extend(rails)

    idx = 0

    # create list to store the radius & dimensions of objects
    radius_signal_light = []
    radius_marker = []
    e_cabinet = []
    points_cabinet = []

    for obj in objects:
        layer_name = obj.dxf.layer
        layer_obj = dwg.layers.get(layer_name)
        c = config.CONFIG_BBOXES[layer_name]
        obj_name = c['name']
        height = c['height']
        resize = c['resize_factor']
        mesh_dxf = None

        if obj_name == 'Rails':
            mesh_dxf = msp_new.add_polyline3d(obj.points(), dxfattribs={'layer': layer_name})
        if obj_name == 'Signal_light':
            # Create wayside object bbox aligned with rails
            # Only create mesh here, otherwise file will contain empty objects and cannot be opened
            mesh_dxf = msp_new.add_mesh(dxfattribs={'layer': layer_name})
            track_angle = find_track_angle(obj.dxf.center, rails)
            extrude_cube(mesh_dxf, obj.dxf.center,
                         width=obj.dxf.radius * resize,
                         depth=obj.dxf.radius * resize,
                         height=height,
                         angle=track_angle)
            radius_signal_light.append(obj.dxf.radius)
        if obj_name == 'Marker':
            mesh_dxf = msp_new.add_mesh(dxfattribs={'layer': layer_name})
            track_angle = find_track_angle(obj.dxf.center, rails)
            extrude_cube(mesh_dxf, obj.dxf.center,
                         width=obj.dxf.radius * resize,
                         depth=obj.dxf.radius * resize,
                         height=height,
                         angle=track_angle)
            radius_marker.append(obj.dxf.radius)
        if obj_name == 'Cabinet':
            if obj.dxftype() == 'LWPOLYLINE':
                e = obj.dxf.elevation
                e_cabinet.append(obj.dxf.elevation)
                if e > 0:
                    mesh_dxf = msp_new.add_mesh(dxfattribs={'layer': layer_name})
                    points = obj.get_points('xy')
                    points = [(x, y, e) for x, y in points]
                    points_cabinet.append(points)
                    extrude_cube_from_shape(mesh_dxf, points, height)
            else:
                mesh_dxf = msp_new.add_mesh(dxfattribs={'layer': layer_name})
                points = list(obj.points())
                points_cabinet.append(points)
                extrude_cube_from_shape(mesh_dxf, points, height)

        if not dwg_new.layers.has_entry(layer_name):
            dwg_new.layers.new(name=layer_name, dxfattribs={'linetype': 'CONTINUOUS', 'color': layer_obj.color})

        if mesh_dxf is not None:
            if only_extract_rails:
                type_name = 'rails'
            elif only_extract_objects:
                type_name = 'bbox'
            else:
                type_name = 'unknown'
            # Storing bbox ids in DXF file, such that they are in one place
            # Group code '1000' from: http://docs.autodesk.com/ACAD_E/2012/ENU/filesDXF/WS1a9193826455f5ff18cb41610ec0a2e719-7a62.htm
            mesh_dxf.set_xdata(config.DXF_APPID, [(1000, f'{type_name}#{idx}')])

        idx += 1

    l_cabinets = []
    w_cabinets = []
    for point in points_cabinet:
        l = math.dist(point[0], point[1])
        l_cabinets.append(l)
        w = math.dist(point[1], point[2])
        w_cabinets.append(w)

    print('extracted layers in new file:')
    print('\n'.join([f'{layer.dxf.name}' for layer in dwg_new.layers]))
    return dwg_new

def convert_to_kitti_format(dwg: object,
                            frames: object,
                            video_metadata: object,
                            cameras_intrinsics: object,
                            output_path: object,
                            shuffle: object) -> object:
    msp = dwg.modelspace()
    bboxes = msp.query('MESH[layer!="0"]')

    data_conversion = DataConversionKITTI(
        frames, video_metadata, cameras_intrinsics, bboxes,
        output_path, shuffle, store_imageset_outside=True
    )
    data_conversion.split_data()
    data_conversion.prepare_directory()
    data_conversion.print_objects_info()

    mapping_kitti_fugro_ids_txt = f'{output_path}/flags/fugro-to-kitti-ids.txt'
    with open(mapping_kitti_fugro_ids_txt, 'w') as f:
        data_conversion.convert(f)

def get_square_center(v):
    xs = [p[0] for p in v]
    ys = [p[1] for p in v]
    # bottom coords
    zs = [p[2] for p in v[0:4]]
    centroid = (sum(xs) / len(v), sum(ys) / len(v), sum(zs) / len(v[0:4]))
    return centroid

def get_cube_dim(v):
    v = np.asarray(v)
    x_min, y_min, z_min = np.amin(v, axis=0)
    x_max, y_max, z_max = np.amax(v, axis=0)
    h = z_max - z_min
    l = y_max - y_min
    w = x_max - x_min
    # h = np.linalg.norm(v[0] - v[4])
    # w = np.linalg.norm(v[0] - v[1])
    # l = np.linalg.norm(v[1] - v[2])
    return round(h, 4), round(w, 4), round(l, 4)

class DataConversionKITTI:
    def __init__(self, frames, video_metadata, cameras_intrinsics, objects,
                 output_path='', shuffle=False, store_imageset_outside=False):
        self.frames = frames
        self.video_metadata = video_metadata
        self.cameras_intrinsics = cameras_intrinsics
        self.objects = objects
        self.output_path = output_path
        self.shuffle = shuffle
        self.splits = ['training', 'testing']
        # count total number of elements from all runs
        self.size = sum(sum(len(x) for x in v.values()) for v in self.frames.values())
        self.kitti_ids = list(range(self.size))
        self.store_imageset_outside = store_imageset_outside

    def split_data(self):
        self.trainval_txt, self.train_txt, self.val_txt, self.test_txt = self.get_imageset_paths()
        self.train_split, self.val_split, self.test_split, self.trainval_split = self._split_data()

    def _split_data(self):
        if self.shuffle:
            print('shuffle enabled, create new dataset?')
            input('Press Enter to continue...')

            train_ids, test_ids, val_ids = self.split_ids()

            train_split = train_ids
            val_split = val_ids
            test_split = test_ids
            trainval_split = train_ids + val_ids
            assert (len(trainval_split) == len(train_ids) + len(val_ids))
        else:
            trainval_split = self.read_imageset_file(self.trainval_txt)
            train_split = self.read_imageset_file(self.train_txt)
            val_split = self.read_imageset_file(self.val_txt)
            test_split = self.read_imageset_file(self.test_txt)
        return train_split, val_split, test_split, trainval_split

    def split_ids(self):
        np.random.shuffle(self.kitti_ids)

        train_split = math.ceil(self.size * config.TRAIN_SPLIT_SIZE)
        test_split = math.floor(self.size * config.TEST_SPLIT_SIZE)
        val_split = math.floor(self.size * config.VAL_SPLIT_SIZE)
        assert (train_split + test_split + val_split == self.size)
        train_ids = self.kitti_ids[:train_split]
        test_ids = self.kitti_ids[train_split:train_split + test_split]
        val_ids = self.kitti_ids[train_split + test_split:train_split + test_split + val_split]
        train_ids.sort()
        test_ids.sort()
        val_ids.sort()
        assert (len(train_ids) + len(test_ids) + len(val_ids) == self.size)
        S1, S2, S3 = set(train_ids), set(test_ids), set(val_ids)
        assert (len(set.intersection(S1, S2, S3)) == 0)
        return train_ids, test_ids, val_ids

    def get_imageset_paths(self):
        if self.store_imageset_outside:
            trainval_txt = f'{self.output_path}ImageSets/trainval.txt'
            train_txt = f'{self.output_path}ImageSets/train.txt'
            val_txt = f'{self.output_path}ImageSets/val.txt'
            test_txt = f'{self.output_path}ImageSets/test.txt'
        else:
            trainval_txt = f'{self.output_path}training/ImageSets/trainval.txt'
            train_txt = f'{self.output_path}training/ImageSets/train.txt'
            val_txt = f'{self.output_path}training/ImageSets/val.txt'
            test_txt = f'{self.output_path}testing/ImageSets/test.txt'
        return trainval_txt, train_txt, val_txt, test_txt

    def prepare_directory(self):
        for split in self.splits:
            dir_calib = f'{self.output_path}{split}/calib/'
            dir_image = f'{self.output_path}{split}/image_2/'
            dir_label = f'{self.output_path}{split}/label_2/'
            dir_ImageSets = f'{self.output_path}/ImageSets/' \
                if self.store_imageset_outside \
                else f'{self.output_path}{split}/ImageSets/'
            dir_velo = f'{self.output_path}{split}/velodyne/'
            dir_flags = f'{self.output_path}/flags/'

            for dir in [dir_calib, dir_image, dir_label, dir_ImageSets, dir_velo, dir_flags]:
                if not os.path.exists(dir):
                    os.makedirs(dir)

    def append_imageset_file(self, txt_file, idx):
        with open(txt_file, 'a') as f:
            f.write(f'{kitti_utils.filename(idx)}\n')

    def write_imageset_file(self, txt_file, ids):
        with open(txt_file, 'w') as f:
            f.writelines(f'{kitti_utils.filename(idx)}\n' for idx in ids)

    def read_imageset_file(self, txt_file):
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            split = [int(line.rstrip()) for line in lines]
        return split

    def create_calib(self, camera_intrinsics, frame_metadata):
        rx, ry, rz = frame_metadata['rotation']  # Roll, pitch, heading
        tx, ty, tz = frame_metadata['position']
        w, h = int(camera_intrinsics['resolution'][0]), int(camera_intrinsics['resolution'][1])
        camera_origin = np.float64([tx, ty, tz - config.Z_CORRECTION])

        # From world to camera coordinate system
        # Camera rotation vector
        rvec = np.float64(utils.euler_to_rotation_matrix(90 - ry, -rx, rz))
        # Rotate translation vector
        tvec = rvec.dot(-1 * camera_origin)
        cam_matrix = camera_data.create_camera_matrix(camera_intrinsics)  # matrix K
        distortion = camera_data.get_distortion(camera_intrinsics)
        cam_matrix_3x4 = np.hstack((cam_matrix, np.asarray([0] * 3).reshape(3, 1)))
        cam_lidar_matrix_3x4 = np.hstack((np.asarray(rvec), np.asarray(tvec).reshape(3, 1)))
        # From camera to world coordinate system
        inv_cam_lidar_matrix_3x4 = kitti_utils.inverse_rigid_trans(cam_lidar_matrix_3x4)

        rect_cam_matrix = np.identity(3)
        # rect_cam_matrix, _ = cv2.getOptimalNewCameraMatrix(cam_matrix, distortion, (w,h), 1, (w,h))

        lines = {
            'P0': ' '.join(' '.join(str(x) for x in y) for y in cam_matrix_3x4),
            # Create P1 & P2 lines according to KITTI dataset format
            'P1': ' '.join(' '.join(str(x) for x in y) for y in cam_matrix_3x4),
            'P2': ' '.join(' '.join(str(x) for x in y) for y in cam_matrix_3x4),
            'P3': ' '.join(' '.join(str(x) for x in y) for y in cam_matrix_3x4),
            'R0_rect': ' '.join(' '.join(str(x) for x in y) for y in rect_cam_matrix),
            'Tr_velo_to_cam': ' '.join(' '.join(str(x) for x in y) for y in cam_lidar_matrix_3x4),
            # IMU positions are taken from coordinates of camera
            'Tr_imu_to_velo': ' '.join(' '.join(str(x) for x in y) for y in inv_cam_lidar_matrix_3x4),
        }
        lines = [f'{k}: {v}' for k, v in lines.items()]

        return lines

    def create_labels_file(self, cam_intrinsics, frame_metadata, frame_id, video_run_name):
        rx, ry, rz = frame_metadata['rotation']  # Roll, pitch, heading
        tx, ty, tz = frame_metadata['position']
        w, h = int(cam_intrinsics['resolution'][0]), int(cam_intrinsics['resolution'][1])
        camera_origin = np.float64([tx, ty, tz - config.Z_CORRECTION])

        # From world to camera coordinate system
        # Camera rotation vector
        rvec = np.float64(utils.euler_to_rotation_matrix(90 - ry, -rx, rz))
        # Rotate translation vector
        tvec = rvec.dot(-1 * camera_origin)
        cam_matrix = camera_data.create_camera_matrix(cam_intrinsics)  # matrix K
        distortion = camera_data.get_distortion(cam_intrinsics)

        camera_transformation = np.hstack((rvec, tvec.reshape(3, 1)))
        # 4x4 transformation matrix
        camera_transformation = np.vstack((camera_transformation, [[0, 0, 0, 1]]))

        lines_labels = []
        lines_labels_flags = []
        trunc = 0.00
        occl = 0
        for obj in self.objects:
            layer_name = obj.dxf.layer
            obj_name = config.CONFIG_BBOXES[layer_name]['name']
            # Centroid (x, y, z) projection from LiDAR coordinate frame to camera coordinate frame

            vs = list(obj.vertices)
            # Cube position defined by center of bottom square
            cube_world_position = get_square_center(vs)

            # Transformation to camera coordinates (3D)
            homogenous_coords = np.vstack((*cube_world_position, 1))
            cube_camera_position = np.asarray(camera_transformation * homogenous_coords)
            cube_camera_position = np.hstack(cube_camera_position[:-1])

            # Transformation to image coordinates (2D) using OpenCV
            # Clip bboxes behind camera and x meters away from camera
            v = list(obj.vertices)
            v.append(cube_world_position)
            vertices = np.asarray(v, dtype=np.float64)
            object_points_world = np.vstack(vertices)

            # Clip points behind and x meters away from camera
            object_points_closeby = overlay_data_camera.clip_points_from_plane(
                object_points_world, config.FARAWAY_DISTANCE, rvec, tvec, clip_behind_plane=True
            )
            object_points_clipped = overlay_data_camera.clip_points_from_plane(
                object_points_closeby, config.CLOSEBY_DISTANCE, rvec, tvec, clip_behind_plane=False
            )

            if len(object_points_clipped) != 0:
                image_points, _ = cv2.projectPoints(
                    np.float64(object_points_clipped), rvec, tvec, cam_matrix, distortion
                )
                pts = np.array(image_points, np.int32)

                pts_reshape = np.squeeze(pts, axis=1)
                x = pts_reshape[:, 0]
                y = pts_reshape[:, 1]
                coords_within_image_bounds = np.size(x[(0 <= x) & (x <= w)]) + np.size(y[(0 <= y) & (y <= h)])

                # All parts of bbox should be visible in image
                if coords_within_image_bounds >= 8 * 2:  # both x & y coords
                    # Vector direction from the camera to the object
                    # From https://github.com/NVlabs/Deep_Object_Pose/issues/86
                    # & https://github.com/traveller59/second.pytorch/issues/98
                    alpha = angle_between(cube_camera_position, [1, 0, 0])
                    bbox_2d = create_bbox_2d(pts)
                    bbox_str = tuple(map(str, bbox_2d))
                    dim = get_cube_dim(vs)
                    dim_str = tuple(map(str, dim))
                    loc = ' '.join(cube_camera_position.astype(str))
                    rot_y = 0  # angle_between(normal_vec, [1, 0, 0])
                    line_label = f'{obj_name} {trunc} {occl} {alpha} {" ".join(bbox_str)} {" ".join(dim_str)} {loc} {rot_y}'

                    tags = obj.get_xdata(config.DXF_APPID)
                    bbox_id = tags[0].value
                    ignore_objects_ids = config.VIDEO_RUNS_CONFIG[video_run_name]['ignore_objects']

                    if bbox_id in ignore_objects_ids:
                        flag = ignore_objects_ids[bbox_id]
                        if type(flag) is list and frame_id in flag:
                            # Store position removed objects in other file
                            line_label_flags = f'{bbox_id}:{line_label}'
                        elif type(flag) is bool and flag:
                            # Store position removed objects in other file
                            line_label_flags = f'{bbox_id}:{line_label}'
                        else:
                            # Store id only of not removed objects in flag file,
                            # visualization of these objects is done via KITTI label files
                            line_label_flags = f'{bbox_id}:'
                            lines_labels.append(line_label)
                    # All objects are removed from frame, so store all ids
                    elif frame_metadata['ignore_frame']:
                        line_label_flags = f'{bbox_id}:{line_label}'
                    else:
                        line_label_flags = f'{bbox_id}:'
                        lines_labels.append(line_label)

                    lines_labels_flags.append(line_label_flags)

        print(f'#visible bboxes in frame {frame_id}: ', len(lines_labels))

        return lines_labels, lines_labels_flags

    def create_velo_file(self, frame_metadata, is_dummy=False):
        pcd_points = np.empty(0)
        if not is_dummy:
            gps_position = [int(x) for x in frame_metadata['position']]
            filenames_las = utils.extract_lidar_filenames_from_metadata(gps_position, perimeter=0)
            for path_las in filenames_las.values():
                with laspy.open(path_las) as inlas:
                    for points in inlas.chunk_iterator(2_000_000):
                        pcd_points = point_cloud_data.append_las_to_array(pcd_points, points)
        return pcd_points

    def print_objects_info(self):
        n_signals = 0
        n_markers = 0
        n_cabinets = 0
        dimensions_signals = []
        dimensions_markers = []
        dimensions_cabinets = []

        for obj in self.objects:
            c = config.CONFIG_BBOXES[obj.dxf.layer]
            obj_name = c['name']
            tags = obj.get_xdata(config.DXF_APPID)
            bbox_id = tags[0].value
            ignore_objects_ids = [config.VIDEO_RUNS_CONFIG[video_run_name]['ignore_objects']
                                  for video_run_name in self.frames.keys()]

            # Only count bboxes not flagged in config
            if not any(bbox_id in id for id in ignore_objects_ids):
                if obj_name == 'Signal_light':
                    vertices = np.asarray(obj.vertices)
                    dim = get_cube_dim(vertices)

                    n_signals += 1
                    dimensions_signals.append(dim)
                if obj_name == 'Marker':
                    vertices = np.asarray(obj.vertices)
                    dim = get_cube_dim(vertices)

                    n_markers += 1
                    dimensions_markers.append(dim)
                if obj_name == 'Cabinet':
                    vertices = np.asarray(obj.vertices)
                    dim = get_cube_dim(vertices)

                    n_cabinets += 1
                    dimensions_cabinets.append(dim)

        self.n_signals = n_signals
        self.n_markers = n_markers
        self.n_cabinets = n_cabinets
        self.mean_signals = np.mean(dimensions_signals, axis=0)
        self.mean_markers = np.mean(dimensions_markers, axis=0)
        self.mean_cabinets = np.mean(dimensions_cabinets, axis=0)
        n_count = f'#bboxes per category: signals = {n_signals}, markers = {n_markers}, cabinets = {n_cabinets}'
        info_signals = f'average signal dimensions: {np.round(self.mean_signals, 2).tolist()}'
        info_markers = f'average marker dimensions: {np.round(self.mean_markers, 2).tolist()}'
        info_cabinets = f'average cabinet dimensions: {np.round(self.mean_cabinets, 2).tolist()}'
        print(n_count)
        print(info_signals)
        print(info_markers)
        print(info_cabinets)
        with open('info-bboxes-count.txt', 'w') as f:
            f.write(n_count + '\n')
            f.write(info_signals + '\n')
            f.write(info_markers + '\n')
            f.write(info_cabinets + '\n')

    def write_kitti_files(self, split, filename_kitti, lines_calib,
                          lines_labels, image_path, points_bin):
        o = self.output_path
        with open(f'{o}{split}/calib/{filename_kitti}.txt', 'w') as f:
            f.writelines(f'{line}\n' for line in lines_calib)
        with open(f'{o}{split}/label_2/{filename_kitti}.txt', 'w') as f:
            f.writelines(f'{line}\n' for line in lines_labels)

        dst = f'{o}{split}/image_2/{filename_kitti}.jpg'
        if not os.path.isfile(dst):
            shutil.copy(image_path, dst)

        # Create point cloud files since MMDetection3D framework expects these
        # files to exist, but they are not used in this implementation
        points_bin.astype('float32').tofile(f'{o}{split}/velodyne/{filename_kitti}.bin')

    def convert(self, kitti_ids_file):
        idx = 0
        train_split = []
        val_split = []
        trainval_split = []
        test_split = []
        for video_run_name, camera_names in self.frames.items():
            for camera_name, frame_data in camera_names.items():
                for frame_id, image_path in frame_data.items():
                    frame_metadata = self.video_metadata[video_run_name][camera_name][frame_id]
                    cam_intrinsics = list(
                        filter(lambda cam: cam['id'] == camera_name, self.cameras_intrinsics)
                    )[0]
                    # camera_data.print_frame_metadata(frame_id, frame_metadata)
                    filename_kitti = kitti_utils.filename(idx)

                    lines_labels, lines_labels_flags = self.create_labels_file(
                        cam_intrinsics, frame_metadata, frame_id, video_run_name
                    )
                    lines_calib = self.create_calib(cam_intrinsics, frame_metadata)
                    points_bin = self.create_velo_file(frame_metadata, is_dummy=True)

                    # If shuffle is enabled and label is empty, add idx to test split
                    # and continue to next idx, otherwise training dataset consists
                    # mostly of empty samples.
                    if self.shuffle and len(lines_labels) == 0:
                        split = 'testing'
                        test_split.append(idx)
                    else:
                        if idx in self.train_split:
                            split = 'training'
                            train_split.append(idx)
                        if idx in self.val_split:
                            split = 'training'
                            val_split.append(idx)
                        if idx in self.trainval_split:
                            split = 'training'
                            trainval_split.append(idx)
                        if idx in self.test_split:
                            split = 'testing'
                            test_split.append(idx)

                    self.write_kitti_files(
                        split, filename_kitti, lines_calib, lines_labels, image_path, points_bin
                    )

                    # Save mapping from Fugro frame ids to KITTI format ids
                    kitti_ids_file.write(f'{filename_kitti} {video_run_name} {frame_id}\n')

                    if len(lines_labels_flags) != 0:
                        with open(f'{self.output_path}flags/{filename_kitti}.txt', 'w') as f:
                            f.writelines(f'{line}\n' for line in lines_labels_flags)

                    idx += 1

        train_split.sort()
        val_split.sort()
        trainval_split.sort()
        test_split.sort()
        self.write_imageset_file(self.train_txt, train_split)
        self.write_imageset_file(self.val_txt, val_split)
        self.write_imageset_file(self.trainval_txt, trainval_split)
        self.write_imageset_file(self.test_txt, test_split)