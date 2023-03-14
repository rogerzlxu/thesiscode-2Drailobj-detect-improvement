from cProfile import label
import os
import cv2
import math
import laspy
import ezdxf
import pickle
import open3d
import numpy as np
from matplotlib import pyplot as plt

from config import config
from scripts import training_data, point_cloud_data, kitti_utils, utils

FONT = cv2.FONT_HERSHEY_PLAIN


class ObjectData:
    '''
    Load and parse object data into a usable format.
    '''

    def __init__(self, root_dir, config, split='testing', file_pred_pkl=None, imageset_outside=False):
        '''
        root_dir contains training and testing folders
        '''
        if split == 'train' or split == 'test' \
                or split == 'trainval' or split == 'val':
            pass
        else:
            print('Unknown split: %s' % (split))
            exit(-1)
        self.root_dir = root_dir
        self.split = split
        # Contains custom user data (for visualization)
        self.config = config
        print(root_dir, split)
        self.split_dir = os.path.join(root_dir, 'training') \
            if self.split != 'test' \
            else os.path.join(root_dir, 'testing')

        self.image_dir = os.path.join(self.split_dir, 'image_2')
        self.label_dir = os.path.join(self.split_dir, 'label_2')
        self.calib_dir = os.path.join(self.split_dir, 'calib')
        self.sets_dir = os.path.join(self.root_dir, 'ImageSets') \
            if imageset_outside \
            else os.path.join(self.split_dir, 'ImageSets')
        self.flags_dir = os.path.join(self.root_dir, 'flags')

        lidar_dir = 'lidar'
        pred_dir = 'pred'

        self.lidar_dir = os.path.join(self.split_dir, lidar_dir)
        self.pred_dir = os.path.join(self.split_dir, pred_dir)
        self.file_pred_pkl = file_pred_pkl

        with open(f'{self.sets_dir}/{self.split}.txt', 'r') as f:
            lines = f.readlines()
            self.indices = [int(line.rstrip()) for line in lines]

    def __len__(self):
        return len(self.indices)

    def get_image(self, idx):
        assert idx in self.indices
        img_filename = os.path.join(self.image_dir, '%06d.jpg' % (idx))
        return cv2.imread(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        assert idx in self.indices

        # Get camera world position from calib file
        calib = self.get_calibration(idx)
        position = calib.get_world_camera_position()
        frame_metadata = {
            # 'timestamp': 8888,
            'position': position,
            # 'rotation': rotation,
        }

        gps_position = [int(x) for x in frame_metadata['position']]
        # TODO: move to function
        las_data = {}
        filenames_las = utils.extract_lidar_filenames_from_metadata(gps_position, perimeter=60)
        for filename_las, path_las in filenames_las.items():
            las_reader = laspy.open(path_las, mode='r')
            if filename_las not in las_data:
                las_data[filename_las] = las_reader.read()

        return las_data

    def get_calibration(self, idx):
        assert idx in self.indices
        calib_filename = os.path.join(self.calib_dir, '%06d.txt' % (idx))
        return kitti_utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        assert idx in self.indices
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        lines = [line.rstrip() for line in open(label_filename)]
        return self.read_label(lines)

    def get_pred_objects(self, idx):
        assert idx in self.indices
        pred_filename = os.path.join(self.pred_dir, self.file_pred_pkl)
        with open(pred_filename, 'rb') as f:
            preds = pickle.load(f)
            pred = []
            for p in preds:
                if len(p['sample_idx']) != 0 and p['sample_idx'][0] == idx:
                    pred.append(p)
            # pred = list(filter(lambda p: p['sample_idx'].size != 0 and p['sample_idx'][0] == idx, preds))
            # Check if there is a prediction
            if len(pred) != 0:
                # Get corresponding prediction (only 1 per idx)
                assert len(pred) == 1
                pred = pred[0]
                p = [dict(zip(pred, t)) for t in zip(*pred.values())]
                return self.read_label(p)
            else:
                return None

    def get_flags_objects(self, idx):
        assert idx in self.indices
        flags_filename = os.path.join(self.flags_dir, '%06d.txt' % (idx))
        objects_flags = [tuple(line.strip().split(':')) for line in open(flags_filename)]
        object_ids_gt = []
        object_ids_removed = []
        lines_obj_removed = []

        for object_id, object_label_str in objects_flags:
            if len(object_label_str) == 0:
                # Object is not removed, so only extract id
                object_ids_gt.append(object_id)
            else:
                # Object was removed, so also extract KITTI information
                object_ids_removed.append(object_id)
                lines_obj_removed.append(object_label_str)

        objects_removed = self.read_label(lines_obj_removed, object_ids_removed)
        return object_ids_gt, objects_removed

    def read_label(self, lines, object_ids=None):
        objects = []
        for idx, line in enumerate(lines):
            if object_ids is not None:
                # Removed objs
                object_kitti = kitti_utils.Object3D(line, object_ids[idx], removed=True)
            else:
                # Ground truths & predictions
                object_kitti = kitti_utils.Object3D(line)

            objects.append(object_kitti)

        return objects

    def isexist_label_objects(self, idx):
        assert idx in self.indices
        label_filename = os.path.join(self.label_dir, '%06d.txt' % (idx))
        return os.path.exists(label_filename)

    def isexist_pred_objects(self, idx):
        assert idx in self.indices
        # Predictions from MMDetection3D are saved as pickle file
        if self.file_pred_pkl is not None:
            pred_filename_pkl = os.path.join(self.pred_dir, self.file_pred_pkl)
            return os.path.exists(pred_filename_pkl)
        # Predictions from original SMOKE are saved as text file
        else:
            pred_filename = os.path.join(self.pred_dir, '%06d.txt' % (idx))
            return os.path.exists(pred_filename)

    def isexist_flags_objects(self, idx):
        assert idx in self.indices
        flag_filename = os.path.join(self.flags_dir, '%06d.txt' % (idx))
        return os.path.exists(flag_filename)

    def get_obj_colors(self, obj):
        if not obj.removed:
            # Spaces are used as separators in KITTI format, so pretty print
            c = list(filter(lambda c: c['name'] == obj.type, self.config.values()))[0]
            return c['color'], c['color_text']
        else:
            # TODO: move to config
            return (75, 75, 75), (255, 255, 255)

    def count_objs(self, idx, objs_gt, objs_pred, objs_removed):
        print(f'======== #{idx}: Objects in Ground Truth ========')
        n_obj_gt = 0
        for obj in objs_gt:
            if obj.type != 'DontCare':
                print(f'=== {n_obj_gt + 1} ground truth object(s) ===')
                obj.print_object()
                n_obj_gt += 1
        print()

        if objs_pred is not None:
            n_obj_pred = 0
            for obj in objs_pred:
                if obj.type != 'DontCare':
                    print(f'=== {n_obj_gt + 1} predicted object(s) ===')
                    n_obj_pred += 1
        else:
            n_obj_pred = -1

        if objs_removed is not None:
            n_obj_removed = 0
            for obj in objs_removed:
                if obj.type != 'DontCare':
                    n_obj_removed += 1
        else:
            n_obj_removed = -1

        return n_obj_gt, n_obj_pred, n_obj_removed


def show_image_with_boxes(img, objects, calib, dataset):
    '''
    Show image with 2D & 3D bounding boxes
    '''
    img1 = np.copy(img)  # for drawing 2d bbox
    img2 = np.copy(img)  # for drawing 3d bbox
    img_height, img_width, _ = img.shape

    if objects is None:
        text = 'no prediction/visual'
        (label_width, label_height), baseline = cv2.getTextSize(text, FONT, fontScale=3, thickness=2)
        pos = (img_width - label_width, img_height - 30)
        # img_center = int(img_height * 0.5), int(img_width * 0.5)
        img1 = draw_text(img1, text, pos, with_background=True,
                         text_color=(255, 255, 255), text_color_bg=(0, 0, 0))
        img2 = draw_text(img2, text, pos, with_background=True,
                         text_color=(255, 255, 255), text_color_bg=(0, 0, 0))
        return img1, img2

    for obj in objects:
        obj_id = f'{obj.id}-' if obj.id is not None else ''
        color, color_text = dataset.get_obj_colors(obj)
        color = np.flip(color).tolist()  # RGB -> BGR
        name = obj.type

        img1 = cv2.rectangle(img1,
                             (int(obj.xmin), int(obj.ymin)),
                             (int(obj.xmax), int(obj.ymax)),
                             color, 5)
        img1 = draw_text(img1, f'{obj_id}{name}', (int(obj.xmin) - 2, int(obj.ymin) - 25),
                         with_background=True, text_color=color_text, text_color_bg=color)

        box3d_pts_2d, _ = kitti_utils.compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print('something wrong in the 3D box.')
            continue

        img2 = kitti_utils.draw_projected_box3d(img2, box3d_pts_2d, color)
        img2 = draw_text(img2, f'{obj_id}{name}', (int(obj.xmin) - 2, int(obj.ymin) - 25),
                         with_background=True, text_color=color_text, text_color_bg=color)

    return img1, img2


def draw_text(img, text,
              pos=(0, 0),
              with_background=True,
              text_color=(255, 255, 255),
              text_color_bg=(0, 0, 0),
              font_scale=3,
              font_thickness=2):
    x, y = pos
    text_size, _ = cv2.getTextSize(text, FONT, font_scale, font_thickness)
    text_w, text_h = text_size

    if with_background:
        cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
    cv2.putText(img, text, (x, y + text_h + font_scale - 1), FONT, font_scale, text_color, font_thickness)

    return img


def show_lidar_topview_with_boxes(
        idx,
        scans,
        calib,
        dataset,
        objects_gt,
        objects_pred=None,
        interactive=False
):
    '''
    top_view image
    '''
    # TODO: get image resolution from config
    w, h = 2016, 2016
    image_top = np.zeros(shape=(w, h, 3), dtype=np.float32)
    image_top.fill(255)

    # 3D - draw camera, objects, points, etc.
    camera_intrinsic = calib.P[:, :-1]
    camera_extrinsic = np.vstack((calib.V2C, [[0, 0, 0, 1]]))
    transformation_matrix_c2v = np.asmatrix(np.vstack((calib.C2V, [[0, 0, 0, 1]])))
    camera_origin = calib.get_world_camera_position()
    plane_point_z = transformation_matrix_c2v * np.asarray([0, 0, 10, 1]).reshape(4, 1)
    camera_lines = open3d.geometry.LineSet.create_camera_visualization(
        w, h, camera_intrinsic, camera_extrinsic, scale=1
    )
    # From https://github.com/opencv/opencv/blob/2.4/modules/calib3d/src/calibration.cpp
    # in degrees
    fovx = 2 * math.atan(w / (2 * calib.f_u)) * 180 / math.pi
    fovy = 2 * math.atan(h / (2 * calib.f_v)) * 180 / math.pi
    camera_lines.paint_uniform_color(np.asarray((0, 0, 0)))

    points = []
    for tile_name, tile in scans.items():
        # Extract points that fall in image frame
        pts = np.column_stack((tile.x, tile.y, tile.z))
        xs = pts[:, 0]
        ys = pts[:, 1]
        pts_filtered = pts[(camera_origin[0] - 25 <= xs)
                           & (xs < camera_origin[0] + 25)
                           & (camera_origin[1] - 25 <= ys)
                           & (ys < camera_origin[1] + 25)]

        points.extend(pts_filtered.tolist())

    def bbox3d(obj):
        _, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    # Ground truth
    boxes3d_gt = [bbox3d(obj) for obj in objects_gt if obj.type != 'DontCare']
    labels_boxes3d_gt = [(obj.type, *dataset.get_obj_colors(obj)) for obj in objects_gt if obj.type != 'DontCare']

    # Prediction
    if objects_pred is not None:
        boxes3d_pred = [bbox3d(obj) for obj in objects_pred if obj.type != 'DontCare']
        labels_boxes3d_pred = [(obj.type, *dataset.get_obj_colors(obj)) for obj in objects_pred if
                               obj.type != 'DontCare']

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    pcds_boxes_gt = []
    pcds_boxes_pred = []
    local_axes_gt = []
    local_axes_pred = []
    # Manually set for now, so this could be improved
    edges = [[0, 1], [0, 3], [0, 4], [1, 2],
             [1, 5], [2, 3], [2, 6], [3, 7],
             [4, 5], [5, 6], [4, 7], [6, 7]]
    for idx, pts in enumerate(boxes3d_gt):
        _, color, _ = labels_boxes3d_gt[idx]
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(pts)
        line_set.lines = open3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color(np.asarray(color) / 255)

        pcds_boxes_gt.append(line_set)

        object_gt = objects_gt[idx]
        tvec = calib.project_rect_to_velo(np.asarray([object_gt.t])).reshape(-1)
        angle_with_camera = training_data.angle_between(object_gt.t, [1, 0, 0])
        # Rotated towards camera when alpha = 0
        angle_observation = -1 * math.degrees(angle_with_camera) + math.degrees(object_gt.alpha) - 180
        rvec = calib.get_world_camera_rotation() * utils.euler_to_rotation_matrix(0, angle_observation, 0)
        local_axis = open3d.geometry.TriangleMesh.create_coordinate_frame()
        local_axis.translate(tvec)
        # Rotate using alpha (observation angle)
        local_axis.rotate(R=rvec, center=tvec)
        local_axes_gt.append(local_axis)

    if objects_pred is not None:
        for idx, pts in enumerate(boxes3d_pred):
            _, color, _ = labels_boxes3d_pred[idx]
            line_set = open3d.geometry.LineSet()
            line_set.points = open3d.utility.Vector3dVector(pts)
            line_set.lines = open3d.utility.Vector2iVector(edges)
            line_set.paint_uniform_color(color)

            pcds_boxes_pred.append(line_set)

    if interactive:
        geometries = [pcd] + pcds_boxes_gt + pcds_boxes_pred + [camera_lines] + local_axes_gt + local_axes_pred
        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window(window_name='Open3D', width=1000, height=1000)
        for idx, geometry in enumerate(geometries):
            visualizer.add_geometry(geometry)

        view_control = visualizer.get_view_control()
        # Orthographic projection
        view_control.change_field_of_view(step=-60)
        view_control.set_zoom(0.6)
        options = visualizer.get_render_option()
        options.point_size = 1

        visualizer.run()
    else:
        renderer = open3d.visualization.rendering.OffscreenRenderer(2016, 2016)
        renderer.scene.set_background([255, 255, 255, 255])  # RGBA

        for idx, geometry in enumerate(pcds_boxes_gt + [camera_lines]):
            # From https://github.com/isl-org/Open3D/pull/3194
            # & https://github.com/isl-org/Open3D/issues/4417
            mat = open3d.visualization.rendering.MaterialRecord()
            mat.shader = 'unlitLine'
            mat.line_width = 0.075

            renderer.scene.add_geometry(f'rest-{idx}', geometry, mat)

        for idx, geometry in enumerate(pcds_boxes_pred):
            # From https://github.com/isl-org/Open3D/pull/3194
            # & https://github.com/isl-org/Open3D/issues/4417
            mat = open3d.visualization.rendering.MaterialRecord()
            mat.shader = 'unlitLine'
            mat.line_width = 0.15

            renderer.scene.add_geometry(f'pred-{idx}', geometry, mat)

        for idx, geometry in enumerate(local_axes_gt + local_axes_pred):
            mat = open3d.visualization.rendering.MaterialRecord()
            mat.shader = 'unlitLine'
            mat.line_width = 0.15
            renderer.scene.add_geometry(f'axis-{idx}', geometry, mat)

        # From https://github.com/isl-org/Open3D/issues/2135
        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.base_color = [0.0, 0.0, 0.0, 1.0]
        mat.point_size = 2
        renderer.scene.add_geometry('pcd', pcd, mat)

        px, py, pz, _ = np.ravel(plane_point_z)
        renderer.setup_camera(1, [px, py, pz], [px, py, pz + 2000], [0, 1, 0])
        # From https://github.com/isl-org/Open3D/issues/2946
        # Disable post processing
        renderer.scene.view.set_post_processing(False)

        img = renderer.render_to_image()
        image_top = cv2.rotate(np.array(img), cv2.ROTATE_180)

        image_top = kitti_utils.draw_text_box3d(image_top, labels_boxes3d_gt, is_gt=True)
        if objects_pred is not None:
            image_top = kitti_utils.draw_text_box3d(image_top, labels_boxes3d_pred, is_gt=False)

        image_top = cv2.cvtColor(image_top, cv2.COLOR_RGBA2BGRA)

        # camera look at offset point
        cv2.circle(image_top, (int(w * 0.5), int(h * 0.5)), radius=10, color=(0, 0, 0), thickness=-1)

    return image_top

def show_lidar_topview(
        idx,
        scans,
        calib,
        dataset,
        objects_gt,
        objects_pred=None,
        interactive=False
):
    '''
    top_view image
    '''
    # TODO: get image resolution from config
    w, h = 2016, 2016
    image_top = np.zeros(shape=(w, h, 3), dtype=np.float32)
    image_top.fill(255)

    # 3D - draw camera, objects, points, etc.
    camera_intrinsic = calib.P[:, :-1]
    camera_extrinsic = np.vstack((calib.V2C, [[0, 0, 0, 1]]))
    transformation_matrix_c2v = np.asmatrix(np.vstack((calib.C2V, [[0, 0, 0, 1]])))
    camera_origin = calib.get_world_camera_position()
    plane_point_z = transformation_matrix_c2v * np.asarray([0, 0, 10, 1]).reshape(4, 1)
    camera_lines = open3d.geometry.LineSet.create_camera_visualization(
        w, h, camera_intrinsic, camera_extrinsic, scale=1
    )
    # From https://github.com/opencv/opencv/blob/2.4/modules/calib3d/src/calibration.cpp
    # in degrees
    fovx = 2 * math.atan(w / (2 * calib.f_u)) * 180 / math.pi
    fovy = 2 * math.atan(h / (2 * calib.f_v)) * 180 / math.pi
    camera_lines.paint_uniform_color(np.asarray((0, 0, 0)))

    points = []
    for tile_name, tile in scans.items():
        # Extract points that fall in image frame
        pts = np.column_stack((tile.x, tile.y, tile.z))
        xs = pts[:, 0]
        ys = pts[:, 1]
        pts_filtered = pts[(camera_origin[0] - 25 <= xs)
                           & (xs < camera_origin[0] + 25)
                           & (camera_origin[1] - 25 <= ys)
                           & (ys < camera_origin[1] + 25)]

        points.extend(pts_filtered.tolist())

    def bbox3d(obj):
        _, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    # Ground truth
    boxes3d_gt = [bbox3d(obj) for obj in objects_gt if obj.type != 'DontCare']
    labels_boxes3d_gt = [(obj.type, *dataset.get_obj_colors(obj)) for obj in objects_gt if obj.type != 'DontCare']

    # Prediction
    if objects_pred is not None:
        boxes3d_pred = [bbox3d(obj) for obj in objects_pred if obj.type != 'DontCare']
        labels_boxes3d_pred = [(obj.type, *dataset.get_obj_colors(obj)) for obj in objects_pred if
                               obj.type != 'DontCare']

    pcd = open3d.geometry.PointCloud()
    pcd.points = open3d.utility.Vector3dVector(points)

    pcds_boxes_gt = []
    pcds_boxes_pred = []
    local_axes_gt = []
    local_axes_pred = []
    # Manually set for now, so this could be improved
    edges = [[0, 1], [0, 3], [0, 4], [1, 2],
             [1, 5], [2, 3], [2, 6], [3, 7],
             [4, 5], [5, 6], [4, 7], [6, 7]]
    for idx, pts in enumerate(boxes3d_gt):
        _, color, _ = labels_boxes3d_gt[idx]
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(pts)
        line_set.lines = open3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color(np.asarray(color) / 255)

        pcds_boxes_gt.append(line_set)

        object_gt = objects_gt[idx]
        tvec = calib.project_rect_to_velo(np.asarray([object_gt.t])).reshape(-1)
        angle_with_camera = training_data.angle_between(object_gt.t, [1, 0, 0])
        # Rotated towards camera when alpha = 0
        angle_observation = -1 * math.degrees(angle_with_camera) + math.degrees(object_gt.alpha) - 180
        rvec = calib.get_world_camera_rotation() * utils.euler_to_rotation_matrix(0, angle_observation, 0)
        local_axis = open3d.geometry.TriangleMesh.create_coordinate_frame()
        local_axis.translate(tvec)
        # Rotate using alpha (observation angle)
        local_axis.rotate(R=rvec, center=tvec)
        local_axes_gt.append(local_axis)

    if interactive:
        geometries = [pcd] + pcds_boxes_gt + pcds_boxes_pred + [camera_lines] + local_axes_gt + local_axes_pred
        visualizer = open3d.visualization.Visualizer()
        visualizer.create_window(window_name='Open3D', width=1000, height=1000)
        for idx, geometry in enumerate(geometries):
            visualizer.add_geometry(geometry)

        view_control = visualizer.get_view_control()
        # Orthographic projection
        view_control.change_field_of_view(step=-60)
        view_control.set_zoom(0.6)
        options = visualizer.get_render_option()
        options.point_size = 1

        visualizer.run()
    else:
        renderer = open3d.visualization.rendering.OffscreenRenderer(2016, 2016)
        renderer.scene.set_background([255, 255, 255, 255])  # RGBA

        for idx, geometry in enumerate(local_axes_gt + local_axes_pred):
            mat = open3d.visualization.rendering.MaterialRecord()
            mat.shader = 'unlitLine'
            mat.line_width = 0.15
            renderer.scene.add_geometry(f'axis-{idx}', geometry, mat)

        # From https://github.com/isl-org/Open3D/issues/2135
        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'defaultUnlit'
        mat.base_color = [0.0, 0.0, 0.0, 1.0]
        mat.point_size = 2
        renderer.scene.add_geometry('pcd', pcd, mat)

        px, py, pz, _ = np.ravel(plane_point_z)
        renderer.setup_camera(1, [px, py, pz], [px, py, pz + 2000], [0, 1, 0])

    return renderer

def draw_boxes_on_topview(
        idx,
        renderer,
        calib,
        dataset,
        objects_gt,
        num_loop,
        num_former_obj_gt,
        num_former_obj_pred,
        objects_pred=None
):
    '''
    top_view image
    '''
    # TODO: get image resolution from config
    w, h = 2016, 2016
    image_top = np.zeros(shape=(w, h, 3), dtype=np.float32)
    image_top.fill(255)

    # 3D - draw camera, objects, points, etc.
    camera_intrinsic = calib.P[:, :-1]
    camera_extrinsic = np.vstack((calib.V2C, [[0, 0, 0, 1]]))
    transformation_matrix_c2v = np.asmatrix(np.vstack((calib.C2V, [[0, 0, 0, 1]])))
    camera_origin = calib.get_world_camera_position()
    plane_point_z = transformation_matrix_c2v * np.asarray([0, 0, 10, 1]).reshape(4, 1)
    camera_lines = open3d.geometry.LineSet.create_camera_visualization(
        w, h, camera_intrinsic, camera_extrinsic, scale=1
    )
    # From https://github.com/opencv/opencv/blob/2.4/modules/calib3d/src/calibration.cpp
    # in degrees
    fovx = 2 * math.atan(w / (2 * calib.f_u)) * 180 / math.pi
    fovy = 2 * math.atan(h / (2 * calib.f_v)) * 180 / math.pi
    camera_lines.paint_uniform_color(np.asarray((0, 0, 0)))

    def bbox3d(obj):
        _, box3d_pts_3d = kitti_utils.compute_box_3d(obj, calib.P)
        box3d_pts_3d_velo = calib.project_rect_to_velo(box3d_pts_3d)
        return box3d_pts_3d_velo

    # Ground truth
    boxes3d_gt = [bbox3d(obj) for obj in objects_gt if obj.type != 'DontCare']
    labels_boxes3d_gt = [(obj.type, *dataset.get_obj_colors(obj)) for obj in objects_gt if obj.type != 'DontCare']

    # Prediction
    if objects_pred is not None:
        boxes3d_pred = [bbox3d(obj) for obj in objects_pred if obj.type != 'DontCare']
        labels_boxes3d_pred = [(obj.type, *dataset.get_obj_colors(obj)) for obj in objects_pred if
                               obj.type != 'DontCare']

    pcds_boxes_gt = []
    pcds_boxes_pred = []
    local_axes_gt = []
    local_axes_pred = []
    # Manually set for now, so this could be improved
    edges = [[0, 1], [0, 3], [0, 4], [1, 2],
             [1, 5], [2, 3], [2, 6], [3, 7],
             [4, 5], [5, 6], [4, 7], [6, 7]]
    for idx, pts in enumerate(boxes3d_gt):
        _, color, _ = labels_boxes3d_gt[idx]
        line_set = open3d.geometry.LineSet()
        line_set.points = open3d.utility.Vector3dVector(pts)
        line_set.lines = open3d.utility.Vector2iVector(edges)
        line_set.paint_uniform_color(np.asarray(color) / 255)

        pcds_boxes_gt.append(line_set)

        object_gt = objects_gt[idx]
        tvec = calib.project_rect_to_velo(np.asarray([object_gt.t])).reshape(-1)
        angle_with_camera = training_data.angle_between(object_gt.t, [1, 0, 0])
        # Rotated towards camera when alpha = 0
        angle_observation = -1 * math.degrees(angle_with_camera) + math.degrees(object_gt.alpha) - 180
        rvec = calib.get_world_camera_rotation() * utils.euler_to_rotation_matrix(0, angle_observation, 0)
        local_axis = open3d.geometry.TriangleMesh.create_coordinate_frame()
        local_axis.translate(tvec)
        # Rotate using alpha (observation angle)
        local_axis.rotate(R=rvec, center=tvec)
        local_axes_gt.append(local_axis)

    if objects_pred is not None:
        for idx, pts in enumerate(boxes3d_pred):
            _, color, _ = labels_boxes3d_pred[idx]
            line_set = open3d.geometry.LineSet()
            line_set.points = open3d.utility.Vector3dVector(pts)
            line_set.lines = open3d.utility.Vector2iVector(edges)
            line_set.paint_uniform_color(color)

            pcds_boxes_pred.append(line_set)

    for idx, geometry in enumerate(pcds_boxes_gt + [camera_lines]):
        # From https://github.com/isl-org/Open3D/pull/3194
        # & https://github.com/isl-org/Open3D/issues/4417
        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'unlitLine'
        mat.line_width = 0.075

        renderer.scene.add_geometry(f'rest-{idx+num_former_obj_gt}', geometry, mat)

    for idx, geometry in enumerate(pcds_boxes_pred):
        # From https://github.com/isl-org/Open3D/pull/3194
        # & https://github.com/isl-org/Open3D/issues/4417
        mat = open3d.visualization.rendering.MaterialRecord()
        mat.shader = 'unlitLine'
        mat.line_width = 0.15

        renderer.scene.add_geometry(f'pred-{idx+num_former_obj_pred}', geometry, mat)

    px, py, pz, _ = np.ravel(plane_point_z)
    # renderer.setup_camera(1, [px, py, pz], [px, py, pz + 2000], [0, 1, 0])
    # From https://github.com/isl-org/Open3D/issues/2946
    # Disable post processing
    renderer.scene.view.set_post_processing(False)

    # image_top = kitti_utils.draw_text_box3d(image_top, labels_boxes3d_gt, is_gt=True)
    # if objects_pred is not None:
    #     image_top = kitti_utils.draw_text_box3d(image_top, labels_boxes3d_pred, is_gt=False)
    #
    # image_top = cv2.cvtColor(image_top, cv2.COLOR_RGBA2BGRA)

    return renderer

def prepare_directory(output_path, split):
    dir_2dbbox = f'{output_path}visualization/{split}/2d-bbox/'
    dir_3dbbox = f'{output_path}visualization/{split}/3d-bbox/'
    dir_bev = f'{output_path}visualization/{split}/bev/'

    for dir in [dir_2dbbox, dir_3dbbox, dir_bev]:
        if not os.path.exists(dir):
            os.makedirs(dir)


def run(data_path, split, predictions_file=None, save_images=False, draw_bev=True, output_path=''):
    dataset = ObjectData(data_path,
                         config.CONFIG_BBOXES,
                         split,
                         file_pred_pkl=predictions_file,
                         imageset_outside=True)
    prepare_directory(output_path, split)

    with open(f'{data_path}flags/fugro-to-kitti-ids.txt', 'r') as f:
        kitti_fugro_ids = [tuple(line.strip().split(' ')) for line in f]

    # use the last frame as background and project all bboxes on it
    data_idx_bg = dataset.indices[15]
    calib = dataset.get_calibration(data_idx_bg)
    scans = dataset.get_lidar(data_idx_bg)

    objects_gt = []
    objects_pred = []
    if dataset.isexist_label_objects(data_idx_bg):
        objects_gt = dataset.get_label_objects(data_idx_bg)
    else:
        objects_gt = None

    if dataset.isexist_pred_objects(data_idx_bg):
        objects_pred = dataset.get_pred_objects(data_idx_bg)
    else:
        # Set to None because '[]' indicates there should
        # be predictions in a file but they are not present
        objects_pred = None

    renderer_bev = show_lidar_topview(
        data_idx_bg, scans, calib, dataset, objects_gt, objects_pred, interactive=False)

    num_gt_obj_in_frame = []
    num_pred_obj_in_frame = []
    for data_idx in dataset.indices:
        objects_gt = []
        objects_pred = []
        if dataset.isexist_label_objects(data_idx):
            objects_gt = dataset.get_label_objects(data_idx)
        else:
            continue

        if dataset.isexist_pred_objects(data_idx):
            objects_pred = dataset.get_pred_objects(data_idx)
        else:
            # Set to None because '[]' indicates there should
            # be predictions in a file but they are not present
            objects_pred = None

        if dataset.isexist_flags_objects(data_idx):
            object_ids_gt, objects_removed = dataset.get_flags_objects(data_idx)
            for idx, obj in enumerate(objects_gt):
                obj.id = object_ids_gt[idx]
        else:
            object_ids_gt = None
            objects_removed = None

        calib = dataset.get_calibration(data_idx)
        img = dataset.get_image(data_idx)
        # img = dataset.get_image(data_idx_bg)
        img_height, img_width, _ = img.shape

        n_obj_gt, n_obj_pred, n_obj_removed = dataset.count_objs(
            data_idx, objects_gt, objects_pred, objects_removed
        )

        # Only create visualization if there are objects to visualize
        # if n_obj_gt > 0 or n_obj_pred > 0 or n_obj_removed > 0:
        filename_kitti = str(data_idx).zfill(6)
        run_name, frame_id = [(item[1], item[2]) for item in kitti_fugro_ids
                              if item[0] == filename_kitti][0]

        # Draw 2d and 3d boxes on image
        img_gt_2dbox, img_gt_3dbox = show_image_with_boxes(
            img, objects_gt, calib, dataset
        ) if objects_removed is None else show_image_with_boxes(
            # Draw removed objects first
            img, objects_removed + objects_gt, calib, dataset
        )
        img_pred_2dbox, img_pred_3dbox = show_image_with_boxes(img, objects_pred, calib, dataset)

        img_2dbbox = np.concatenate((img_gt_2dbox, img_pred_2dbox), axis=1)
        img_3dbbox = np.concatenate((img_gt_3dbox, img_pred_3dbox), axis=1)

        img_2dbbox = draw_text(img_2dbbox, 'ground truth', (0, 0))
        img_2dbbox = draw_text(img_2dbbox, 'prediction', (img_width, 0))
        img_2dbbox = draw_text(img_2dbbox, f'fugro run: {run_name}, id: {frame_id}', (0, img_height - 30))

        img_3dbbox = draw_text(img_3dbbox, 'ground truth', (0, 0))
        img_3dbbox = draw_text(img_3dbbox, 'prediction', (img_width, 0))
        img_3dbbox = draw_text(img_3dbbox, f'fugro run: {run_name}, id: {frame_id}', (0, img_height - 30))

        if save_images:
            # cv2.imwrite(f'{output_path}visualization/{split}/2d-bbox/2d-bbox-{filename_kitti}.jpg', img_2dbbox)
            cv2.imwrite(f'{output_path}visualization/{split}/3d-bbox/3d-bbox-{filename_kitti}.jpg', img_3dbbox)

        # loop number
        num_loop = data_idx - dataset.indices[0]

        # BEV view of 3d boxes on LiDAR
        # Takes longer to draw
        if draw_bev:
            scans = dataset.get_lidar(data_idx)
            print('#scans:', len(scans))

            # img_bev = show_lidar_topview_with_boxes(
            #     data_idx, scans, calib, dataset, objects_gt, objects_pred, interactive=False
            # ) if objects_removed is None else show_lidar_topview_with_boxes(
            #     data_idx, scans, calib, dataset, objects_removed + objects_gt, objects_pred, interactive=False
            # )
            # img_bev = draw_text(img_bev, f'fugro run: {run_name}, id: {frame_id}', (0, img_height - 30))
            #
            # if save_images:
            #     cv2.imwrite(f'{output_path}visualization/{split}/bev/bev-{filename_kitti}.jpg', img_bev)

            # if want to get a bev with all predicted boxes with camera trajectory, use the code below
            renderer_bev = draw_boxes_on_topview(
                data_idx, renderer_bev, calib, dataset, objects_gt,
                num_loop, sum(num_gt_obj_in_frame), sum(num_pred_obj_in_frame), objects_pred)

            # Disable post processing
            renderer_bev.scene.view.set_post_processing(False)
            img = renderer_bev.render_to_image()
            img_bev_bg = cv2.rotate(np.array(img), cv2.ROTATE_180)
            img_bev_bg = cv2.cvtColor(img_bev_bg, cv2.COLOR_RGBA2BGRA)

            cv2.imwrite(f'{output_path}visualization/{split}/bev/bev-{num_loop+1}.jpg', img_bev_bg)

        num_gt_obj_in_frame.append(n_obj_gt)
        num_pred_obj_in_frame.append(n_obj_pred)




def visualize_bbox(pcd_cropped,
                   annotation,
                   bbox_lineset,
                   axis,
                   angle_line,
                   obj_name,
                   is_sfm=False):
    # From http://www.open3d.org/docs/release/python_example/visualization/index.html
    renderer = open3d.visualization.rendering.OffscreenRenderer(620, 1024)
    renderer.scene.set_background([0, 0, 0, 0])  # RGBA
    renderer.scene.view.set_post_processing(False)

    # Add point cloud
    mat = open3d.visualization.rendering.MaterialRecord()
    mat.shader = 'defaultUnlit'
    if not is_sfm:
        mat.base_color = [255, 255, 255, 1.0]
    mat.point_size = 1 if is_sfm else 3
    renderer.scene.add_geometry('PointCloud', pcd_cropped, mat)

    # Add annotation
    mat = open3d.visualization.rendering.MaterialRecord()
    mat.shader = 'unlitLine'
    mat.line_width = 7
    mat.thickness = 4
    renderer.scene.add_geometry('Annotation', annotation, mat)

    bounds = renderer.scene.bounding_box

    if obj_name == 'Marker':
        # FOV, center, eye, up
        renderer.setup_camera(60, bounds.get_center(), bounds.get_center() + [1, 1, 2], [0, 0, 1])
    elif obj_name == 'Cabinet':
        renderer.setup_camera(60, bounds.get_center(), bounds.get_center() + [1, 1, 2], [0, 0, 1])
    else:
        renderer.setup_camera(60, bounds.get_center(), bounds.get_center() + [3, 3, 4], [0, 0, 1])

    img = renderer.render_to_image()
    return np.array(img)


def crop_pcd_bboxes(dwg,
                    pcd_type='lidar',
                    save_images=False,
                    output_path='',
                    sfm_filepath=None):
    msp = dwg.modelspace()
    bboxes = msp.query('MESH[layer!="0"]')

    print('layers in file:')
    print('\n'.join([f'{layer.dxf.name}' for layer in dwg.layers]))
    print()
    if pcd_type == 'sfm':
        print('SfM point clouds can be very dense, can take a bit longer to run')

    for idx, bbox in enumerate(bboxes):
        layer_name = bbox.dxf.layer
        layer_obj = dwg.layers.get(layer_name)
        obj_color = ezdxf.colors.aci2rgb(layer_obj.color)
        c = config.CONFIG_BBOXES[layer_name]
        obj_name = c['name']
        vs = list(bbox.vertices)
        gps_position = [int(x) for x in vs[0]]

        if pcd_type == 'sfm':
            # Create Open3D point cloud from PLY data
            pcd = open3d.io.read_point_cloud(sfm_filepath)
            print(pcd)

            # Skip if there are no point clouds to display for this bbox
            if len(pcd.points) == 0:
                continue
        elif pcd_type == 'depth-estimation':
            pass
        else:
            # TODO: move to function
            pcd_data = {}
            filenames_las = utils.extract_lidar_filenames_from_metadata(gps_position, perimeter=60)
            for filename_las, path_las in filenames_las.items():
                las_reader = laspy.open(path_las, mode='r')
                if filename_las not in pcd_data:
                    pcd_data[filename_las] = las_reader.read()

            if len(pcd_data) == 0:
                continue

            # Create Open3D point cloud
            pcd = point_cloud_data.convert_lidars_to_open3d(pcd_data.values())

        cube_corner_bottom = vs[0]
        # Create Open3D geometries
        vs = np.asarray(vs)
        tvec = np.asarray(cube_corner_bottom) * -1

        # Open3D does not properly generate a oriented bbox.
        # This can be inconsistent: https://github.com/isl-org/Open3D/issues/4716
        # obbox = open3d.geometry.OrientedBoundingBox.create_from_points(ps, robust=True)
        # So manually compute bbox properties for now
        c = training_data.get_cube_center(vs) + tvec
        h, w, l = training_data.get_cube_dim(vs)
        edge_vec = training_data.get_cube_edge_vector(vs, cube_corner_bottom)
        angle = training_data.angle_between(edge_vec, [1, 0, 0])
        angle_deg = -1 * math.degrees(angle)
        # Add 45 degrees because we took the square diagonal
        rvec = np.float64(utils.euler_to_rotation_matrix(0, 0, angle_deg - 45))
        # Add some height to bbox, otherwise bbox might cut off object
        obbox = open3d.geometry.OrientedBoundingBox(center=c, R=rvec, extent=[w, l, h + 2])
        obbox.scale(1.2, obbox.center)

        obbox_lineset = open3d.geometry.LineSet.create_from_oriented_bounding_box(obbox)
        obbox_lineset.paint_uniform_color([1, 1, 1])
        # Create Open3D lineset from bottom square of bbox
        pts_annotation = list(filter(lambda v: v[2] == cube_corner_bottom[2], vs))
        lines = [[0, 1], [1, 2], [2, 3], [3, 0]]
        annotation = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(pts_annotation),
            lines=open3d.utility.Vector2iVector(lines),
        )
        annotation.paint_uniform_color(np.asarray(obj_color) / 255)

        axis = open3d.geometry.TriangleMesh.create_coordinate_frame()
        axis.scale(2, center=axis.get_center())
        angle_line = open3d.geometry.LineSet(
            points=open3d.utility.Vector3dVector(np.asarray([[0, 0, 0], edge_vec])),
            lines=open3d.utility.Vector2iVector(np.asarray([[0, 1]])),
        )
        angle_line.paint_uniform_color((0, 1, 0))

        pcd.translate(tvec)
        pcd_cropped = pcd.crop(obbox)
        if len(pcd_cropped.points) == 0:
            continue

        # Translate and reset rotation for all boxes to (0, 0, 0) origin for easier visualization
        annotation.translate(tvec)
        rvec_inv = open3d.core.Tensor.from_numpy(obbox.R).inv().numpy()
        obbox_lineset.rotate(rvec_inv, center=(0, 0, 0))
        pcd_cropped.rotate(rvec_inv, center=(0, 0, 0))
        annotation.rotate(rvec_inv, center=(0, 0, 0))
        angle_line.rotate(rvec_inv, center=(0, 0, 0))
        image_cropped_bbox = visualize_bbox(
            pcd_cropped, annotation, obbox_lineset, axis, angle_line, obj_name, pcd_type == 'sfm'
        )

        if save_images:
            print(f'save image: {idx} - GPS {gps_position}')
            img = cv2.cvtColor(image_cropped_bbox, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f'{output_path}cropped-{obj_name}-{idx}.jpg', img)
        else:
            print('show image')
            _, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(image_cropped_bbox)
            ax.axis('off')
            plt.show()