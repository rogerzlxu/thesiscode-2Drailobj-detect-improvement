import sys
import math
import cv2
import numpy as np
import ezdxf
import matplotlib.pyplot as plt

from config import config
from scripts import camera_data, utils, training_data

def draw_bbox_3d(img, img_pts, color, thickness):
    img_pts = np.int32(img_pts).reshape(8, 2)
    img = cv2.drawContours(img, [img_pts[:4]], -1, color, thickness)

    for i, j in zip(range(4), range(4, 8)):
        img = cv2.line(img, tuple(img_pts[i]), tuple(img_pts[j]), color, thickness)

    img = cv2.drawContours(img, [img_pts[4:]], -1, color, thickness)

    return img

def equation_plane(point1, point2, point3):
    x1, y1, z1, _ = point1
    x2, y2, z2, _ = point2  # Ignore w-component
    x3, y3, z3, _ = point3
    a1 = x2 - x1
    b1 = y2 - y1
    c1 = z2 - z1
    a2 = x3 - x1
    b2 = y3 - y1
    c2 = z3 - z1
    a = b1 * c2 - b2 * c1
    b = a2 * c1 - a1 * c2
    c = a1 * b2 - b1 * a2
    d = (- a * x1 - b * y1 - c * z1)
    return a, b, c, d

def clip_points(points, plane, flip):
    a, b, c, d = plane
    clipped_points = []

    for point in points:
        x, y, z = point[0], point[1], point[2]
        # Which side of plane
        if not flip:
            # Point is 'in front of' plane if true
            if a * x + b * y + c * z + d > 0:
                clipped_points.append(point)
        else:
            if a * x + b * y + c * z + d < 0:
                clipped_points.append(point)

    return clipped_points

def clip_points_from_plane(object_points, distance, rvec, tvec, clip_behind_plane):
    transformation_inverse = np.hstack((rvec, tvec.reshape(3, 1)))
    # # 4x4 transformation matrix
    transformation_inverse = np.vstack((transformation_inverse, [[0, 0, 0, 1]]))
    transformation = np.linalg.inv(transformation_inverse)

    # TODO probably better to use normal vector instead
    plane_point_z = transformation * np.asarray([0, 0, distance, 1]).reshape(4, 1)  # Point on camera plane in z direction
    plane_point_x = transformation * np.asarray([1, 0, distance, 1]).reshape(4, 1)  # Point on camera plane in x direction
    plane_point_y = transformation * np.asarray([0, 1, distance, 1]).reshape(4, 1)  # Point on camera plane in y direction

    plane = equation_plane(np.ravel(plane_point_z), np.ravel(plane_point_x), np.ravel(plane_point_y))
    object_points = clip_points(object_points, plane, clip_behind_plane)

    return object_points

def map_depth(points, plane):
    a, b, c, d = plane
    depth_map = []

    # Compute distance between point and camera plane
    for point in points:
        x, y, z = point[0], point[1], point[2]
        distance = abs(a * x + b * y + c * z + d) / math.sqrt(a**2 + b**2 + c**2)
        depth_map.append(distance)

    return np.array(depth_map)

def map_lidar_points_onto_image(camera_intrinsics, image_original, lidars, frame):
    image_canvas = np.copy(image_original)

    print('prepare lidar points.. ', end='')
    object_points_list = []
    for lidar in lidars:
        a = np.float64([lidar.x, lidar.y, lidar.z])
        a = a.transpose()
        object_points_list.append(a)
    # object_points_list = [a.transpose() for a in object_points_list]
    object_points = np.vstack(object_points_list)
    print('done')

    rx, ry, rz = frame['rotation']  # Roll, pitch, heading (= yaw)
    tx, ty, tz = frame['position']
    camera_origin = np.float64([tx, ty, tz - config.Z_CORRECTION])

    # From world to camera coordinate system
    # Camera rotation vector
    rvec = np.float64(utils.euler_to_rotation_matrix(90 - ry, -rx, rz))
    # Rotate translation vector
    tvec = rvec.dot(-1 * camera_origin)

    cam_matrix = camera_data.create_camera_matrix(camera_intrinsics)  # matrix K
    distortion = camera_data.get_distortion(camera_intrinsics)

    # Clip points behind and x meters away from camera
    object_points_closeby = clip_points_from_plane(object_points, config.FARAWAY_DISTANCE, rvec, tvec, clip_behind_plane=True)
    object_points_clipped = clip_points_from_plane(object_points_closeby, config.CLOSEBY_DISTANCE, rvec, tvec, clip_behind_plane=False)

    print('compute image points.. ', end='')
    image_points, _ = cv2.projectPoints(np.float64(object_points_clipped), rvec, tvec, cam_matrix, distortion)
    print('done')

    print('generate image.. ', end='')
    print('#image_points', len(image_points))

    transformation_inverse = np.hstack((rvec, tvec.reshape(3, 1)))
    # # 4x4 transformation matrix
    transformation_inverse = np.vstack((transformation_inverse, [[0, 0, 0, 1]]))
    transformation = np.linalg.inv(transformation_inverse)

    # TODO probably better to use normal vector instead
    camera_point_z = transformation * np.asarray([0, 0, 0, 1]).reshape(4, 1)  # Point on camera plane in x direction
    camera_point_x = transformation * np.asarray([1, 0, 0, 1]).reshape(4, 1)  # Point on camera plane in x direction
    camera_point_y = transformation * np.asarray([0, 1, 0, 1]).reshape(4, 1)  # Point on camera plane in y direction
    camera_plane = equation_plane(np.ravel(camera_point_z), np.ravel(camera_point_x), np.ravel(camera_point_y))
    object_points_depth = map_depth(object_points_clipped, camera_plane)

    min_distance = np.min(object_points_depth)
    max_distance = np.max(object_points_depth)
    normalized_distance = (object_points_depth - min_distance) / (max_distance - min_distance)
    # # Sort by farthest points to closest points (latter are then drawn last).
    # # Note that np.argsort() only works in ascending order
    image_points_sorted_by_depth = image_points[np.argsort(-normalized_distance)]
    normalized_distance_sorted = -np.sort(-normalized_distance)

    colors = [np.asarray(utils.hsv_to_rgb(0.75 * d, 1.0, 1.0)) * 255 for d in normalized_distance_sorted]

    # TODO very slow implementation, change to something such as
    # image_canvas[rows, cols] = (color[2], color[1], color[0])
    # -> see visualization.py?
    for idx, p in enumerate(image_points_sorted_by_depth):
        p = p.astype(np.int)
        point = {'x': p[0][0], 'y': p[0][1]}
        color = colors[idx]
        image_canvas = cv2.circle(image_canvas,
                                  (int(point['x']), int(point['y'])),
                                  radius=1,
                                  color=(color[2], color[1], color[0]),
                                  thickness=-1)

    print('done')
    return image_canvas

def run(frames, lidar_data, cameras_intrinsics, video_metadata, save_images=False, output_path=''):
    for run_name, camera_names in frames.items():
        for camera_name, frame_data in camera_names.items():
            for frame_id, image_path in frame_data.items():
                frame_metadata = video_metadata[run_name][camera_name][frame_id]
                camera_intrinsics = list(filter(lambda cam: cam['id'] == camera_name, cameras_intrinsics))[0]
                camera_data.print_frame_metadata(frame_id, frame_metadata)

                image_data = cv2.imread(image_path)
                mapped_image = map_lidar_points_onto_image(camera_intrinsics, image_data, lidar_data.values(), frame_metadata)

                _, ax = plt.subplots(figsize=(20, 20))
                ax.imshow(mapped_image[:, :, ::-1])  # RGB -> BGR
                ax.axis('off')

                if save_images:
                    print('save image')
                    plt.savefig(f'{output_path}{camera_name}/{frame_id}.jpg')
                else:
                    print('show image')
                    plt.show()
    print()


def draw_annotations_onto_image(annotations, camera_intrinsics, image_canvas_2d, frame, dwg, colors):
    image_canvas_3d = np.copy(image_canvas_2d)
    line_thickness = 7
    font_size = 2
    font_thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    rx, ry, rz = frame['rotation']  # Roll, pitch, heading
    tx, ty, tz = frame['position']
    camera_origin = np.float64([tx, ty, tz - config.Z_CORRECTION])
    w, h = int(camera_intrinsics['resolution'][0]), \
           int(camera_intrinsics['resolution'][1])

    # From world to camera coordinate system
    # Camera rotation vector
    rvec = np.float64(utils.euler_to_rotation_matrix(90 - ry, -rx, rz))
    # Rotate translation vector
    tvec = rvec.dot(-1 * camera_origin)

    cam_matrix = camera_data.create_camera_matrix(camera_intrinsics)  # matrix K
    distortion = camera_data.get_distortion(camera_intrinsics)

    print('visualize annotation points.. ')
    count = 0
    for key, annotations in annotations.items():
        for annotation in annotations:
            layer_name = annotation.dxf.layer
            layer_obj = dwg.layers.get(layer_name)
            layer_color = ezdxf.colors.aci2rgb(layer_obj.color)
            # print(f'layer {layer_name}, color {layer_color}')
            if not annotation.has_xdata(config.DXF_APPID):
                sys.exit('no custom XDATA present, exit')
            tags = annotation.get_xdata(config.DXF_APPID)
            bbox_id = tags[0].value

            v = []
            if key == 'polylines':
                v = list(annotation.points())
            if key == 'circles':
                annotation.dxf.radius = annotation.dxf.radius * 1.1
                # Get points on circle at x specified angles
                angles = list(range(0, 360, 5))
                v = list(annotation.vertices(angles))
            if key == 'bboxes':
                v = list(annotation.vertices)

            vertices = np.asarray(v, dtype=np.float64)
            object_points_world = np.vstack(vertices)

            # Clip bboxes and circles behind camera and x meters away from camera
            if key != 'polylines':
                # Clip points behind and x meters away from camera
                object_points_closeby = clip_points_from_plane(
                    object_points_world, config.FARAWAY_DISTANCE, rvec, tvec, clip_behind_plane=True
                )
                object_points_clipped = clip_points_from_plane(
                    object_points_closeby, config.CLOSEBY_DISTANCE, rvec, tvec, clip_behind_plane=False
                )
            else:
                # Clip polylines right behind camera
                object_points_closeby = clip_points_from_plane(
                    object_points_world, config.FARAWAY_DISTANCE, rvec, tvec, clip_behind_plane=True
                )
                object_points_clipped = clip_points_from_plane(
                    object_points_closeby, 0.1, rvec, tvec, clip_behind_plane=False
                )

            if len(object_points_clipped) != 0:
                print('compute image points.. ', end='')
                image_points, _ = cv2.projectPoints(np.float64(object_points_clipped), rvec, tvec, cam_matrix,
                                                    distortion)
                print('done')

                print('generate image.. ', end='')
                pts = np.array(image_points, np.int32)
                color = layer_color[::-1]  # RGB -> BGR

                coords_within_image_bounds = np.size(
                    pts[(0 <= pts) & (pts <= w) & (pts <= h)]
                )

                if key == 'polylines':
                    rail_id = int(bbox_id.split('#')[-1])
                    text_pos = (50, 50 + 50 * rail_id)
                    image_canvas_3d = cv2.rectangle(image_canvas_3d, (text_pos[0] - 10, text_pos[1] - 50),
                                                    (text_pos[0] + 50, text_pos[1] + 10), (0, 0, 0), -1)
                    image_canvas_3d = cv2.putText(image_canvas_3d, str(rail_id), text_pos, font, font_size,
                                                  colors[rail_id], font_thickness)
                    image_canvas_3d = cv2.polylines(image_canvas_3d, [pts], isClosed=False, color=colors[rail_id],
                                                    thickness=line_thickness)

                # Check if all 8*2 (for x and y) corners to be drawn are present on image canvas, otherwise we don't draw a bbox
                if coords_within_image_bounds >= 8 * 2:
                    if key == 'circles':
                        pts = pts.reshape((-1, 2))
                        image_canvas_3d = cv2.polylines(image_canvas_3d, [pts], isClosed=True, color=color,
                                                        thickness=line_thickness)
                    if key == 'bboxes':
                        x_min, y_min, x_max, y_max = training_data.create_bbox_2d(pts)
                        image_canvas_2d = cv2.rectangle(image_canvas_2d, (x_min, y_min), (x_max, y_max), color,
                                                        line_thickness)

                        image_canvas_3d = draw_bbox_3d(image_canvas_3d, pts, color, line_thickness)

                    count += 1

    print('#visible objects in image', count)

    print('done')
    return image_canvas_2d, image_canvas_3d

def visualize_dwg_annotations(dwg,
                              frames,
                              cameras_intrinsics,
                              video_metadata,
                              save_images=True,
                              output_path=''):
    msp = dwg.modelspace()
    polylines = msp.query('POLYLINE[layer!="0"]')
    bboxes = msp.query('MESH[layer!="0"]')
    circles = msp.query('CIRCLE[layer!="0"]')
    n_polylines = len(polylines)
    print('#polylines', n_polylines)
    annotations = {
        'polylines': polylines,
        'bboxes': bboxes,
        'circles': circles,
    }
    colors = [tuple(np.random.choice(range(256), size=3).tolist())
              for _ in range(n_polylines)]

    print('layers in file:')
    print('\n'.join([f'{layer.dxf.name}' for layer in dwg.layers]))
    print()

    for run_name, camera_names in frames.items():
        for camera_name, frame_data in camera_names.items():
            for frame_id, image_path in frame_data.items():
                frame_metadata = video_metadata[run_name][camera_name][frame_id]
                camera_intrinsics = list(filter(lambda cam: cam['id'] == camera_name, cameras_intrinsics))[0]
                camera_data.print_frame_metadata(frame_id, frame_metadata)

                image_data = cv2.imread(image_path)
                image_2d_bbox, image_3d_bbox = draw_annotations_onto_image(
                    annotations, camera_intrinsics, image_data, frame_metadata, dwg, colors
                )

                if save_images:
                    print('save image')
                    run_name_short = run_name.split('-')[-1]
                    cv2.imwrite(f'{output_path}3d_dwg_{run_name_short}{frame_id}.jpg', image_3d_bbox)
                else:
                    print('show image')
                    _, ax = plt.subplots(figsize=(20, 20))
                    ax.imshow(image_3d_bbox[:, :, ::-1])  # RGB -> BGR
                    # ax.imshow(image_2d_bbox[:, :, ::-1])
                    ax.axis('off')
                    plt.show()
                print()