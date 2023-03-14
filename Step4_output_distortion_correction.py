import pickle
import numpy as np
import glob
from scripts import camera_data
import math
import matplotlib.pyplot as plt

# This code corresponds to Step 4: Distortion correction for detection output
# mainly used to correct the influence of lens distortion in the detection output

# predefined configuration
RUN_NAMES = ['ep11-201002303-20190430-080329', 'ep11-201002303-20190430-093039']
CAMS = ['cam2']
root_path = './data/datasets/bboxes_new/'

# read the output pickle file
dir_output = root_path + 'testing/pred/pred_railway_img_bbox_test_cleaned.pkl'
# dir_output = root_path + 'testing/pred/pred_railway_sl_cleaned.pkl'
# dir_output = root_path + 'testing/pred/pred_railway_cab_cleaned.pkl'
# dir_output = root_path + 'testing/pred/pred_railway_mark_cleaned.pkl'
pkl_output = pickle.load(open(dir_output, "rb"))

# predefined lens distortion correction functions
# input undistorted dimensionless ray, output distorted pixel after adding distortion
def get_pixel(ray_coord):
    # input: ray_coord - a tuple (x', y') contain dimensionless ray coordinate (undistorted)
    # output: a distorted 2d ray (x'', y'') also in tuple
    # ray components are dimensionless
    x2 = ray_coord[0] * ray_coord[0]
    y2 = ray_coord[1] * ray_coord[1]
    # TODO: what's the meaning of original csharp code?
    # r2 = _modelParams.UseModel2 ? Math.Pow(Math.Atan(Math.Sqrt(x2+y2)),2) : x2 + y2
    r2 = x2 + y2
    xy2 = 2 * ray_coord[0] * ray_coord[1]
    kr = 1 + ((0 * r2 + distortion[1]) * r2 + distortion[0]) * r2
    u = ray_coord[0] * kr + distortion[2] * xy2 + distortion[3] * (r2 + 2 * x2)
    v = ray_coord[1] * kr + distortion[2] * (r2 + 2 * y2) + distortion[3] * xy2

    # compute pixel coordinates by multiply by focal length and apply principal point shift
    x = focal_length[0] * u + principal_point[0]
    y = focal_length[1] * v + principal_point[1]

    return (x, y)

# input distorted pixel coordinates, output undistorted dimensionless ray
def get_ray(pixel_coord):
    # input: pixel_coord - a tuple (x, y) contains the pixel coordinate in x and y-axis on the frame
    # the input coordinate is distorted (uncorrected)
    # output: tuple (x', y') - undistorted ray
    # get coordinates relative to principal point and make dimensionless
    x = (pixel_coord[0] - principal_point[0])/focal_length[0]
    y = (pixel_coord[1] - principal_point[1])/focal_length[1]
    x0 = x
    y0 = y

    # reverse corrections are to be approximated by iteration
    delta_square = 0.001
    max_iteration = 25

    for i in range(max_iteration):
        r2 = x * x + y * y
        # var icdist = 1 / (1 + (((_modelParams.K4 * r2 + _modelParams.K3) * r2 + _modelParams.K2) * r2 + _modelParams.K1) * r2);
        icdist = 1/(1 + ((0 * r2 + distortion[1]) * r2 + distortion[0]) * r2)
        # correcting (inverse) radial distortion coefficient  
        # var deltaX = 2 * _modelParams.P1 * x * y + _modelParams.P2 * (r2 + 2 * x * x);
        deltaX = 2 * distortion[2] * x * y + distortion[3] * (r2 + 2 * x * x)
        # var deltaY = _modelParams.P1 * (r2 + 2 * y * y) + 2 * _modelParams.P2 * x * y;
        deltaY = distortion[2] * (r2 + 2 * y * y) + 2 * distortion[3] * x * y
        x = (x0 - deltaX) * icdist
        y = (y0 - deltaY) * icdist
        ray_coord = (x, y)

        if math.dist(get_pixel(ray_coord), pixel_coord) < delta_square:
            break
    return ray_coord

def get_undistorted_pixel(ray_coord):
    # transfer undistorted ray to undistorted pixel coordinate
    # compute pixel coordinates by multiply by focal length and apply principal point shift
    x = focal_length[0] * ray_coord[0] + principal_point[0]
    y = focal_length[1] * ray_coord[1] + principal_point[1]
    pixel_undistorted = (x, y)

    return pixel_undistorted

# some functions can be used to transfer camera xyz to world xyz
# camera [xyz]^T = R^(-1) * (V-T)
# R: rotation matrix from extrinsic, V: object position in world coordinate, T: camera position in world coordinate
# to be computed: V, V = T + R * camera [xyz]^T
# from camera extrinsic to rotation matrix
def eulerAnglesToRotationMatrix(roll, pitch, heading):
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(roll), -math.sin(roll)],
                    [0, math.sin(roll), math.cos(roll)]
                    ])

    R_y = np.array([[math.cos(pitch), 0, math.sin(pitch)],
                    [0, 1, 0],
                    [-math.sin(pitch), 0, math.cos(pitch)]
                    ])

    R_z = np.array([[math.cos(heading), -math.sin(heading), 0],
                    [math.sin(heading), math.cos(heading), 0],
                    [0, 0, 1]
                    ])

    R = np.dot(R_z, np.dot(R_y, R_x))
    return R
# the output camera xyz is in opencv coordinate system
# need to transfer them to NED system
def NEDCVConvert(t):
    # t: translation vector (position) in openCV frame, shape should be (3, 1)
    # convert OpenCV system to NED system
    R_CV2NED = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]])
    t_NED = np.dot(R_CV2NED, t)

    return t_NED
# from camera xyz to world xyz
def cam2world(t, loc, R):
    # t: world xyz for camera (easting, northing, up)
    # loc: camera xyz for object in opencv system (output)
    # R: rotation from camera extrinsic
    # transfer t to NED system and reshape
    t_NED = [t[1], t[0], -t[2]]
    t_reshape = np.array(t_NED).reshape(3, 1)
    # reshape loc to (3, 1) and transfer it to NED system
    loc_reshape = np.array(loc).reshape(3, 1)
    loc_NED = NEDCVConvert(loc_reshape)
    # compute world xyz for object: V = T + R * camera [xyz]^T
    # also in NED system
    loc_world_NED = t_reshape + np.dot(R, loc_NED)
    # transfer world xyz from NED system to (easting, northing, up)
    loc_world = [loc_world_NED[1, 0], loc_world_NED[0, 0], -loc_world_NED[2, 0]]

    return loc_world

# get the camera intrinsics
camera_intrinsics = camera_data.get_camera_intrinsics(CAMS)[0]
frames = camera_data.prepare_frames(RUN_NAMES, CAMS)
video_metadata = camera_data.get_video_metadata(RUN_NAMES, CAMS)

# distortion: a list, from left to right - k1, k2, p1, p2
distortion = camera_data.get_distortion(camera_intrinsics)
principal_point = camera_intrinsics['center']
focal_length = camera_intrinsics['focal']
# transfer the string in focal_length and principal_point as float numbers
principal_point = (float(principal_point[0]), float(principal_point[1]))
focal_length = (float(focal_length[0]), float(focal_length[1]))

with open(f'{root_path}flags/fugro-to-kitti-ids.txt', 'r') as f:
    kitti_fugro_ids = [tuple(line.strip().split(' ')) for line in f]

bbox_center_change = []
location_change = []
world_locs_old = []
world_locs_new = []

# for each object in the pkl file
for frame in pkl_output:
    world_location = []
    for i in range(len(frame['name'])):
        # bbox: from left to right - left, top, right, bottom
        bbox = frame['bbox'][i]
        # location: camera xyz in meters (in opencv frame)
        location = frame['location'][i]
        bbox_center = ((bbox[0] + bbox[2])/2, (bbox[1] + bbox[3])/2)
        # vertex (left, bottom)
        bbox_vertex1 = (bbox[0], bbox[3])
        # vertex (right, top)
        bbox_vertex2 = (bbox[2], bbox[1])

        # from data_idx in output file to its correspond kitti file name
        data_idx = frame['sample_idx'][i]
        filename_kitti = str(data_idx).zfill(6)
        # get the frame's run_name and fugro frame_id
        run_name, frame_id = [(item[1], item[2]) for item in kitti_fugro_ids
                              if item[0] == filename_kitti][0]
        for cam in CAMS:
            # roll, pitch, heading in degrees
            rotation = video_metadata[run_name][cam][frame_id]['rotation']
            # camera position t in world coordinate
            t = video_metadata[run_name][cam][frame_id]['position']
        # compute the rotation matrix
        R = eulerAnglesToRotationMatrix(np.radians(rotation[0]), np.radians(rotation[1]), np.radians(rotation[2]))

        world_locs_old.append(cam2world(t, location, R))

        # get the undistorted ray (x', y')
        ray_undistorted = get_ray(bbox_center)
        ray_undistorted_v1 = get_ray(bbox_vertex1)
        ray_undistorted_v2 = get_ray(bbox_vertex2)
        # from ray (x', y') compute the corrected pixel coordinate for center
        bbox_center_undistorted = get_undistorted_pixel(ray_undistorted)
        bbox_center_undistorted_v1 = get_undistorted_pixel(ray_undistorted_v1)
        bbox_center_undistorted_v2 = get_undistorted_pixel(ray_undistorted_v2)
        # use z information compute corrected camera xyz
        z = location[2]
        location_corrected = [ray_undistorted[0] * z, ray_undistorted[1] * z, z]
        # world xyz
        location_world_corrected = cam2world(t, location_corrected, R)
        world_location.append(location_world_corrected)

        world_locs_new.append(world_location)
        bbox_center_change.append((bbox_center_undistorted[0] - bbox_center[0], bbox_center_undistorted[1] - bbox_center[1]))
        location_change.append(location_corrected - location)

        # corrected the data in the pkl file
        frame['bbox'][i] = [bbox_center_undistorted_v1[0], bbox_center_undistorted_v2[1],
                            bbox_center_undistorted_v2[0], bbox_center_undistorted_v1[1]]
        # frame['rotation_y'][i] = 0
        frame['location'][i] = location_corrected
        frame.update({'world_location': np.array(world_location)})

# save the corrected list as a new pickle file
new_pkl_path = root_path + 'testing/pred/pred_railway_img_bbox_test_corrected.pkl'
with open(new_pkl_path, 'wb') as pkl:
    pickle.dump(pkl_output, pkl)

bbox_center_change_u = [coord[0] for coord in bbox_center_change]
bbox_center_change_v = [coord[1] for coord in bbox_center_change]
location_change_x = [loc[0] for loc in location_change]
location_change_y = [-loc[1] for loc in location_change]

# plt.figure()
# plt.hist(location_change_x, bins=50)
# plt.xlabel('Difference of corrected camera x [m]')
# plt.ylabel('Frequency')
# plt.title('Distribution of the Change of Camera X After Distortion Correction')
# plt.show()

# plt.figure()
# plt.hist(location_change_y, bins=20)
# plt.xlabel('Difference of corrected camera y [m]')
# plt.ylabel('Frequency')
# plt.title('Distribution of the Change of Camera Y After Distortion Correction')
# plt.show()

# plt.figure()
# plt.hist(location_change_y_sl, alpha=0.5, bins=20, label='Signal Light')
# plt.hist(location_change_y_cab, alpha=0.5, bins=20, label='Cabinet')
# plt.hist(location_change_y_mark, alpha=0.5, bins=20, label='Marker')
# plt.xlabel('Difference of corrected camera y [m]', fontsize=13)
# plt.ylabel('Frequency', fontsize=13)
# plt.title('Distribution of the Change of Camera Y After Distortion Correction', fontsize=13)
# plt.legend()
# plt.show()

# plt.figure()
# plt.hist(bbox_center_change_u, bins=50)
# plt.xlabel('Difference of corrected u [pixel]')
# plt.ylabel('Frequency')
# plt.title('Distribution of the Change of Pixel u After Distortion Correction')
# plt.show()

# plt.figure()
# plt.hist(bbox_center_change_v, bins=50)
# plt.xlabel('Difference of corrected v [pixel]')
# plt.ylabel('Frequency')
# plt.title('Distribution of the Change of Pixel v After Distortion Correction')
# plt.show()

# # take one 2D bbox pixel coordinate for test
# # bbox: from left to right - left, top, right, bottom
# bbox = pkl_output[1]['bbox']
# # location: camera xyz in meters (in opencv frame)
# location = pkl_output[1]['location']
# # all vertices coordinate in pixels
# # (left, bottom), (left, top), (right, top), (right, bottom)
# bbox_vertices = [(bbox[0, 0], bbox[0, 3]), (bbox[0, 0], bbox[0, 1]),
#                  (bbox[0, 2], bbox[0, 1]), (bbox[0, 2], bbox[0, 3])]
#
# bbox_center = ((bbox[0, 0] + bbox[0, 2])/2, (bbox[0, 1] + bbox[0, 3])/2)
#
# # get the camera intrinsics
# camera_intrinsics = camera_data.get_camera_intrinsics(['cam2'])[0]
# frames = camera_data.prepare_frames(['ep11-201002303-20190430-080329'], ['cam2'])
# video_metadata = camera_data.get_video_metadata(['ep11-201002303-20190430-080329'], ['cam2'])
#
# # distortion: a list, from left to right - k1, k2, p1, p2
# distortion = camera_data.get_distortion(camera_intrinsics)
# principal_point = camera_intrinsics['center']
# focal_length = camera_intrinsics['focal']
#
# # transfer the string in focal_length and principal_point as float numbers
# principal_point = (float(principal_point[0]), float(principal_point[1]))
# focal_length = (float(focal_length[0]), float(focal_length[1]))
#
# ray_undistorted = get_ray(bbox_center)
# bbox_center_undistorted = get_undistorted_pixel(ray_undistorted)
#
# z = location[0, 2]
# location_corrected = [ray_undistorted[0]*z, ray_undistorted[1]*z, z]