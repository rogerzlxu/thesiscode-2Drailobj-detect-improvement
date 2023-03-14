import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import math
from mpl_toolkits.mplot3d import Axes3D

# the directory of
dir_output = './data/datasets/bboxes_new/testing/pred/pred_railway_img_bbox_test.pkl'
# dir_train = './data/datasets/bboxes/kitti_infos_trainval.pkl'
dir_test = './data/datasets/bboxes_new/kitti_infos_test.pkl'

# output file of SMOKE
pkl_output = pickle.load(open(dir_output, "rb"))
# test files with annotations
pkl_test = pickle.load(open(dir_test, "rb"))
# del pkl_test[0:10]
# del pkl_output[0:10]

pkl_test_annos = [anno['annos'] for anno in pkl_test]

x = []
y = []
z = []

for frame in pkl_output:
    for i in range(len(frame['name'])):
        location = frame['location'][i]
        x.append(location[0])
        y.append(location[1])
        z.append(location[2])

# get the false positive cases in the output file
def get_fp(pkl_output):
    fp = []
    for i in range(len(pkl_output)):
        if len(pkl_test_annos[i]['name']) == 0 and len(pkl_output[i]['name']) > 0:
            fp.append(pkl_output[i])
        elif len(pkl_test_annos[i]['name']) != 0 and len(pkl_output[i]['name']) != 0:
            for name in pkl_output[i]['name']:
                for gt_name in pkl_test_annos[i]['name']:
                    if name != gt_name:
                        fp.append({
                            'name': ([name]),
                            'truncated': ([pkl_output[i]['truncated'][np.where(pkl_output[i]['name'] == name)[0][0]]]),
                            'occluded': ([pkl_output[i]['occluded'][np.where(pkl_output[i]['name'] == name)[0][0]]]),
                            'alpha': ([pkl_output[i]['alpha'][np.where(pkl_output[i]['name'] == name)[0][0]]]),
                            'bbox': ([pkl_output[i]['bbox'][np.where(pkl_output[i]['name'] == name)[0][0]]]),
                            'dimensions': (
                            [pkl_output[i]['dimensions'][np.where(pkl_output[i]['name'] == name)[0][0]]]),
                            'location': ([pkl_output[i]['location'][np.where(pkl_output[i]['name'] == name)[0][0]]]),
                            'rotation_y': (
                            [pkl_output[i]['rotation_y'][np.where(pkl_output[i]['name'] == name)[0][0]]]),
                            'score': ([pkl_output[i]['score'][np.where(pkl_output[i]['name'] == name)[0][0]]]),
                            'sample_idx': ([pkl_output[i]['sample_idx'][np.where(pkl_output[i]['name'] == name)[0][0]]])
                        })
        elif len(pkl_output[i]['name']) > len(pkl_test_annos[i]['name']):
            if len(Counter(pkl_output[i]['name']).keys()) == len(Counter(pkl_test_annos[i]['name']).keys()):
                for k in range(len(Counter(pkl_output[i]['name']).keys())):
                    if Counter(pkl_output[i]['name'])[list(Counter(pkl_output[i]['name']).keys())[k]] != \
                            Counter(pkl_test_annos[i]['name'])[list(Counter(pkl_output[i]['name']).keys())[k]]:
                        distance = []
                        type = list(Counter(pkl_output[i]['name']).keys())[k]
                        for index in np.where(pkl_output[i]['name'] == type)[0]:
                            dist = math.dist(pkl_output[i]['location'][index], pkl_test_annos[i]['location'][
                                np.where(pkl_test_annos[i]['name'] == type)[0][0]])
                            distance.append(dist)
    return fp

fp = get_fp(pkl_output)
x_fp = []
y_fp = []
z_fp = []

for obj in fp:
    for i in range(len(obj['name'])):
        location = obj['location'][i]
        x_fp.append(location[0])
        y_fp.append(location[1])
        z_fp.append(location[2])

plt.figure()
plt.scatter(x, z, s=8)
plt.scatter(x_fp, z_fp, s=10, c='r')
plt.xlabel('X [m]')
plt.ylabel('Z [m]')
# plt.xlim(-4, 4)
plt.title('X-Z Profile of the Distribution of All Predictions Camera XYZ')
# plt.axhline(y=30, color='r', linestyle='--')
# plt.axhline(y=25, color='k', linestyle='--', label='z=25m')
# plt.axvline(x=-4, color='g', linestyle='--', label='-4<x<5m')
# plt.axvline(x=5, color='g', linestyle='--')
# plt.axvline(x=-2, color='b', linestyle='--', label='-2<x<2m')
# plt.axvline(x=2, color='b', linestyle='--')
# plt.hlines(y=29, xmin=-4, xmax=-2, color='k', linestyle='-')
# plt.hlines(y=29, xmin=2, xmax=5, color='k', linestyle='-')
# plt.vlines(x=-2, ymin=5, ymax=25, color='b', linestyle='-')
# plt.vlines(x=2, ymin=5, ymax=25, color='b', linestyle='-')
# plt.vlines(x=-4, ymin=5, ymax=25, color='g', linestyle='-')
# plt.vlines(x=5, ymin=5, ymax=25, color='g', linestyle='-')
plt.grid()
# plt.legend()
plt.show()

plt.figure()
plt.scatter(x, y, s=10)
plt.scatter(x_fp, y_fp, s=10, c='r')
# plt.axvline(x=-4, color='g', linestyle='--', label='-4<x<5m')
# plt.axvline(x=5, color='g', linestyle='--')
# plt.axvline(x=-2, color='b', linestyle='--', label='-2<x<2m')
# plt.axvline(x=2, color='b', linestyle='--')
# plt.axhline(y=2.5, color='k', linestyle='--', label='0<y<2.5m')
# plt.axhline(y=0, color='k', linestyle='--')
# plt.hlines(y=0, xmin=-4, xmax=-2, color='k', linestyle='-')
# plt.hlines(y=0, xmin=2, xmax=5, color='k', linestyle='-')
# plt.hlines(y=2.5, xmin=-4, xmax=-2, color='k', linestyle='-')
# plt.hlines(y=2.5, xmin=2, xmax=5, color='k', linestyle='-')
# plt.vlines(x=-2, ymin=0, ymax=2.5, color='b', linestyle='-')
# plt.vlines(x=2, ymin=0, ymax=2.5, color='b', linestyle='-')
# plt.vlines(x=-4, ymin=0, ymax=2.5, color='g', linestyle='-')
# plt.vlines(x=5, ymin=0, ymax=2.5, color='g', linestyle='-')
plt.xlabel('X [m]')
plt.ylabel('Y [m]')
plt.title('X-Y Profile of the Distribution of All Predictions Camera XYZ')
plt.grid()
# plt.legend()
plt.show()

plt.figure()
plt.scatter(z, y, s=10)
plt.scatter(z_fp, y_fp, s=10, c='r')
# plt.axvline(x=29, color='b', linestyle='--', label='z=29m')
# plt.axhline(y=2.5, color='k', linestyle='--', label='0<y<2.5m')
# plt.axhline(y=0, color='k', linestyle='--')
# plt.hlines(y=0, xmin=5, xmax=29, color='k', linestyle='-')
# plt.hlines(y=2.5, xmin=5, xmax=29, color='k', linestyle='-')
# plt.vlines(x=5, ymin=0, ymax=2.5, color='b', linestyle='-')
# plt.vlines(x=29, ymin=0, ymax=2.5, color='b', linestyle='-')
plt.xlabel('Z [m]')
plt.ylabel('Y [m]')
plt.title('Z-Y Profile of the Distribution of All Predictions Camera XYZ')
plt.grid()
# plt.legend()
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x, z, y, s=10, label='true positive')
# ax.scatter(x_fp, z_fp, y_fp, s=15, marker='o', c='r', label='false positive')
# # ax.set_xlim((-4, 4))
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Z [m]')
# ax.set_zlabel('Y [m]')
# ax.set_title('3D Plot for the Distribution of All Predictions Camera XYZ')
# plt.legend()
# plt.show()

# # set a limitation of x with [-4, 4] meters
# # and select ground truths in this range
# for frame in pkl_output:
#     for loc in frame['location']:
#         if loc[0] < -4 or loc[0] > 5 or -2 < loc[0] < 2 or loc[1] < 0 or loc[1] > 2.5 or loc[2] > 29:
#             i = np.where(frame['location']==loc)[0][0]
#             frame['location'] = np.delete(frame['location'], i, axis=0)
#             frame['name'] = np.delete(frame['name'], i)
#             frame['truncated'] = np.delete(frame['truncated'], i)
#             frame['occluded'] = np.delete(frame['occluded'], i)
#             frame['alpha'] = np.delete(frame['alpha'], i)
#             frame['bbox'] = np.delete(frame['bbox'], i, axis=0)
#             frame['dimensions'] = np.delete(frame['dimensions'], i, axis=0)
#             frame['rotation_y'] = np.delete(frame['rotation_y'], i)
#             frame['score'] = np.delete(frame['score'], i)
#             frame['sample_idx'] = np.delete(frame['sample_idx'], i)
#
# x = []
# y = []
# z = []
#
# for frame in pkl_output:
#     for i in range(len(frame['name'])):
#         location = frame['location'][i]
#         x.append(location[0])
#         y.append(location[1])
#         z.append(location[2])
#
# fp = get_fp(pkl_output)
# x_fp = []
# y_fp = []
# z_fp = []
#
# for obj in fp:
#     for i in range(len(obj['name'])):
#         location = obj['location'][i]
#         x_fp.append(location[0])
#         y_fp.append(location[1])
#         z_fp.append(location[2])
#
# plt.figure()
# plt.scatter(z, y, s=10)
# plt.scatter(z_fp, y_fp, s=12, c='r')
# plt.xlabel('Z [m]')
# plt.ylabel('Y [m]')
# plt.title('Z-Y Profile of the Distribution of Cleaned-up Predictions Camera XYZ')
# plt.grid()
# plt.show()
#
# plt.figure()
# plt.scatter(x, z, s=10)
# plt.scatter(x_fp, z_fp, s=12, c='r')
# plt.xlabel('X [m]')
# plt.ylabel('Z [m]')
# plt.title('X-Z Profile of the Distribution of Cleaned-up Predictions Camera XYZ')
# plt.grid()
# plt.show()
#
# plt.figure()
# plt.scatter(x, y, s=10)
# plt.scatter(x_fp, y_fp, s=12, c='r')
# plt.xlabel('X [m]')
# plt.ylabel('Y [m]')
# plt.title('X-Y Profile of the Distribution of Cleaned-up Predictions Camera XYZ')
# plt.grid()
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(x, z, y, s=10, label='true positive')
# ax.scatter(x_fp, z_fp, y_fp, s=20, marker='o', c='r', label='false positive')
# # ax.set_xlim((-4, 4))
# ax.set_xlabel('X [m]')
# ax.set_ylabel('Z [m]')
# ax.set_zlabel('Y [m]')
# ax.set_title('3D Plot for the Distribution of All Predictions Camera XYZ')
# plt.legend()
# plt.show()

# rotation_y = []
# for frame in pkl_output:
#     for ry in frame['rotation_y']:
#         rotation_y.append(ry)
#
# rotation_y_gt = []
# for frame in pkl_test_annos:
#     for ry in frame['rotation_y']:
#         rotation_y_gt.append(ry)
#
# # a case on rotation angle
# rotation_y_gt = [pkl_test_annos[1031]['rotation_y'][0], pkl_test_annos[1032]['rotation_y'][0], pkl_test_annos[1033]['rotation_y'][1],
#                  pkl_test_annos[1034]['rotation_y'][1], pkl_test_annos[1035]['rotation_y'][1], pkl_test_annos[1036]['rotation_y'][1],
#                  pkl_test_annos[1037]['rotation_y'][1], pkl_test_annos[1038]['rotation_y'][1], pkl_test_annos[1039]['rotation_y'][1],
#                  pkl_test_annos[1040]['rotation_y'][1], pkl_test_annos[1041]['rotation_y'][1], pkl_test_annos[1042]['rotation_y'][1],
#                  pkl_test_annos[1043]['rotation_y'][1], pkl_test_annos[1044]['rotation_y'][1], pkl_test_annos[1045]['rotation_y'][1],
#                  pkl_test_annos[1046]['rotation_y'][2], pkl_test_annos[1047]['rotation_y'][2], pkl_test_annos[1048]['rotation_y'][2]]
#
# rotation_y_output = [frame['rotation_y'][0] for frame in pkl_output[1031:1049]]
#
# plt.figure()
# plt.scatter(np.arange(1, 19, 1), rotation_y_gt, s=10, label='ground truths')
# plt.scatter(np.arange(1, 19, 1), rotation_y_output, s=10, c='r', label='predictions')
# plt.xlabel('frames')
# plt.ylabel('rotation_y [radians]')
# plt.xlim(0, 18)
# plt.legend()
# plt.show()
#
# x_gt = [pkl_test_annos[1031]['location'][0][0], pkl_test_annos[1032]['location'][0][0], pkl_test_annos[1033]['location'][1][0],
#         pkl_test_annos[1034]['location'][1][0], pkl_test_annos[1035]['location'][1][0], pkl_test_annos[1036]['location'][1][0],
#         pkl_test_annos[1037]['location'][1][0], pkl_test_annos[1038]['location'][1][0], pkl_test_annos[1039]['location'][1][0],
#         pkl_test_annos[1040]['location'][1][0], pkl_test_annos[1041]['location'][1][0], pkl_test_annos[1042]['location'][1][0],
#         pkl_test_annos[1043]['location'][1][0], pkl_test_annos[1044]['location'][1][0], pkl_test_annos[1045]['location'][1][0],
#         pkl_test_annos[1046]['location'][2][0], pkl_test_annos[1047]['location'][2][0], pkl_test_annos[1048]['location'][2][0]]
#
# x_output = [frame['location'][0][0] for frame in pkl_output[1031:1049]]
#
# y_gt = [pkl_test_annos[1031]['location'][0][1], pkl_test_annos[1032]['location'][0][1], pkl_test_annos[1033]['location'][1][1],
#         pkl_test_annos[1034]['location'][1][1], pkl_test_annos[1035]['location'][1][1], pkl_test_annos[1036]['location'][1][1],
#         pkl_test_annos[1037]['location'][1][1], pkl_test_annos[1038]['location'][1][1], pkl_test_annos[1039]['location'][1][1],
#         pkl_test_annos[1040]['location'][1][1], pkl_test_annos[1041]['location'][1][1], pkl_test_annos[1042]['location'][1][1],
#         pkl_test_annos[1043]['location'][1][1], pkl_test_annos[1044]['location'][1][1], pkl_test_annos[1045]['location'][1][1],
#         pkl_test_annos[1046]['location'][2][1], pkl_test_annos[1047]['location'][2][1], pkl_test_annos[1048]['location'][2][1]]
#
# y_output = [frame['location'][0][1] for frame in pkl_output[1031:1049]]
#
# z_gt = [pkl_test_annos[1031]['location'][0][2], pkl_test_annos[1032]['location'][0][2], pkl_test_annos[1033]['location'][1][2],
#         pkl_test_annos[1034]['location'][1][2], pkl_test_annos[1035]['location'][1][2], pkl_test_annos[1036]['location'][1][2],
#         pkl_test_annos[1037]['location'][1][2], pkl_test_annos[1038]['location'][1][2], pkl_test_annos[1039]['location'][1][2],
#         pkl_test_annos[1040]['location'][1][2], pkl_test_annos[1041]['location'][1][2], pkl_test_annos[1042]['location'][1][2],
#         pkl_test_annos[1043]['location'][1][2], pkl_test_annos[1044]['location'][1][2], pkl_test_annos[1045]['location'][1][2],
#         pkl_test_annos[1046]['location'][2][2], pkl_test_annos[1047]['location'][2][2], pkl_test_annos[1048]['location'][2][2]]
#
# z_output = [frame['location'][0][2] for frame in pkl_output[1031:1049]]


