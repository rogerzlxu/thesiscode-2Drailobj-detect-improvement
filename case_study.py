import pickle
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import math

from scripts.visualization import ObjectData
from config import config

# root path
# case for cabinet
# data_path = './data/datasets/bboxes_cabinet1/'
data_path = './data/datasets/bboxes_cabinet2/'
# case for signal light
# data_path = './data/datasets/bboxes_signallight/'
# data_path = './data/datasets/bboxes_signallight2/'
# case for markers
# data_path = './data/datasets/bboxes_marker1/'
# data_path = './data/datasets/bboxes_marker2/'
# data_path = './data/datasets/bboxes_marker3/'

output_case_dir = data_path + 'testing/pred/pred_railway_img_bbox_test_corrected.pkl'

# read the pkl files of the cases
output_case = pickle.load(open(output_case_dir, "rb"))

# delete objects with unexpected type
def object_cleanup(output_case, type_tobe_kept):
    for frame in output_case:
        for name in frame['name']:
            if name != type_tobe_kept:
                i = np.where(frame['name'] == name)[0][0]
                frame['location'] = np.delete(frame['location'], i, axis=0)
                frame['name'] = np.delete(frame['name'], i)
                frame['truncated'] = np.delete(frame['truncated'], i)
                frame['occluded'] = np.delete(frame['occluded'], i)
                frame['alpha'] = np.delete(frame['alpha'], i)
                frame['bbox'] = np.delete(frame['bbox'], i, axis=0)
                frame['dimensions'] = np.delete(frame['dimensions'], i, axis=0)
                frame['rotation_y'] = np.delete(frame['rotation_y'], i)
                frame['score'] = np.delete(frame['score'], i)
                frame['sample_idx'] = np.delete(frame['sample_idx'], i)
                frame['world_location'] = np.delete(frame['world_location'], i, axis=0)
    return output_case

# output_case = object_cleanup(output_case, 'Marker')
# output_case = object_cleanup(output_case, 'Signal_light')
output_case = object_cleanup(output_case, 'Cabinet')

# for frame in output_case:
#     for loc in frame['location']:
#         if loc[1] < 1:
#             i = np.where(frame['location']==loc)[0][0]
#
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

# get the world location of each bbox
world_locs = []
# also get the camera xyz for each prediction
x = []
y = []
z = []
idx = []
for frame in output_case:
    for i in range(len(frame['name'])):
        world_locs.append(frame['world_location'][i])
        x.append(frame['location'][i][0])
        y.append(frame['location'][i][1])
        z.append(frame['location'][i][2])
        idx.append(str(frame['sample_idx'][i]))

# get 2d world locations
world_locs_2d = []
for loc in world_locs:
    loc_2d = np.delete(loc, -1)
    world_locs_2d.append(loc_2d)

# get the ground truth
dataset = ObjectData(data_path,
                     config.CONFIG_BBOXES,
                     split='test',
                     file_pred_pkl='pred_railway_img_bbox_test_corrected.pkl',
                     imageset_outside=True)

with open(f'{data_path}flags/fugro-to-kitti-ids.txt', 'r') as f:
    kitti_fugro_ids = [tuple(line.strip().split(' ')) for line in f]

gt_locs = []
for data_idx in dataset.indices:
    if dataset.isexist_label_objects(data_idx):
        objects_gt = dataset.get_label_objects(data_idx)
        calib = dataset.get_calibration(data_idx)
        for obj in objects_gt:
            gt_locs.append(calib.project_rect_to_velo(np.asarray([obj.t])).reshape(-1))
    else:
        continue
gt_loc = gt_locs[0]

gt_loc_2d = np.delete(gt_loc, -1)

dists = [math.dist(gt_loc_2d, world_loc_2d) for world_loc_2d in world_locs_2d]

fig, axs = plt.subplots(1, 2, figsize=(12, 4))
axs[0].scatter(range(len(idx)), dists)
axs[0].set_xticks(range(len(idx)), range(len(idx)))
axs[0].set_xlabel('frame number')
axs[0].set_ylabel('distance to ground truth [m]')
for i, v in enumerate(dists):
    axs[0].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
# axs[0, 0].set_ylim(0, 2)
axs[1].scatter(range(len(idx)), z)
axs[1].set_xticks(range(len(idx)), range(len(idx)))
axs[1].set_xlabel('frame number')
axs[1].set_ylabel('z w.s.t. the camera [m]')
for i, v in enumerate(z):
    axs[1].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
# # axs[0, 1].set_ylim(5, 25)
# axs[1, 0].scatter(range(len(idx)), x)
# axs[1, 0].set_xticks(range(len(idx)), range(len(idx)))
# axs[1, 0].set_xlabel('frame number')
# axs[1, 0].set_ylabel('x w.s.t. the camera [m]')
# for i, v in enumerate(x):
#     if v > 0:
#         axs[1, 0].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     else:
#         axs[1, 0].annotate(str(v)[0:5], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
# axs[1, 1].scatter(range(len(idx)), y)
# axs[1, 1].set_xticks(range(len(idx)), range(len(idx)))
# axs[1, 1].set_xlabel('frame number')
# axs[1, 1].set_ylabel('y w.s.t. the camera [m]')
# for i, v in enumerate(y):
#     if v > 0:
#         axs[1, 1].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     else:
#         axs[1, 1].annotate(str(v)[0:5], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
fig.suptitle('World position offset and camera z variation for image sequence of Cabinet case 2')
plt.show(block=True)

# fig, axs = plt.subplots(2, 2, figsize=(12, 8))
# axs[0, 0].scatter(range(len(idx)), dists)
# axs[0, 0].set_xticks(range(len(idx)), range(len(idx)))
# axs[0, 0].set_xlabel('frame number')
# axs[0, 0].set_ylabel('distance to ground truth [m]')
# for i, v in enumerate(dists):
#     axs[0, 0].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
# axs[0, 0].set_ylim(0, 2)
# axs[0, 1].scatter(range(len(idx)), z)
# axs[0, 1].set_xticks(range(len(idx)), range(len(idx)))
# axs[0, 1].set_xlabel('frame number')
# axs[0, 1].set_ylabel('z w.s.t. the camera [m]')
# for i, v in enumerate(z):
#     axs[0, 1].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
# axs[0, 1].set_ylim(5, 25)
# axs[1, 0].scatter(range(len(idx)), x)
# axs[1, 0].set_xticks(range(len(idx)), range(len(idx)))
# axs[1, 0].set_xlabel('frame number')
# axs[1, 0].set_ylabel('x w.s.t. the camera [m]')
# for i, v in enumerate(x):
#     if v > 0:
#         axs[1, 0].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     else:
#         axs[1, 0].annotate(str(v)[0:5], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
# axs[1, 1].scatter(range(len(idx)), y)
# axs[1, 1].set_xticks(range(len(idx)), range(len(idx)))
# axs[1, 1].set_xlabel('frame number')
# axs[1, 1].set_ylabel('y w.s.t. the camera [m]')
# for i, v in enumerate(y):
#     if v > 0:
#         axs[1, 1].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     else:
#         axs[1, 1].annotate(str(v)[0:5], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
# plt.show(block=True)

# if len(dists) < len(output_case):
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     axs[0, 0].scatter(range(len(idx)), dists)
#     axs[0, 0].set_xticks(range(len(idx)), range(len(idx)))
#     axs[0, 0].set_xlabel('frame number')
#     axs[0, 0].set_ylabel('distance to ground truth [m]')
#     for i, v in enumerate(dists):
#         axs[0, 0].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     axs[0, 0].set_ylim(0, 2)
#     axs[0, 1].scatter(range(len(idx)), z)
#     axs[0, 1].set_xticks(range(len(idx)), range(len(idx)))
#     axs[0, 1].set_xlabel('frame number')
#     axs[0, 1].set_ylabel('z w.s.t. the camera [m]')
#     for i, v in enumerate(z):
#         axs[0, 1].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     axs[0, 1].set_ylim(5, 25)
#     axs[1, 0].scatter(range(len(idx)), x)
#     axs[1, 0].set_xticks(range(len(idx)), range(len(idx)))
#     axs[1, 0].set_xlabel('frame number')
#     axs[1, 0].set_ylabel('x w.s.t. the camera [m]')
#     for i, v in enumerate(x):
#         if v > 0:
#             axs[1, 0].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#         else:
#             axs[1, 0].annotate(str(v)[0:5], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     axs[1, 1].scatter(range(len(idx)), y)
#     axs[1, 1].set_xticks(range(len(idx)), range(len(idx)))
#     axs[1, 1].set_xlabel('frame number')
#     axs[1, 1].set_ylabel('y w.s.t. the camera [m]')
#     for i, v in enumerate(y):
#         if v > 0:
#             axs[1, 1].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#         else:
#             axs[1, 1].annotate(str(v)[0:5], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     plt.show(block=True)
# else:
#     idx = [int(data_id) for data_id in idx]
#     idx = [idx[i]-idx[0] for i in range(len(idx))]
#     fig, axs = plt.subplots(2, 2, figsize=(12, 8))
#     axs[0, 0].scatter(idx, dists)
#     axs[0, 0].set_xticks(range(len(output_case)), range(len(output_case)))
#     axs[0, 0].set_xlabel('frame number')
#     axs[0, 0].set_ylabel('distance to ground truth [m]')
#     axs[0, 0].grid()
#     # for i, v in enumerate(dists):
#     #     if v < 1:
#     #         axs[0, 0].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     # axs[0, 0].set_ylim(0, 2)
#     axs[0, 1].scatter(idx, z)
#     axs[0, 1].set_xticks(range(len(output_case)), range(len(output_case)))
#     axs[0, 1].set_xlabel('frame number')
#     axs[0, 1].set_ylabel('z w.s.t. the camera [m]')
#     axs[0, 1].grid()
#     # for i, v in enumerate(z):
#     #     if v > 20:
#     #         axs[0, 1].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     axs[0, 1].set_ylim(5, 25)
#     axs[1, 0].scatter(idx, x)
#     axs[1, 0].set_xticks(range(len(output_case)), range(len(output_case)))
#     axs[1, 0].set_xlabel('frame number')
#     axs[1, 0].set_ylabel('x w.s.t. the camera [m]')
#     axs[1, 0].grid()
#     # for i, v in enumerate(x):
#     #     if v > 0:
#     #         axs[1, 0].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     #     else:
#     #         axs[1, 0].annotate(str(v)[0:5], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     axs[1, 1].scatter(idx, y)
#     axs[1, 1].set_xticks(range(len(output_case)), range(len(output_case)))
#     axs[1, 1].set_xlabel('frame number')
#     axs[1, 1].set_ylabel('y w.s.t. the camera [m]')
#     axs[1, 1].grid()
#     # for i, v in enumerate(y):
#     #     if v > 0:
#     #         axs[1, 1].annotate(str(v)[0:4], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     #     else:
#     #         axs[1, 1].annotate(str(v)[0:5], xy=(i, v), xytext=(-5, 5), textcoords='offset points', size=8)
#     plt.show(block=True)



