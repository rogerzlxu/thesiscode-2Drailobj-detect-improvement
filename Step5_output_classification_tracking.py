import pickle
import numpy as np
import glob
from scripts import camera_data
import math

# This script corresponds to Step 5: Output Classification
# and positioning improvement using image sequence
# Main purpose of classification is: for each prediction from model,
# classify which object they belong to, then extract all the objects
# Main purpose of positioning is, for all the predictions of each object,
# select several reliable predictions (can be selected based on its z distance)
# then take the mean of them as a final reported position

# import the detected pkl file
# use the distance between different objects to classify them
# read the output pickle file
dir_output = './data/datasets/bboxes_new/testing/pred/pred_railway_img_bbox_test_corrected.pkl'
pkl_output = pickle.load(open(dir_output, "rb"))

# extract all the objects in the pkl file,
# classify them based on object type, and put them one-by-one in a list
obj_pred_sl = []       # list for signal light boxes
obj_pred_cabinet = []  # list for cabinet boxes
obj_pred_marker = []   # list for marker boxes

for frame in pkl_output:
    for i in range(len(frame['name'])):
        if frame['name'][i] == 'Signal_light':
            obj_pred_sl.append({'name': ([frame['name'][i]]),
                                'truncated': ([frame['truncated'][i]]),
                                'occluded': ([frame['occluded'][i]]),
                                'alpha': ([frame['alpha'][i]]),
                                'bbox': ([frame['bbox'][i]]),
                                'dimensions': ([frame['dimensions'][i]]),
                                'location': ([frame['location'][i]]),
                                'rotation_y': ([frame['rotation_y'][i]]),
                                'score': ([frame['score'][i]]),
                                'sample_idx': ([frame['sample_idx'][i]]),
                                'world_location': ([frame['world_location'][i]]),
                                'object_id': ([])
                                })
        elif frame['name'][i] == 'Cabinet':
            obj_pred_cabinet.append({'name': ([frame['name'][i]]),
                                     'truncated': ([frame['truncated'][i]]),
                                     'occluded': ([frame['occluded'][i]]),
                                     'alpha': ([frame['alpha'][i]]),
                                     'bbox': ([frame['bbox'][i]]),
                                     'dimensions': ([frame['dimensions'][i]]),
                                     'location': ([frame['location'][i]]),
                                     'rotation_y': ([frame['rotation_y'][i]]),
                                     'score': ([frame['score'][i]]),
                                     'sample_idx': ([frame['sample_idx'][i]]),
                                     'world_location': ([frame['world_location'][i]]),
                                     'object_id': ([])
                                     })
        elif frame['name'][i] == 'Marker':
            obj_pred_marker.append({'name': ([frame['name'][i]]),
                                    'truncated': ([frame['truncated'][i]]),
                                    'occluded': ([frame['occluded'][i]]),
                                    'alpha': ([frame['alpha'][i]]),
                                    'bbox': ([frame['bbox'][i]]),
                                    'dimensions': ([frame['dimensions'][i]]),
                                    'location': ([frame['location'][i]]),
                                    'rotation_y': ([frame['rotation_y'][i]]),
                                    'score': ([frame['score'][i]]),
                                    'sample_idx': ([frame['sample_idx'][i]]),
                                    'world_location': ([frame['world_location'][i]]),
                                    'object_id': ([])
                                    })

def object_classify(obj_pred, threshold, num_preds):
    # function classify detected bounding boxes based on distances between them
    # if distances is close enough, those bboxes will be corresponded to the same object
    # input: obj_pred - a list contains all dictionaries of object bboxes (can be obtained from pkl files)
    # note that all bboxes in the input should correspond to a single object type!
    # create a dictionary as the output
    obj_classified = {}
    # get the type of objects
    obj_type = obj_pred[0]['name'][0]
    for o in range(len(obj_pred)):
        if len(obj_pred) == 0:
            break
        obj = obj_pred[0]
        # compute its distance with all other objects and store them in a list
        dists = []
        # compute the distance between one object to the other
        # need to first localize the objects to world coordinate
        # then compute the distance between them
        for i in range(len(obj_pred)):
            loc = obj_pred[i]['world_location'][0]
            distance = math.dist(loc, obj['world_location'][0])
            dists.append(distance)

        # use distance threshold to classify objects
        # for signal lights, set a distance threshold of 4 meters
        # for cabinets, set a distance threshold of 4 meters
        dist_close = [dist for dist in dists if dist < threshold]

        # use object_index to number the classified objects
        object_index = o + 1
        # only consider the object if it has more than two correspond bboxes
        if len(dist_close) > num_preds:
            globals()[f'{obj_type}_{object_index}'] = []

        idxs = []
        for d in range(len(dists)):
            # if bboxes are close enough, put them together in a list and give each of them a same label
            if dists[d] in dist_close:
                if len(dist_close) > num_preds:
                    if obj_pred[d]['sample_idx'][0] not in idxs:
                        obj_pred[d]['object_id'] = [f'{obj_type}_{object_index}']
                        globals()[f'{obj_type}_{object_index}'].append(obj_pred[d])
                        idxs.append(obj_pred[d]['sample_idx'][0])
                        # set all classified bboxes as 0 and delete them later
                        obj_pred[d] = 0
                    else:
                        obj_pred[d] = 0
                else:
                    # if less than 2 bboxes are classified together
                    # only set them to 0 but not put them in a list
                    obj_pred[d] = 0
        # delete the bboxes which have already been classified
        obj_pred = [obj for obj in obj_pred if obj != 0]
        # save the list as a classified object in the dictionary
        if len(dist_close) > num_preds:
            obj_classified[f'{obj_type}_{object_index}'] = globals()[f'{obj_type}_{object_index}']

        # renumber the objects,
        # since some classified objects are ignored and their numbers are not continuous
        object_num = list(obj_classified.keys())
        object_num_new = [f'{obj_type}_{i + 1}' for i in range(len(object_num))]
        # also change the labels
        for i in range(len(object_num)):
            obj_classified[object_num_new[i]] = obj_classified.pop(object_num[i])
            for j in range(len(obj_classified[object_num_new[i]])):
                obj_classified[object_num_new[i]][j]['object_id'] = object_num_new[i]

    print(len(obj_classified), f'{obj_type} are classified')
    return obj_classified

# sl_classified = object_classify(obj_pred_sl, 4, 10)
cab_classified = object_classify(obj_pred_cabinet, 3, 5)
# marker_classified = object_classify(obj_pred_marker, 4, 5)

# save the classified objects as separate files
for i in range(len(cab_classified)):
    new_pkl_path = f'./data/datasets/bboxes_multi_cabinets/testing/pred/Cabinet_{i+1}.pkl'
    with open(new_pkl_path, 'wb') as pkl:
        pickle.dump(cab_classified[f'Cabinet_{i+1}'], pkl)

# using certain conditions to get the position of certain object
# the input should be an object with all classified predictions
def get_pred_obj_loc(object_in_list):
    obj_type = object_in_list[0]['name'][0]
    if obj_type == 'Signal_light':
        obj_z = []
        obj_world_loc = []
        for obj in object_in_list:
            obj_z.append(obj['location'][0][2])
            obj_world_loc.append(obj['world_location'])

        filtered_world_loc = []
        for z in obj_z:
            if 16 < z < 22:
                index = obj_z.index(z)
                filtered_world_loc.append(obj_world_loc[index])

        filtered_world_loc_x = [loc[0][0] for loc in filtered_world_loc]
        filtered_world_loc_y = [loc[0][1] for loc in filtered_world_loc]
        filtered_world_loc_z = [loc[0][2] for loc in filtered_world_loc]

        final_position_obj = [np.mean(filtered_world_loc_x), np.mean(filtered_world_loc_y),
                              np.mean(filtered_world_loc_z)]
    elif obj_type == 'Cabinet':
        obj_z = []
        obj_world_loc = []
        for obj in object_in_list:
            obj_z.append(obj['location'][0][2])
            obj_world_loc.append(obj['world_location'])

        filtered_world_loc = []
        for z in obj_z:
            if 17 < z < 20:
                index = obj_z.index(z)
                filtered_world_loc.append(obj_world_loc[index])

        filtered_world_loc_x = [loc[0][0] for loc in filtered_world_loc]
        filtered_world_loc_y = [loc[0][1] for loc in filtered_world_loc]
        filtered_world_loc_z = [loc[0][2] for loc in filtered_world_loc]

        final_position_obj = [np.mean(filtered_world_loc_x), np.mean(filtered_world_loc_y),
                              np.mean(filtered_world_loc_z)]

    elif obj_type == 'Marker':
        obj_z = []
        obj_world_loc = []
        for obj in object_in_list:
            obj_z.append(obj['location'][0][2])
            obj_world_loc.append(obj['world_location'])

        for i in range(len(obj_z)-1):
            if obj_z[i] - obj_z[i+1] > 1:
                obj_z[i] = 0

        filtered_world_loc = []
        for z in obj_z:
            if 16 < z < 20:
                index = obj_z.index(z)
                filtered_world_loc.append(obj_world_loc[index])

        filtered_world_loc_x = [loc[0][0] for loc in filtered_world_loc]
        filtered_world_loc_y = [loc[0][1] for loc in filtered_world_loc]
        filtered_world_loc_z = [loc[0][2] for loc in filtered_world_loc]

        final_position_obj = [np.mean(filtered_world_loc_x), np.mean(filtered_world_loc_y),
                              np.mean(filtered_world_loc_z)]

    return final_position_obj


