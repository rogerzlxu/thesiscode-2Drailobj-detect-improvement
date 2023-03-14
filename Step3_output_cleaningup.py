import pickle
import numpy as np

# The code of Step 3 in the thesis: Detection result cleaning-up

dir_output = './data/datasets/bboxes/testing/pred/pred_railway_img_bbox_test.pkl'
# output file of SMOKE
pkl_output = pickle.load(open(dir_output, "rb"))

# set a limitation of x with [-4, 5] meters
# and select ground truths in this range
for frame in pkl_output:
    for loc in frame['location']:
        if loc[0] < -4 or loc[0] > 5 or -2 < loc[0] < 2 or loc[1] < 0 or loc[1] > 2.5 or loc[2] > 25:
            i = np.where(frame['location']==loc)[0][0]

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

# set all rotation angle as 0
for frame in pkl_output:
    for i in range(len(frame['name'])):
        if frame['rotation_y'][i] != 0:
            frame['rotation_y'][i] = 0

# save the cleaned list as a new pickle file
new_pkl_path = './data/datasets/bboxes/testing/pred/pred_railway_img_bbox_test_cleaned_z25.pkl'
with open(new_pkl_path, 'wb') as pkl:
    pickle.dump(pkl_output, pkl)

# pkl_case = pkl_output[2715:2739]  # cabinet2
# pkl_case = pkl_output[63:85]  # marker1
# pkl_case = pkl_output[955:979]  # marker3
# pkl_case = pkl_output[1992:2011]  # signal_light2
# pkl_case = pkl_output[2501:2522]  # multi cabinet


