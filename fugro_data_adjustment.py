import os
import glob
import pandas as pd
import numpy as np

# This script is used for adjust the names of extracted images from mjpg
# also rewrite the correspond trajectory files to the same format as the original dataset
# The extract name is image_0001, will rename them to names like frame_00001
dir_images = 'F://zelin.xu_account//Documents//My_data//20190430_2D//ep11-201002303-20190430-093039//cam2//'
list_images = glob.glob(dir_images + '*.jpg')

# rename from image_8888 to frame_88888
for image in list_images:
    if len(str(int(image[-8:-4]))) == 1:
        old_name = image
        new_name = dir_images + 'frame_0000' + str(int(image[-8:-4])) + '.jpg'
        os.rename(old_name, new_name)
    elif len(str(int(image[-8:-4]))) == 2:
        old_name = image
        new_name = dir_images + 'frame_000' + str(int(image[-8:-4])) + '.jpg'
        os.rename(old_name, new_name)
    elif len(str(int(image[-8:-4]))) == 3:
        old_name = image
        new_name = dir_images + 'frame_00' + str(int(image[-8:-4])) + '.jpg'
        os.rename(old_name, new_name)
    else:
        old_name = image
        new_name = dir_images + 'frame_0' + str(int(image[-8:-4])) + '.jpg'
        os.rename(old_name, new_name)
        print(old_name, 'has been renamed to', new_name)

# rewrite the trajectory .csv files
# so that it can be read by ./scripts/camera_data.py
metadata_cam2 = pd.read_csv('F://zelin.xu_account//Documents//My_data//20190430_2D//3DoneCAM//ep11-201002303-20190430-093039//cam2.csv')

GpsTime = metadata_cam2['GpsTime'] - 1240444653.799897
Easting = metadata_cam2['Easting']
Northing = metadata_cam2['Northing']
Elevation = metadata_cam2['Elevation']
Roll = np.degrees(metadata_cam2['Roll'])
Pitch = np.degrees(metadata_cam2['Pitch'])
Heading = np.degrees(metadata_cam2['Heading'])

# creating frame id
frame_id = []

for i in range(metadata_cam2.shape[0]):
    if len(str(i)) == 1:
        frame_id.append('frame_0000' + str(i))
    elif len(str(i)) == 2:
        frame_id.append('frame_000' + str(i))
    elif len(str(i)) == 3:
        frame_id.append('frame_00' + str(i))
    else:
        frame_id.append('frame_0' + str(i))

frame_id = pd.Series(frame_id)
frame_id.name = 'frame_id'

metadata_cam2_new = pd.concat([frame_id, GpsTime, Easting, Northing, Elevation, Roll, Pitch, Heading], axis=1)

# metadata_cam2_new.to_csv('F://zelin.xu_account//Documents//My_data//20190430_2D//cam2.mjpg_EO_InScope.csv', index=False)
