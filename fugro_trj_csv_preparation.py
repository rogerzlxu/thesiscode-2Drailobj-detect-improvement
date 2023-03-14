import numpy as np
import pandas as pd

# rewrite the video metadata file cam2.csv
# so that it can be read by ./scripts/camera_data.py

metadata_cam2 = pd.read_csv('./data/Video/externalOrientation/ep11-201002303-20190430-075921/cam2.csv')

GpsTime = metadata_cam2['GpsTime']
Easting = metadata_cam2['Easting']
Northing = metadata_cam2['Northing']
Elevation = metadata_cam2['Elevation']
Roll = np.degrees(metadata_cam2['Roll'])
Pitch = np.degrees(metadata_cam2['Pitch'])
Heading = np.degrees(metadata_cam2['Heading'])

# creating frame id
frame_id = []

# when the frame id with the format: image_8888
# for i in range(metadata_cam2.shape[0]):
#     if len(str(i)) == 1:
#         frame_id.append('image_000' + str(i))
#     elif len(str(i)) == 2:
#         frame_id.append('image_00' + str(i))
#     elif len(str(i)) == 3:
#         frame_id.append('image_0' + str(i))
#     else:
#         frame_id.append('image_' + str(i))

# when the frame id with the format: frame_88888
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

# metadata_cam2_new.to_csv('./data/Video/externalOrientation/ep11-201002303-20190430-075921/cam2.mjpg_EO_InScope.csv', index=False)