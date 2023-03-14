This folder contains the scripts I developed and used in my MSc thesis: Rail Corridor Object Detection and Positioning Improvement from a Railway Mobile Mapping System
For the thesis please check: https://repository.tudelft.nl/islandora/object/uuid%3Af9755750-36b6-451f-a733-7c7ad9448330?collection=education

The scripts of my project is developed based on a former work based on RILA, for this project, please check:
https://git.science.uu.nl/a.f.bundel/thesis-code-3d-reconstruction

For the scripts in the main folder:
- Step3_output_cleaningup.py: This corresponds the work in Step 3 - Detection result cleaning-up
- Step4_output_distortion_correction.py: This corresponds to Step 4 - Distortion correction for detection output
- Step5_output_classification_tracking.py: This corresponds to Step 5 - Output classification and positioning improvement

- fugro_data_adjustment.py, fugro_trj_csv_preparation.py, input_preparation_smoke.py: these three scripts are mainly used for dataset preprocessing
fugro_data_adjustment.py: This is used to adjust the name of RILA video frames
fugro_trj_csv_preparation.py: This is used to adjust the generated trajectory file, to delete the useless information and adjust the required information so that it can be read by the developed scripts
input_preparation_smoke.py: This is used to adjust the structure of input dataset

- output_analysis.py, case_study.py are used to make analysis on the results of Step345.
- main.py defined some essential functions to preparing datasets and visualize results, the folder 'scripts' gives the functions and classes support the run of main.py

- The folder 'config' contains the script with some predefined parameters, such as which camera to be used, and the predefined dimensions of ground truth bounding boxes
- The folder 'data' contains the datasets used for this project
The structure of 'data' is:
data
  ├──datasets
 |          └──bboxes
 |          │──flags - 000000.txt ...
 |          |        - fugro-to-kitti-ids.txt
 |          | 
 |          |──ImageSets - test.txt
 |          |            - train.txt
 |          |            - trainval.txt
 |          |            - val.txt
 |                    │──training
 |                    │    ├──calib - 000000.txt ...
 |                    │    ├──label_2 - 000000.txt ... 
 |                    │    ├──image_2 - 000000.jpg ...
 |                    │    └──velodyne - 000000.bin ...
 |                    └──testing
 |               ├──calib - 000000.txt ...
 |               ├──image_2 - 000000.jpg ...
 |               ├──label_2 - 000000.txt ...
 |               └──velodyne - 000000.bin ...
 |
  ├── ground-truth - bboxes.dxf
 |
 ├── LAZ
 |    │── 2019-04-30_b_9 - Tilexxx.laz
 |    └── 2019-04-30_b_10 - Tilexxx.laz
 |
 ├── Videos
 |     │──Calibration - 3DOneCAM_20180706_201002303.xml
  |          │──ep11-201002303-20190430-075921 - 000000.jpg
 |     │──ep11-201002303-...
 |     └──externalOrientation
  |                    │──ep11-201002303-20190430-075921
  |                    │          │──cam1.csv
  |          │          │──cam2.csv
  |          │          └──cam3.csv
  |                    └──ep11-201002303-...
 |
 └──visualizations

In addition, for Step 2: Recognize objects from 2D images using SMOKE, the work is finished by the jupyter notebook: Step2_MMDetection3D_Rail.ipynb
This notebook is developed from the online platform MMDetection3D. To make it running, first it should be uploaded to google drive and open with Colab. Later, the folder 'bboxes' in 'data' above should be zipped and renamed as 'dataset_3d_object_detection.zip', and upload to google drive.