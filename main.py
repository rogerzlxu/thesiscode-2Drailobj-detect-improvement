import os
import argparse
import warnings

from config import config
from scripts import camera_data, utils, training_data, overlay_data_camera, visualization

def parse_args():
    parser = argparse.ArgumentParser(
        description='Overlay LiDAR with camera data or create training data for \
        3D object detection'
    )
    parser.add_argument('--cams',
                        nargs='+',
                        default=['cam2'],
                        choices=config.CAMS,
                        help='specify which cameras to include (default is middle camera)')
    parser.add_argument('--runs',
                        nargs='+',
                        default=['ep11-201002303-20190430-080329', 'ep11-201002303-20190430-093039'],
                        choices=config.VIDEO_RUNS_CONFIG.keys(),
                        help='specify which scanning runs to include (default is 1st)')
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--overlay_data',
                       action='store_true',
                       help='overlay camera and LiDAR data from specified run(s) and camera(s)')
    group.add_argument('--create_dwg_bboxes',
                       action='store_true',
                       help='Create bounding boxes DWG file')
    group.add_argument('--create_dwg_tracks',
                       action='store_true',
                       help='Create track lines DWG file')
    group.add_argument('--create_training_dataset_bboxes',
                       action='store_true',
                       help='Create dataset for SMOKE (3D object detection) in KITTI format from bounding boxes')
    group.add_argument('--create_training_dataset_tracks',
                       action='store_true',
                       help='Create dataset for Gen-LaneNet (3D object detection) in Apollo format from track lines')
    group.add_argument('--project_dwg_on_camera',
                       action='store_true',
                       help='Visualize annotations from DWG format')
    group.add_argument('--visualize_kitti_bboxes',
                       action='store_true',
                       help='Visualize bounding boxes from KITTI format')
    args = parser.parse_args()

    return args

def overlay_data(frames, lidars, camera_intrinsics, video_metadata):
    print('-------------Overlay camera and LiDAR data-------------')
    # The function that project RILA point clouds onto 2D images and visualize them
    overlay_data_camera.run(frames,
                            lidars,
                            camera_intrinsics,
                            video_metadata,
                            save_images=False,
                            output_path=os.path.join(config.ROOT_PATH_DATA, f'output/overlay/'))

    print('---------------------------------------------')

def create_dwg_bboxes():
    # The function that read the original ground truth mapping file,
    # and extract the objects of interest, then save them into a .dxf file
    print(f'---------------Create dwg file bboxes---------------')
    dwg = utils.read_dwg_file(os.path.join(config.ROOT_PATH_DATA, 'ground-truth/ground_truth_mapping.dxf'))

    dwg_new = training_data.extract_dwg_data(
        dwg, only_extract_objects=True, only_extract_rails=False
    )
    dwg_new.saveas(os.path.join(config.ROOT_PATH_DATA, 'ground-truth/bboxes.dxf'))
    print('done')

def create_training_dataset_bboxes(frames, video_metadata, camera_intrinsics):
    # The function using created .dxf ground truth file
    # to create 3D bounding boxes for the ground truths
    # information of 3D bboxes are stored in the label files of each frame
    print(f'---------------Create training dataset: KITTI format---------------')
    dwg = utils.read_dwg_file(
        os.path.join(config.ROOT_PATH_DATA, 'ground-truth/bboxes.dxf')
    )
    output_path = os.path.join(config.ROOT_PATH_DATA, f'datasets/bboxes/')

    training_data.convert_to_kitti_format(
        dwg, frames, video_metadata, camera_intrinsics,
        output_path,
        shuffle=True
    )
    print('done')


def project_dwg_on_camera(frames, camera_intrinsics, video_metadata):
    print(f'---------------Project DWG annotations to video---------------')
    dwg = utils.read_dwg_file(os.path.join(config.ROOT_PATH_DATA, 'ground-truth/bboxes.dxf'))
    # dwg = utils.read_dwg_file(os.path.join(config.ROOT_PATH_DATA, 'ground-truth/rails.dxf'))

    # function project generated 3D bounding boxes to images and visualized them
    overlay_data_camera.visualize_dwg_annotations(
        dwg, frames, camera_intrinsics, video_metadata,
        save_images=False,
        output_path=os.path.join(config.ROOT_PATH_DATA, f'visualizations/')
    )
    print('done')


def visualize_kitti_bboxes():
    print(f'---------------Project KITTI format to video---------------')
    # function project the detected bounding boxes to images
    # and make comparisons with ground truths
    data_path = os.path.join(config.ROOT_PATH_DATA, f'datasets/bboxes/')
    output_path = os.path.join(config.ROOT_PATH_DATA, f'visualizations/')

    # Test split
    # the file with predicted bboxes should be indicated in the parameter 'predictions_file'
    # if not, only ground truths will be visualized
    visualization.run(data_path,
                      split='test',
                      # predictions_file='pred_fugro_img_bbox-test-epochs72.pkl',
                      predictions_file='pred_railway_img_bbox_test_corrected.pkl',
                      save_images=True,
                      output_path=output_path)
    # Val split
    # visualization.run(data_path,
    #                   split='val',
    #                   # predictions_file='pred_fugro_img_bbox-val-epochs72.pkl',
    #                   predictions_file='pred_railway_img_bbox_val.pkl',
    #                   save_images=True,
    #                   output_path=output_path)
    # No inference was done on training split for visualization purposes,
    # so just visualize the ground truth
    # visualization.run(data_path,
    #                   split='train',
    #                   predictions_file=None,
    #                   save_images=True,
    #                   output_path=output_path)

# Press the green button in the gutter to run the script.
# you can select with function you want to run based on your requirements
if __name__ == '__main__':
    args = parse_args()
    camera_intrinsics = camera_data.get_camera_intrinsics(args.cams)
    frames = camera_data.prepare_frames(args.runs, args.cams)
    video_metadata = camera_data.get_video_metadata(args.runs, args.cams)

    lidars = utils.get_lidar_run_data(video_metadata)
    overlay_data(frames, lidars, camera_intrinsics, video_metadata)
    create_dwg_bboxes()
    create_training_dataset_bboxes(frames, video_metadata, camera_intrinsics)
    # project_dwg_on_camera(frames, camera_intrinsics, video_metadata)
    visualize_kitti_bboxes()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
