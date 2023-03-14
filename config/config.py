from datetime import datetime

ROOT_PATH_DATA = './data/'
PATH_VIDEO = f'{ROOT_PATH_DATA}Video/'
PATH_LAZ = f'{ROOT_PATH_DATA}LAZ/'
CAMERA_CALIBRATION_FILE = f'{ROOT_PATH_DATA}Video/Calibration/3DOneCAM_20180706_201002303.xml'
# PREDICTION_PKL_FILE = f'{ROOT_PATH_DATA}mmdet3d_pred_results.pkl'

DXF_APPID = 'GROUND_TRUTH'

# training-testing split parameters
TRAIN_SPLIT_SIZE = 1
TEST_SPLIT_SIZE = 0
VAL_SPLIT_SIZE = 0

# For each folder contains frames,
# define which type of frames will be selected
# and which objects should be ignored
VIDEO_RUNS_CONFIG = {
    'ep11-201002303-20190430-075921': {
        'min': 2400,  # Min & maximum frame id to select per video run
        'max': 3721,
        # Degenerate cases, flag object to remove it from frame (list)
        # or remove it entirely (bool) when creating training data
        # ('Small [object]', 'Large [object]' means: needs novel class)
        # We remove most of the smaller markers because their bounding
        # boxes are too high. Order of bounding boxes are given in
        # order of camera frames
        'ignore_objects': {
            # bbox#88: ['frame_08888'],      # Reason to ignore object in frame(s)
            # bbox#888: True,                # Reason to ignore object entirely
            # rail#8: True,                  # Reason to ignore object entirely

            'bbox#291': True,  # Back side marker
            'bbox#126': True,  # Small cabinet
            'bbox#259': True,
            'bbox#271': True,
            'bbox#438': True,
            'bbox#444': True,  # Double marker
            'bbox#4': True,    # Back side marker
            'bbox#67': True,   # Small cabinet
            'bbox#92': True,
            'bbox#104': True,
            'bbox#385': True,
            'bbox#402': True,  # Double occluded marker
            'bbox#379': True,  # Small cabinet
            'bbox#354': True,  # Small marker
            'bbox#269': True,
            'bbox#353': True,
            'bbox#365': True,
            'bbox#398': True,   # Double back side marker
            'bbox#415': True,   # Small cabinet
            'bbox#279': True,   # Small marker
            'bbox#325': True,   # Small cabinet
            'bbox#153': True,   # Not a railroad marker
            'bbox#282': True,
            'bbox#302': True,
            'bbox#340': True,  # Double back side marker
            'bbox#201': True,
            'bbox#213': True,  # Double back side marker
            'bbox#193': ['frame_03702'],  # Cabinet behind signal
            'bbox#161': ['frame_03704'],  # Cabinet behind signal
            'bbox#97': True,
            'bbox#102': True,
            'bbox#119': True,
            'bbox#125': True,
            'bbox#171': True,
            'bbox#184': True,  # Not a railroad marker

        },
        # Degenerate cases, remove all labels from frame
        'ignore_frames': [
            # 'frame_88888',      # Reason to ignore all labels from frame
            'frame_02901',
            'frame_02902',
            'frame_02903',
            'frame_02904',
            'frame_02905',
            'frame_02906',
            'frame_02907',
            'frame_02908',
            'frame_02909',
            'frame_02910',
            'frame_02911',
            'frame_02912',
            'frame_02913',
            'frame_02914',
            'frame_02915'         # Occluded object
        ],
    },
    'ep11-201002303-20190430-080329': {
        'min': 0,
        'max': 3342,
        'ignore_objects': {
            'bbox#125': True,  # Double marker
            'bbox#102': True,  # Double marker
            'bbox#97': True,   # Double marker
            'bbox#119': True,  # Double marker
            'bbox#230': True,  # back side marker
            'bbox#422': True,  # Cabinet behind vegetation
            'bbox#401': True,  # Cabinet behind vegetation
            'bbox#30': True,   # back side marker
            'bbox#196': True,  # Small marker
            'bbox#315': True,  # Small marker
            'bbox#290': True,  # Small signal light
            'bbox#121': True,  # Small marker
            'bbox#395': True,  # Occluded marker
            'bbox#272': True,  # Occluded cabinet
            'bbox#330': True,  # Occluded signal light
            'bbox#287': True,  # Occluded cabinet
            'bbox#449': True,  # Small marker
            'bbox#244': True,  # Small marker
            'bbox#35': True,   # back side marker
            'bbox#268': True,  # Occluded cabinet
            'bbox#2': True,  # Small signal light
            'bbox#174': True,  # Occluded cabinet
            'bbox#360': True,  # Not a railroad signal light
            'bbox#233': ['frame_02375',  # Occluded marker
                         'frame_02376',
                         'frame_02377',
                         'frame_02378',
                         'frame_02379',
                         'frame_02380'],
            'bbox#33': True,   # Occluded cabinet
            'bbox#52': True,   # Occluded cabinet
            'bbox#304': True,  # Occluded cabinet
            'bbox#366': True,  # Occluded cabinet
            'bbox#375': True,  # Not a railroad marker
            'bbox#74': True,   # Large cabinet
            'bbox#77': True,   # Small cabinet
            'bbox#256': ['frame_02422',
                         'frame_02423',
                         'frame_02424'],  # Cabinet behind marker
            'bbox#183': True,  # Large cabinet
            'bbox#388': True,  # Large cabinet
            'bbox#336': True,  # Occluded cabinet
            'bbox#266': True,  # Small cabinet
            'bbox#68': True,  # Small marker
            'bbox#95': True,  # Back side marker
            'bbox#450': True,  # Small marker
            'bbox#229': True,  # Small marker
            'bbox#217': True,  # Small marker
            'bbox#219': True,  # Small marker
            'bbox#22': ['frame_02791',
                        'frame_02792',
                        'frame_02793',
                        'frame_02794'],  # Cabinet behind marker
            'bbox#70': True,  # Small marker
            'bbox#355': True,  # Small marker
            'bbox#78': True,  # Small marker
            'bbox#157': True,  # Small marker
            'bbox#429': True,  # Not a railroad marker
            'bbox#226': ['frame_03294',
                         'frame_03295'],  # Cabinet behind cabinet

        },
        'ignore_frames': [],
    },
    'ep11-201002303-20190430-080712': {
        'min': 0,
        'max': 2922,
        'ignore_objects': {
            'bbox#421': True,  # Small cabinet
            'bbox#152': True,  # Small marker
            'bbox#284': True,  # Not a railroad marker
            'bbox#295': True,  # Not a railroad signal light
            'bbox#297': True,  # Not a railroad cabinet
            'bbox#311': True,  # Not a railroad marker
            'bbox#321': True,  # Not a railroad marker
            'bbox#307': True,  # Not a railroad marker
            'bbox#19': True,  # Not a railroad marker
            'bbox#331': True,  # Not a railroad signal light
            'bbox#139': True,  # Not a railroad cabinet
            'bbox#273': True,  # Large cabinet
            'bbox#343': True,  # Not a railroad signal light
            'bbox#347': True,  # Not a railroad marker
            'bbox#164': True,  # Large cabinet
            'bbox#175': True,  # Not a railroad signal light
            'bbox#417': True,  # Back side marker
            'bbox#158': True,  # Cabinet behind vegetation
            'bbox#368': True,  # Unclear cabinet
            'bbox#363': ['frame_00585',
                         'frame_00586',
                         'frame_00587'],  # Occluded cabinet
            'bbox#243': True,  # Small signal light
            'bbox#83': True,   # Small marker
            'bbox#253': True,  # Small marker
            'bbox#12': True,  # Small marker
            'bbox#31': True,  # Occluded cabinet
            'bbox#43': True,  # Occluded cabinet
            'bbox#61': True,  # Occluded cabinet
            'bbox#75': True,  # Large cabinet
            'bbox#151': True,  # Back side marker
            'bbox#176': True,  # Back side marker
            'bbox#440': True,  # Back side marker
            'bbox#203': True,  # Small cabinet
            'bbox#257': True,  # Small cabinet
            'bbox#338': True,  # Occluded cabinet
            'bbox#34': True,  # Large cabinet
            'bbox#23': ['frame_00773',
                        'frame_00774',
                        'frame_00775',
                        'frame_00776',
                        'frame_00777'],  # Occluded cabinet
            'bbox#32': True,  # Small marker
            'bbox#47': True,  # Double marker
            'bbox#60': True,  # Double marker
            'bbox#122': True,  # Not a railroad marker
            'bbox#132': True,  # Not a railroad marker
            'bbox#197': True,  # Not a railroad signal light
            'bbox#361': True,  # Not a railroad marker
            'bbox#369': True,  # Not a railroad marker
            'bbox#178': True,  # Not a railroad signal light
            'bbox#133': True,  # Not a railroad signal light
            'bbox#130': True,  # Not a railroad signal light
            'bbox#160': True,  # Large cabinet
            'bbox#147': True,  # Large cabinet
            'bbox#72': ['frame_01015',
                        'frame_01016',
                        'frame_01017',
                        'frame_01018',
                        'frame_01019',
                        'frame_01020'],  # Occluded cabinet
            'bbox#73': True,  # Small cabinet
            'bbox#79': True,  # Small cabinet
            'bbox#358': True,  # Small marker
            'bbox#76': True,  # Small marker
            'bbox#182': True,  # Small cabinet
            'bbox#418': True,  # Occluded cabinet
            'bbox#362': True,  # Back side marker
            'bbox#136': True,  # Small signal light
            'bbox#334': True,  # Not a railroad marker
            'bbox#124': True,  # Small marker
            'bbox#7': True,  # Small marker
            'bbox#51': True,  # Small marker
            'bbox#71': True,  # Not a railroad marker
            'bbox#25': True,  # Not a railroad marker
            'bbox#285': True,  # Not a railroad marker
            'bbox#293': True,  # Not a railroad marker
            'bbox#308': True,  # Not a railroad marker
            'bbox#377': True,  # Not a railroad marker
            'bbox#384': True,  # Not a railroad marker
            'bbox#400': True,  # Not a railroad marker
            'bbox#381': True,  # Small cabinet
            'bbox#425': True,  # Small marker
            'bbox#39': True,  # Occluded cabinet
            'bbox#134': True,  # Occluded cabinet
            'bbox#431': True,  # Occluded marker
            'bbox#16': True,  # Occluded cabinet
            'bbox#255': True,  # Occluded marker
            'bbox#261': True,  # Small cabinet
            'bbox#251': True,  # Small cabinet
            'bbox#90': True,  # Back side marker
            'bbox#29': True,  # Small cabinet
            'bbox#8': True,  # Double marker
            'bbox#346': True,  # Double marker
            'bbox#84': True,  # Not a railroad marker
            'bbox#420': True,  # Large cabinet
            'bbox#348': True,  # Not a railroad cabinet
            'bbox#91': True,  # Small marker
            'bbox#55': True,  # Not a railroad signal light
            'bbox#352': True,  # Not a railroad signal light
            'bbox#286': True,  # Large cabinet
            'bbox#111': True,  # Not a railroad signal light
            'bbox#276': True,  # Not a railroad signal light
            'bbox#305': True,  # Not a railroad marker
            'bbox#49': True,  # Double marker
            'bbox#57': True,  # Double marker
            'bbox#140': True,  # Small cabinet
            'bbox#446': True,  # Small marker
            'bbox#116': True,  # Small marker
            'bbox#221': True,  # Small cabinet
            'bbox#225': True,  # Small marker


        },
        'ignore_frames': [],
    },
    'ep11-201002303-20190430-093039': {
        'min': 0,
        'max': 1450,
        'ignore_objects': {
            'bbox#152': True,  # Small marker
            'bbox#421': True,  # Small cabinet
            'bbox#198': ['frame_00417',
                         'frame_00418',
                         'frame_00419',
                         'frame_00420',
                         'frame_00421',
                         'frame_00422',
                         'frame_00423',
                         'frame_00424',
                         'frame_00425',
                         'frame_00426',
                         'frame_00427'],  # Occluded cabinet
            'bbox#82': ['frame_00980',
                        'frame_00981',
                        'frame_00982',
                        'frame_00983',
                        'frame_00984',
                        'frame_00985',
                        'frame_00986',
                        'frame_00987',
                        'frame_00988',
                        'frame_00989',
                        'frame_00990'],  # Occluded cabinet
            'bbox#108': ['frame_00985',
                         'frame_00986',
                         'frame_00987',
                         'frame_00988',
                         'frame_00989',
                         'frame_00990',
                         'frame_00991',
                         'frame_00992',
                         'frame_00993',
                         'frame_00994'],  # Occluded cabinet
            'bbox#22': ['frame_00990',
                        'frame_00991',
                        'frame_00992',
                        'frame_00993',
                        'frame_00994',
                        'frame_00995',
                        'frame_00996'],  # Occluded cabinet
            'bbox#443': True,  # Back side marker
            'bbox#89': True,  # Back side marker
            'bbox#68': True,  # Back side marker
            'bbox#21': ['frame_01337',
                        'frame_01338',
                        'frame_01339'],  # Cabinet behind marker
            'bbox#110': True,  # Small marker
            'bbox#41': True,  # Occluded cabinet

            'bbox#174': True,  # Occluded cabinet
            'bbox#360': True,  # Not a railroad signal light
            'bbox#233': True,  # Back side marker
            'bbox#33': True,   # Occluded cabinet
            'bbox#52': True,   # Occluded cabinet
            'bbox#304': True,  # Occluded cabinet
            'bbox#366': True,  # Occluded cabinet
            'bbox#375': True,  # Not a railroad marker
            'bbox#74': True,   # Large cabinet
            'bbox#77': True,   # Small cabinet
            'bbox#183': True,  # Large cabinet
            'bbox#388': True,  # Large cabinet
            'bbox#336': True,  # Occluded cabinet
            'bbox#266': True,  # Small cabinet
            'bbox#450': True,  # Small marker
            'bbox#229': True,  # Small marker
            'bbox#217': True,  # Small marker
            'bbox#219': True,  # Small marker
            'bbox#70': True,  # Small marker
            'bbox#355': True,  # Small marker
            'bbox#78': True,  # Small marker
            'bbox#157': True,  # Small marker
            'bbox#429': True,  # Not a railroad marker
        },
        'ignore_frames': [],
    },
}

LIDAR_RUNS = ['2019-04-30_b_9', '2019-04-30_b_10']
DATE_DATA = datetime(2019, 4, 30)
CAMS = ['cam1', 'cam2', 'cam3']
CONFIG_BBOXES = {  # Configure bounding box around object type
    'SU-TR-Rails-Z': {
        'name': 'Rails',
        'height': 1,
        'resize_factor': 1,
        'color': (255, 0, 255),
        'color_text': (0, 0, 0),
    },
    'TL-TL-Signal light-Z': {
        'name': 'Signal_light',
        'height': 5,
        'resize_factor': 20,
        'color': (0, 255, 0),
        'color_text': (0, 0, 0),
    },
    'SU-TR-Markers-Z': {
        'name': 'Marker',  # Signs
        'height': 2.5,
        'resize_factor': 15,  # DWG circle is smaller than bbox, so add resize factor
        'color': (255, 0, 0),
        'color_text': (0, 0, 0),
    },
    'SU-UT-Cabinet-Z': {
        'name': 'Cabinet',
        'height': 2,
        'resize_factor': 5,
        'color': (246, 190, 0),
        'color_text': (0, 0, 0),
    },
}

Z_CORRECTION = 0
# Z_CORRECTION = -0.020
CLOSEBY_DISTANCE = 3
FARAWAY_DISTANCE = 30

LASPY_FILE_VERSION = '1.4'
LASPY_POINT_FORMAT = 6


def get_video_metadata_path(video_run_name, camera_name):
    return f'{ROOT_PATH_DATA}Video/externalOrientation/{video_run_name}/{camera_name}.mjpg_EO_InScope.csv'