import os
os.getcwd()
# arrange an instance segmentation model for test
from sahi.utils.yolov8onnx import download_yolov8n_onnx_model

from sahi import AutoDetectionModel
from sahi.utils.cv import read_image
from sahi.utils.file import download_from_url
from sahi.predict import get_prediction, get_sliced_prediction, predict
from IPython.display import Image
yolov8_onnx_model_path = "demo/models/yolov8n.onnx"
yolo11n_model_path = "demo/models/yolov8n-seg.onnx"
category_mapping = {'0': 'person','1': 'bicycle','2': 'car','3': 'motorcycle','4': 'airplane','5': 'bus','6': 'train','7': 'truck','8': 'boat','9': 'traffic light','10': 'fire hydrant','11': 'stop sign','12': 'parking meter','13': 'bench','14': 'bird','15': 'cat','16': 'dog','17': 'horse','18': 'sheep','19': 'cow','20': 'elephant','21': 'bear','22': 'zebra','23': 'giraffe','24': 'backpack','25': 'umbrella','26': 'handbag','27': 'tie','28': 'suitcase','29': 'frisbee','30': 'skis','31': 'snowboard','32': 'sports ball','33': 'kite','34': 'baseball bat','35': 'baseball glove','36': 'skateboard','37': 'surfboard','38': 'tennis racket','39': 'bottle','40': 'wine glass','41': 'cup','42': 'fork','43': 'knife','44': 'spoon','45': 'bowl','46': 'banana','47': 'apple','48': 'sandwich','49': 'orange','50': 'broccoli','51': 'carrot','52': 'hot dog','53': 'pizza','54': 'donut','55': 'cake','56': 'chair','57': 'couch','58': 'potted plant','59': 'bed','60': 'dining table','61': 'toilet','62': 'tv','63': 'laptop','64': 'mouse','65': 'remote','66': 'keyboard','67': 'cell phone','68': 'microwave','69': 'oven','70': 'toaster','71': 'sink','72': 'refrigerator','73': 'book','74': 'clock','75': 'vase','76': 'scissors','77': 'teddy bear','78': 'hair drier','79': 'toothbrush'}
detection_model = AutoDetectionModel.from_pretrained(
    model_type='yolov8onnx',
    model_path=yolov8_onnx_model_path,
    confidence_threshold=0.3,
    category_mapping=category_mapping,
    device="cuda:0", # or 'cuda:0'
)
# result = get_prediction(read_image("demo_data/small-vehicles1.jpeg"), detection_model)
result = get_sliced_prediction(
    "demo_data/small-vehicles1.jpeg",
    # "demo_data/pcbsmall1.png",
    detection_model,
    slice_height = 256,
    slice_width = 256,
    overlap_height_ratio = 0.2,
    overlap_width_ratio = 0.2
)
result.export_visuals(export_dir="demo_data/")
