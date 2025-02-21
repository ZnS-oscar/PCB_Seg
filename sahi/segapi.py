# import onnx
# import onnxruntime as ort
# import torch
# import numpy as np
# import cv2
# import torch.nn.functional as F
# from typing import List
import pynvml
import logging
import os
import time
from typing import List, Optional
from tqdm import tqdm
import numpy as np

from sahi.auto_model import AutoDetectionModel
from sahi.models.base import DetectionModel
from sahi.postprocess.combine import (
    GreedyNMMPostprocess,
    LSNMSPostprocess,
    NMMPostprocess,
    NMSPostprocess,
    PostprocessPredictions,
)
from sahi.prediction import ObjectPrediction, PredictionResult
from sahi.slicing import slice_image
from sahi.utils.coco import Coco, CocoImage
from sahi.utils.cv import (
    IMAGE_EXTENSIONS,
    VIDEO_EXTENSIONS,
    crop_object_predictions,
    cv2,
    get_video_reader,
    read_image_as_pil,
    visualize_object_predictions,
)
from sahi.utils.file import Path, increment_path, list_files, save_json, save_pickle
from sahi.utils.import_utils import check_requirements
import sys
print(sys.path)
from sahi.utils.segresult import ComponentResult,SegResult
POSTPROCESS_NAME_TO_CLASS = {
    "GREEDYNMM": GreedyNMMPostprocess,
    "NMM": NMMPostprocess,
    "NMS": NMSPostprocess,
    "LSNMS": LSNMSPostprocess,
}

class SEGAPI:
    def __init__(self, model_path:str, label_path:str,model_confidence_threshold:float=0.5,device:str="cuda:0"):
        kwargs={}
        self.model=AutoDetectionModel.from_pretrained(
            model_type="yolov8onnx",
            model_path=model_path,
            config_path=None,
            confidence_threshold=model_confidence_threshold,
            device=device,
            category_mapping=SEGAPI.get_label(label_path),
            category_remapping=None,
            load_at_init=False,
            image_size=None,
            **kwargs,
        )
        self.model.load_model()
        self.durations_in_seconds={}
    @staticmethod
    def get_label(label_path:str):
        # return  {'0': 'person','1': 'bicycle','2': 'car','3': 'motorcycle','4': 'airplane','5': 'bus','6': 'train','7': 'truck','8': 'boat','9': 'traffic light','10': 'fire hydrant','11': 'stop sign','12': 'parking meter','13': 'bench','14': 'bird','15': 'cat','16': 'dog','17': 'horse','18': 'sheep','19': 'cow','20': 'elephant','21': 'bear','22': 'zebra','23': 'giraffe','24': 'backpack','25': 'umbrella','26': 'handbag','27': 'tie','28': 'suitcase','29': 'frisbee','30': 'skis','31': 'snowboard','32': 'sports ball','33': 'kite','34': 'baseball bat','35': 'baseball glove','36': 'skateboard','37': 'surfboard','38': 'tennis racket','39': 'bottle','40': 'wine glass','41': 'cup','42': 'fork','43': 'knife','44': 'spoon','45': 'bowl','46': 'banana','47': 'apple','48': 'sandwich','49': 'orange','50': 'broccoli','51': 'carrot','52': 'hot dog','53': 'pizza','54': 'donut','55': 'cake','56': 'chair','57': 'couch','58': 'potted plant','59': 'bed','60': 'dining table','61': 'toilet','62': 'tv','63': 'laptop','64': 'mouse','65': 'remote','66': 'keyboard','67': 'cell phone','68': 'microwave','69': 'oven','70': 'toaster','71': 'sink','72': 'refrigerator','73': 'book','74': 'clock','75': 'vase','76': 'scissors','77': 'teddy bear','78': 'hair drier','79': 'toothbrush'}
        return {str(i): str(i) for i in range(201)}
    @staticmethod
    def convert_to_segresult(object_prediction_list):
        # print("I have to say the object_prediction_list len is ",len(object_prediction_list))
        if not object_prediction_list:
            # 如果列表为空，创建一个全零的掩码
            mask = np.zeros((1, 1), dtype=np.int32)
            return SegResult(mask)

        # 获取第一个掩码的形状作为整体掩码的形状
        mask_shape = object_prediction_list[0].mask.shape
        # 初始化整体掩码为全零
        combined_mask = np.zeros(mask_shape, dtype=np.int32)

        component_results = []
        for prediction in object_prediction_list:
            score = prediction.score.value
            category = prediction.category.id
            center_x, center_y, width, height = prediction.bbox.to_xywh()

            # 创建 ComponentResult 对象
            result = ComponentResult(
                confidence=score,
                label_ID=category,
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
                class_name="dummy_class",
                part_name="dummy_class",
                component_name="dummy_class"
            )
            component_results.append(result)

            # 更新整体掩码
            combined_mask[prediction.mask.bool_mask] = category

        # 创建 SegResult 对象
        segresult = SegResult(combined_mask, component_results)
        # print("the segresult success")
        cv2.imwrite("the mask.png",segresult.mask)
        return segresult
    # take in get_sliced_prediction
    def infer(self,image,postprocess_type:str="GREEDYNMM",postprocess_match_threshold:float=0.5,postprocess_match_metric:str="IOS",postprocess_class_agnostic:bool=False,perform_standard_pred:bool=True):

        time_start = time.time()
        slice_image_result = slice_image(
            image=image,
            output_file_name=None,
            output_dir=None,
            slice_height=None,
            slice_width=None,
            overlap_height_ratio=None,
            overlap_width_ratio=None,
            auto_slice_resolution=True,
        )
        num_slices = len(slice_image_result)
        time_end = time.time() - time_start
        self.durations_in_seconds["slice"] = time_end

        postprocess_constructor = POSTPROCESS_NAME_TO_CLASS[postprocess_type]
        postprocess = postprocess_constructor(
            match_threshold=postprocess_match_threshold,
            match_metric=postprocess_match_metric,
            class_agnostic=postprocess_class_agnostic,
        )  
        slice_height,slice_width=slice_image_result.images[0].shape[:2]
        num_batch=SEGAPI.get_available_batch_size(image_size=slice_height*slice_width*3,num_images=num_slices)
        num_group = int(num_slices / num_batch)
        verbose=2
        if verbose == 1 or verbose == 2:
            # tqdm.write(f"Performing prediction on {num_slices} slices.")
            tqdm.write(f"Performing prediction on {num_slices} slices. slice size {slice_image_result.images[0].shape}")
        object_prediction_list = []
        # perform sliced prediction
        for group_ind in range(num_group):
            # prepare batch (currently supports only 1 batch)
            image_list = []
            shift_amount_list = []
            for image_ind in range(num_batch):
                image_list.append(slice_image_result.images[group_ind * num_batch + image_ind])
                shift_amount_list.append(slice_image_result.starting_pixels[group_ind * num_batch + image_ind])
            # perform batch prediction
            prediction_result = self.get_prediction(
                images=image_list,
                shift_amount=shift_amount_list,
                full_shape=[
                    slice_image_result.original_image_height,
                    slice_image_result.original_image_width,
                ],
            )
                # convert sliced predictions to full predictions
            for prediction_result_per_image in prediction_result:
                for object_prediction in prediction_result_per_image.object_prediction_list:
                    if object_prediction:  # if not empty
                        object_prediction_list.append(object_prediction.get_shifted_object_prediction())
            print("the len of object_prediction_list is ",len(object_prediction_list))
        # perform standard prediction
        if num_slices > 1 and perform_standard_pred:
            prediction_result = self.get_prediction(
                images=[image],
                shift_amount=[0, 0],
                full_shape=[
                    slice_image_result.original_image_height,
                    slice_image_result.original_image_width,
                ],
                postprocess=None,
            )
            object_prediction_list.extend(prediction_result[0].object_prediction_list)
        # merge matching predictions
        if len(object_prediction_list) > 1:
            object_prediction_list = postprocess(object_prediction_list)
        time_end = time.time() - time_start
        self.durations_in_seconds["prediction"] = time_end

        if verbose == 2:
            print(
                "Slicing performed in",
                self.durations_in_seconds["slice"],
                "seconds.",
            )
            print(
                "Prediction performed in",
                self.durations_in_seconds["prediction"],
                "seconds.",
            )
        
        project="runs/predict"
        name="exp"
        save_dir = Path(increment_path(Path(project) / name, exist_ok=False))  # increment run
        visual_dir = save_dir / "visuals"
        relative_filepath = "1.jpg"
        output_dir = str(visual_dir / Path(relative_filepath).parent)
        print("saved in ",output_dir)
        result = visualize_object_predictions(
            np.ascontiguousarray(image),
            object_prediction_list=object_prediction_list,
            rect_th=None,
            text_size=None,
            text_th=None,
            hide_labels=False,
            hide_conf=False,
            output_dir=output_dir ,
            file_name="1",
            export_format="jpg",
        )
        # if not novisual and source_is_video:  # export video
        #     output_video_writer.write(cv2.cvtColor(result["image"], cv2.COLOR_RGB2BGR))

        # return PredictionResult(
        #     image=image, object_prediction_list=object_prediction_list, durations_in_seconds=self.durations_in_seconds
        # )
        return SEGAPI.convert_to_segresult(object_prediction_list)

    @staticmethod
    def get_available_batch_size(image_size, num_images):
        """
        检测 GPU 剩余显存，并根据输入图片大小和数量输出合适的 batch size。

        :param image_size: 单张图片的大小（以字节为单位）
        :param num_images: 图片的总数量
        :return: 合适的 batch size
        """
        try:
            # 初始化 pynvml
            pynvml.nvmlInit()

            # 获取第一个 GPU 的句柄
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            # 获取 GPU 的显存信息
            info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            free_memory = info.free

            # 考虑一定的显存预留，例如 20%
            
            available_memory = free_memory *0.8
            print("available_memory",available_memory/1024/1024)
            # 计算理论上能容纳的最大 batch size
            max_batch_size = int(available_memory / image_size/300)#经验值, 1024*1024*3需要1GB
            
            print("theory_batchsize",max_batch_size)
            # 确保 batch size 不超过图片总数
            batch_size = min(max_batch_size, num_images)

            return batch_size
        except pynvml.NVMLError as e:
            print(f"NVML error: {e}")
            return 1
        finally:
            # 关闭 pynvml
            pynvml.nvmlShutdown()

        
    def get_prediction(
        self,
        images,
        shift_amount: list = [0, 0],
        full_shape=None,
        postprocess: Optional[PostprocessPredictions] = None,
        verbose: int = 0,
    ) -> List[PredictionResult]:
        """
        Function for performing prediction for given image using given detection_model.

        Arguments:
            images: not str or np.ndarray, but list of np array
                Location of image or numpy image matrix to slice
            detection_model: model.DetectionMode
            shift_amount: List
                To shift the box and mask predictions from sliced image to full
                sized image, should be in the form of [shift_x, shift_y]
            full_shape: List
                Size of the full image, should be in the form of [height, width]
            postprocess: sahi.postprocess.combine.PostprocessPredictions
            verbose: int
                0: no print (default)
                1: print prediction duration

        Returns:
            A dict with fields:
                object_prediction_list: a list of ObjectPrediction
                durations_in_seconds: a dict containing elapsed times for profiling
        """
        durations_in_seconds = dict()

        # read image as pil
        # image_as_pil = read_image_as_pil(image)
        # get prediction
        images=[np.ascontiguousarray(read_image_as_pil(image)) for image in images]
        time_start = time.time()
        self.model.perform_inference(np.ascontiguousarray(images))
        time_end = time.time() - time_start
        durations_in_seconds["prediction"] = time_end

        # process prediction
        time_start = time.time()
        # works only with 1 batch
        self.model.convert_original_predictions(
            shift_amount=shift_amount,
            full_shape=full_shape,
        )
        # object_prediction_list: List[ObjectPrediction] = detection_model.object_prediction_list
        # print("here is object prediction list @@@",object_prediction_list.segmentation[0][0])
        # postprocess matching predictions
        # if postprocess is not None:
        #     object_prediction_list = postprocess(object_prediction_list)

        time_end = time.time() - time_start
        durations_in_seconds["postprocess"] = time_end

        if verbose == 1:
            print(
                "Prediction performed in",
                durations_in_seconds["prediction"],
                "seconds.",
            )

        return [PredictionResult(
            image=images[ind], object_prediction_list=self.model.object_prediction_list_per_image[ind], durations_in_seconds=durations_in_seconds
        ) for ind in range(len(images))]

