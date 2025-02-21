# OBSS SAHI Tool
# Code written by Karl-Joan Alesma and Michael García, 2023.

import logging
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
import time
logger = logging.getLogger(__name__)

from sahi.models.base import DetectionModel
from sahi.prediction import ObjectPrediction
from sahi.utils.compatibility import fix_full_shape_list, fix_shift_amount_list
from sahi.utils.import_utils import check_requirements
from sahi.utils.yolov8onnx import non_max_supression, xywh2xyxy
from sahi.utils.cv import get_coco_segmentation_from_bool_mask, get_coco_segmentation_from_obb_points

class Yolov8OnnxDetectionModel(DetectionModel):
    def __init__(self, *args, iou_threshold: float = 0.7, **kwargs):
        """
        Args:
            iou_threshold: float
                IOU threshold for non-max supression, defaults to 0.7.
        """
        super().__init__(*args, **kwargs)
        self.iou_threshold = iou_threshold

    def check_dependencies(self) -> None:
        check_requirements(["onnxruntime"])

    def load_model(self, ort_session_kwargs: Optional[dict] = {}) -> None:
        """Detection model is initialized and set to self.model.

        Options for onnxruntime sessions can be passed as keyword arguments.
        """

        import onnxruntime
        # print("Available providers:", onnxruntime.get_available_providers())

        try:
            if self.device == torch.device("cpu"):
                EP_list = ["CPUExecutionProvider"]
            else:
                EP_list = ["CUDAExecutionProvider"]

            options = onnxruntime.SessionOptions()

            for key, value in ort_session_kwargs.items():
                setattr(options, key, value)

            ort_session = onnxruntime.InferenceSession(self.model_path, sess_options=options, providers=EP_list)

            self.set_model(ort_session)

        except Exception as e:
            raise TypeError("model_path is not a valid onnx model path: ", e)

    def set_model(self, model: Any) -> None:
        """
        Sets the underlying ONNX model.

        Args:
            model: Any
                A ONNX model
        """

        self.model = model

        # set category_mapping
        if not self.category_mapping:
            raise TypeError("Category mapping values are required")

    def _preprocess_image(self, images: Optional[List[np.ndarray]], input_shape: Optional[Tuple[int, int]]) -> np.ndarray:
        """Prepapre image for inference by resizing, normalizing and changing dimensions.

        Args:
            image: np.ndarray
                Input image with color channel order RGB.
        """
        image_tensor_list=[]
        for image in images:
            input_image = cv2.resize(image, input_shape)

            input_image = input_image / 255.0
            input_image = input_image.transpose(2, 0, 1)
            image_tensor = input_image[np.newaxis, :, :, :].astype(np.float32)
            image_tensor_list.append(image_tensor)
        merged_array = np.concatenate(image_tensor_list, axis=0)
        return merged_array

    def _post_process_ori(
        self, outputs: np.ndarray, input_shape: Tuple[int, int], image_shape: Tuple[int, int]
    ) -> List[torch.Tensor]:
        image_h, image_w = image_shape
        input_w, input_h = input_shape

        predictions = np.squeeze(outputs[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.confidence_threshold, :]
        scores = scores[scores > self.confidence_threshold]
        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = predictions[:, :4]

        # Scale boxes to original dimensions
        input_shape = np.array([input_w, input_h, input_w, input_h])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([image_w, image_h, image_w, image_h])
        boxes = boxes.astype(np.int32)

        # Convert from xywh two xyxy
        boxes = xywh2xyxy(boxes).round().astype(np.int32)

        # Perform non-max supressions
        indices = non_max_supression(boxes, scores, self.iou_threshold)

        # Format the results
        prediction_result = []
        for bbox, score, label in zip(boxes[indices], scores[indices], class_ids[indices]):
            bbox = bbox.tolist()
            cls_id = int(label)
            prediction_result.append([bbox[0], bbox[1], bbox[2], bbox[3], score, cls_id])

        prediction_result = [torch.tensor(prediction_result)]
        # prediction_result = [prediction_result]

        return prediction_result
    @staticmethod
    def intersection(box1,box2):
        box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
        box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
        x1 = max(box1_x1,box2_x1)
        y1 = max(box1_y1,box2_y1)
        x2 = min(box1_x2,box2_x2)
        y2 = min(box1_y2,box2_y2)
        return (x2-x1)*(y2-y1) 
    @staticmethod
    def union(box1,box2):
        box1_x1,box1_y1,box1_x2,box1_y2 = box1[:4]
        box2_x1,box2_y1,box2_x2,box2_y2 = box2[:4]
        box1_area = (box1_x2-box1_x1)*(box1_y2-box1_y1)
        box2_area = (box2_x2-box2_x1)*(box2_y2-box2_y1)
        return box1_area + box2_area - Yolov8OnnxDetectionModel.intersection(box1,box2)
    @staticmethod
    def iou(box1,box2):
        return Yolov8OnnxDetectionModel.intersection(box1,box2)/Yolov8OnnxDetectionModel.union(box1,box2)
    # @staticmethod
    # def get_mask(batch_rows, batch_boxes, img_widths, img_heights):
    #     def sigmoid(x):
    #         return 1 / (1 + np.exp(-x))
    #     # batch_size = len(batch_rows)
    #     # not batched temp solution
    #     batch_size=1
    #     img_widths, img_heights= [img_widths], [img_heights]

    #     masks = []

    #     # 重塑并应用 sigmoid 函数
    #     batch_masks = batch_rows.reshape(batch_size, 160, 160)
    #     batch_masks = sigmoid(batch_masks)
    #     batch_masks = (batch_masks > 0.5).astype("uint8") * 255

    #     for i in range(batch_size):
    #         row = batch_masks[i]
    #         box = batch_boxes[i]
    #         img_width = img_widths[i]
    #         img_height = img_heights[i]

    #         # crop the object defined by "box" from mask
    #         x1, y1, x2, y2 = box
    #         mask_x1 = round(x1 / img_width * 160)
    #         mask_y1 = round(y1 / img_height * 160)
    #         mask_x2 = round(x2 / img_width * 160)
    #         mask_y2 = round(y2 / img_height * 160)

    #         # 裁剪掩码
    #         cropped_mask = row[mask_y1:mask_y2, mask_x1:mask_x2]

    #         # resize the cropped mask to the size of object
    #         img_mask = Image.fromarray(cropped_mask, "L")
    #         img_mask = img_mask.resize((round(x2 - x1), round(y2 - y1)))
    #         mask = np.array(img_mask)

    #         masks.append(mask)

    #     return np.array(masks)

    # parse segmentation mask
    @staticmethod
    def get_mask(row, box, img_width, img_height):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        # convert mask to image (matrix of pixels)
        mask = row.reshape(160,160)
        mask = sigmoid(mask)
        mask = (mask > 0.5).astype("uint8")*255
        # crop the object defined by "box" from mask
        x1,y1,x2,y2 = box
        x1,y1,x2,y2=x1.item(),y1.item(),x2.item(),y2.item()
        mask_x1 = round(x1/img_width*160)
        mask_y1 = round(y1/img_height*160)
        mask_x2 = round(x2/img_width*160)
        mask_y2 = round(y2/img_height*160)
        mask = mask[mask_y1:mask_y2,mask_x1:mask_x2]
        # resize the cropped mask to the size of object
        img_mask = Image.fromarray(mask,"L")
        img_mask = img_mask.resize((round(x2-x1),round(y2-y1)))
        mask = np.array(img_mask)
        return mask
    # def _post_process(self,outputs, img_widths, img_heights, input_size=640):
    #     batch_size = outputs[0].shape[0]
    #     output0 = outputs[0].transpose(0, 2, 1)  # (batch_size, 8400, 116)
    #     output1 = outputs[1]  # (batch_size, 32, 160, 160)

    #     # 提取边界框和类别分数
    #     boxes = output0[..., :84]  # (batch_size, 8400, 84)
    #     masks_coeffs = output0[..., 84:]  # (batch_size, 8400, 32)

    #     output1 = output1.reshape(batch_size, 32, -1)  # (batch_size, 32, 160*160)
    #     masks = np.einsum('bik,bkj->bij', masks_coeffs, output1)  # (batch_size, 8400, 160*160)

    #     boxes = np.concatenate([boxes, masks], axis=-1)  # (batch_size, 8400, 84 + 160*160)

    #     img_widths = np.array([img_widths])[:, np.newaxis, np.newaxis]
    #     img_heights = np.array([img_heights])[:, np.newaxis, np.newaxis]

    #     # 提取边界框中心坐标和宽高
    #     xc = boxes[..., 0:1]
    #     yc = boxes[..., 1:2]
    #     w = boxes[..., 2:3]
    #     h = boxes[..., 3:4]

    #     # 计算边界框左上角和右下角坐标
    #     x1 = (xc - w / 2) / input_size * img_widths
    #     y1 = (yc - h / 2) / input_size * img_heights
    #     x2 = (xc + w / 2) / input_size * img_widths
    #     y2 = (yc + h / 2) / input_size * img_heights

    #     # 提取类别分数并找到最大分数和对应类别索引
    #     class_scores = boxes[..., 4:84]
    #     probs = np.max(class_scores, axis=-1, keepdims=True)  # (batch_size, 8400, 1)
    #     class_ids = np.argmax(class_scores, axis=-1)  # (batch_size, 8400)

    #     # 筛选置信度高于阈值的检测结果
    #     valid_mask = probs[..., 0] >= 0.5

    #     prediction_results = []
    #     for batch_idx in range(batch_size):
    #         valid_indices = np.where(valid_mask[batch_idx])[0]
    #         valid_boxes = np.hstack([
    #             x1[batch_idx][valid_indices],
    #             y1[batch_idx][valid_indices],
    #             x2[batch_idx][valid_indices],
    #             y2[batch_idx][valid_indices],
    #             # np.array([yolo_classes[class_id] for class_id in class_ids[batch_idx][valid_indices]])[:, np.newaxis],
    #             probs[batch_idx][valid_indices],
    #             class_ids[batch_idx][valid_indices][:, np.newaxis],
    #         ])

    #         valid_masks = boxes[batch_idx, valid_indices, 84:]
    #         valid_masks = ([
    #             Yolov8OnnxDetectionModel.get_mask(mask_info, (x1_, y1_, x2_, y2_), img_widths[batch_idx, 0, 0], img_heights[batch_idx, 0, 0])
    #             for mask_info, x1_, y1_, x2_, y2_ in zip(valid_masks,
    #                                                     x1[batch_idx][valid_indices],
    #                                                     y1[batch_idx][valid_indices],
    #                                                     x2[batch_idx][valid_indices],
    #                                                     y2[batch_idx][valid_indices])
    #         ])

    #         # 应用非极大值抑制 但是这里不用.nms
    #         # sorted_indices = np.argsort(valid_boxes[:, -2].astype(float))[::-1]
    #         # valid_boxes = valid_boxes[sorted_indices]
    #         # valid_masks = valid_masks[sorted_indices]
    #         # 原有的排序索引获取
    #         sorted_indices = np.argsort(valid_boxes[:, -1].astype(float))[::-1]

    #         # 将 valid_boxes 和 valid_masks 组合成一个列表
    #         # combined = list(zip(valid_boxes, valid_masks))

    #         # 根据 sorted_indices 对组合列表进行排序
    #         # sorted_combined = [combined[i] for i in sorted_indices]

    #         # 分离排序后的 valid_boxes 和 valid_masks
    #         # valid_boxes = np.array([box for box, _ in sorted_combined])
    #         # valid_masks = [mask for _, mask in sorted_combined]

    #         result_boxes = []
    #         result_masks = []
    #         while len(valid_boxes) > 0:
    #             result_boxes.append(valid_boxes[0])
    #             result_masks.append(valid_masks[0])
    #             ious = np.array([Yolov8OnnxDetectionModel.iou(box, valid_boxes[0]) for box in valid_boxes])
    #             valid_indices = ious < 0.7
    #             valid_boxes = valid_boxes[valid_indices]
    #             # valid_masks = valid_masks[valid_indices]
    #             valid_masks = [mask for i, mask in enumerate(valid_masks) if valid_indices[i]]

    #         # 处理 result_masks，将其调整到 input_size 大小并放置到正确位置
    #         resized_result_masks = []
    #         for box, mask in zip(result_boxes, result_masks):
    #             x1, y1, x2, y2, class_id, prob = box
    #             x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

    #             # 创建一个全零的 input_size 大小的画布
    #             canvas = np.zeros((input_size,input_size), dtype=np.uint8)

    #             # 将 mask 调整到检测框的大小
    #             img_mask = Image.fromarray(mask, "L")
    #             img_mask = img_mask.resize((x2 - x1, y2 - y1))
    #             resized_mask = np.array(img_mask)

    #             # 将调整后的 mask 放置到画布上正确的位置
    #             canvas[y1:y2, x1:x2] = resized_mask
    #             resized_result_masks.append(canvas)
    #         result_boxes = np.array(result_boxes)
    #         result_masks = np.array(resized_result_masks)
    #         result_boxes=torch.from_numpy(result_boxes)
    #         result_masks=torch.from_numpy(result_masks)
    #         prediction_results.append([result_boxes, result_masks])

    #     return prediction_results
    @staticmethod
    def crop_mask(masks, boxes):
        """
        Takes a mask and a bounding box, and returns a mask that is cropped to the bounding box, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (Numpy.ndarray): [n, h, w] tensor of masks.
            boxes (Numpy.ndarray): [n, 4] tensor of bbox coordinates in relative point form.

        Returns:
            (Numpy.ndarray): The masks are being cropped to the bounding box.
        """
        n, h, w = masks.shape
        x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, 1)
        r = np.arange(w, dtype=x1.dtype)[None, None, :]
        c = np.arange(h, dtype=x1.dtype)[None, :, None]
        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))
    @staticmethod
    def scale_mask(masks, im0_shape, ratio_pad=None):
        """
        Takes a mask, and resizes it to the original image size, from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (np.ndarray): resized and padded masks/images, [h, w, num]/[h, w, 3].
            im0_shape (tuple): the original image shape.
            ratio_pad (tuple): the ratio of the padding to the original image.

        Returns:
            masks (np.ndarray): The masks that are being returned.
        """
        im1_shape = masks.shape[:2]
        if ratio_pad is None:  # calculate from im0_shape
            gain = min(im1_shape[0] / im0_shape[0], im1_shape[1] / im0_shape[1])  # gain  = old / new
            pad = (im1_shape[1] - im0_shape[1] * gain) / 2, (im1_shape[0] - im0_shape[0] * gain) / 2  # wh padding
        else:
            pad = ratio_pad[1]

        # Calculate tlbr of mask
        top, left = int(round(pad[1] - 0.1)), int(round(pad[0] - 0.1))  # y, x
        bottom, right = int(round(im1_shape[0] - pad[1] + 0.1)), int(round(im1_shape[1] - pad[0] + 0.1))
        if len(masks.shape) < 2:
            raise ValueError(f'"len of masks shape" should be 2 or 3, but got {len(masks.shape)}')
        masks = masks[top:bottom, left:right]
        masks = cv2.resize(
            masks, (im0_shape[1], im0_shape[0]), interpolation=cv2.INTER_LINEAR
        )  # INTER_CUBIC would be better
        if len(masks.shape) == 2:
            masks = masks[:, :, None]
        return masks
    def process_mask(self, protos, masks_in, bboxes, im0_shape):
        """
        Takes the output of the mask head, and applies the mask to the bounding boxes. This produces masks of higher
        quality but is slower, from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            protos (numpy.ndarray): [mask_dim, mask_h, mask_w].
            masks_in (numpy.ndarray): [n, mask_dim], n is number of masks after nms.
            bboxes (numpy.ndarray): bboxes re-scaled to original image shape.
            im0_shape (tuple): the size of the input image (h,w,c).

        Returns:
            (numpy.ndarray): The upsampled masks.
        """
        c, mh, mw = protos.shape
        masks = np.matmul(masks_in, protos.reshape((c, -1))).reshape((-1, mh, mw)).transpose(1, 2, 0)  # HWN
        masks = np.ascontiguousarray(masks)
        masks = self.scale_mask(masks, im0_shape)  # re-scale mask from P3 shape to original input image shape
        masks = np.einsum("HWN -> NHW", masks)  # HWN -> NHW
        masks = self.crop_mask(masks, bboxes)
        return np.greater(masks, 0.5)
    def postprocess(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold, nm=32):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.
            nm (int): the number of masks.

        Returns:
            boxes (List): list of bounding boxes.
            segments (List): list of segments.
            masks (np.ndarray): [N, H, W], output masks.
        """
        x, protos = preds[0], preds[1]  # Two outputs: predictions and protos

        # Transpose dim 1: (Batch_size, xywh_conf_cls_nm, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls_nm)
        x = np.einsum("bcn->bnc", x)

        batch_size = x.shape[0]
        # batch_boxes = []
        # batch_segments = []
        # batch_masks = []
        prediction_results=[]
        # 按批次处理
        for batch_idx in range(batch_size):
            batch_x = x[batch_idx]

            # Predictions filtering by conf-threshold
            batch_x = batch_x[np.amax(batch_x[..., 4:-nm], axis=-1) > conf_threshold]

            # Create a new matrix which merge these(box, score, cls, nm) into one
            batch_x = np.c_[batch_x[..., :4], np.amax(batch_x[..., 4:-nm], axis=-1), np.argmax(batch_x[..., 4:-nm], axis=-1), batch_x[..., -nm:]]

            # NMS filtering
            if len(batch_x) > 0:
                batch_x = batch_x[cv2.dnn.NMSBoxes(batch_x[:, :4], batch_x[:, 4], conf_threshold, iou_threshold)]

                # Bounding boxes format change: cxcywh -> xyxy
                batch_x[..., [0, 1]] -= batch_x[..., [2, 3]] / 2
                batch_x[..., [2, 3]] += batch_x[..., [0, 1]]

                # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
                batch_x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
                batch_x[..., :4] /= min(ratio)

                # Bounding boxes boundary clamp
                batch_x[..., [0, 2]] = batch_x[:, [0, 2]].clip(0, im0.shape[1])
                batch_x[..., [1, 3]] = batch_x[:, [1, 3]].clip(0, im0.shape[0])

                # Process masks
                batch_masks_ = self.process_mask(protos[batch_idx], batch_x[:, 6:], batch_x[:, :4], im0.shape)

                # Masks -> Segments(contours)
                # batch_segments_ = self.masks2segments(batch_masks_)
                prediction_results.append([
                    batch_x[..., :6],
                    batch_masks_
                ])
            #     batch_boxes.append(batch_x[..., :6])
            #     # batch_segments.append(batch_segments_)
            #     batch_masks.append(batch_masks_)
            else:
                # batch_boxes.append(np.empty((0, 6)))
                # batch_segments.append([])
                # batch_masks.append(np.empty((0, *im0[batch_idx].shape[:2])))
                prediction_results.append([
                    np.empty((0, 6)),
                    np.empty((0, *im0.shape[:2]))
                ])

        # return batch_boxes, batch_segments, batch_masks
        
        return prediction_results
    @staticmethod
    def masks2segments(masks):
        """
        Takes a list of masks(n,h,w) and returns a list of segments(n,xy), from
        https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/ops.py.

        Args:
            masks (numpy.ndarray): the output of the model, which is a tensor of shape (batch_size, 160, 160).

        Returns:
            segments (List): list of segment masks.
        """
        segments = []
        for x in masks.astype("uint8"):
            c = cv2.findContours(x, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]  # CHAIN_APPROX_SIMPLE
            if c:
                c = np.array(c[np.array([len(x) for x in c]).argmax()]).reshape(-1, 2)
            else:
                c = np.zeros((0, 2))  # no segments found
            segments.append(c.astype("float32"))
        return segments


    def perform_inference(self, image:Optional[List[np.ndarray]] ):
        """
        Prediction is performed using self.model and the prediction result is set to self._original_predictions.
        Args:
            image: np.ndarray
                A numpy array that contains the image to be predicted. 3 channel image should be in RGB order.
        """

        # Confirm model is loaded
        if self.model is None:
            raise ValueError("Model is not loaded, load it by calling .load_model()")

        # Get input/output names shapes
        model_inputs = self.model.get_inputs()
        model_output = self.model.get_outputs()

        input_names = [model_inputs[i].name for i in range(len(model_inputs))]
        output_names = [model_output[i].name for i in range(len(model_output))]

        # input_shape = model_inputs[0].shape[2:]  # w, h
        image_shape = image[0].shape[:2]  # h, w

        input_shape=image[0].shape[:2]
        # input_shape=[image.shape[0],image.shape[1]]
        ratio=[input_shape[0]/image[0].shape[0],input_shape[1]/image[0].shape[1]]
        # Prepare image
        st=time.time()
        images_tensor = self._preprocess_image(image, input_shape)
        print("preprocess time", time.time()-st)
        # image_tensor = np.vstack([image_tensor,image_tensor]).squeeze()
        # Inference
        st=time.time()
        former_peak_mem=torch.cuda.max_memory_allocated()
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info_before = pynvml.nvmlDeviceGetMemoryInfo(handle)
        former_usage = info_before.used 
        # images_tensor = np.vstack([images_tensor,images_tensor,images_tensor]).squeeze()
        # for i in range(10):
        outputs = self.model.run(output_names, {input_names[0]: images_tensor})
        torch.cuda.synchronize()
        later_peak_mem=torch.cuda.max_memory_allocated() 
        info_after = pynvml.nvmlDeviceGetMemoryInfo(handle)
        later_usage = info_after.used
        print("gpu mem use and iamge_tensor.shape",later_peak_mem-former_peak_mem,images_tensor.shape)
        print("onnx time", time.time()-st)
        print(f"显存变化: {(later_usage - former_usage)/1024/1024}")
        #循环10次 1,3,1024,1024 532; 2 1558 3 2582

        pynvml.nvmlShutdown()
        nm=outputs[1].shape[1]
        # 输入的outputs情况:,outputs[0]为bbox,shape为(1, 116, 8400), outputs[1]为mask(1, 32, 160, 160)
        # Post-process
        # prediction_results = self._post_process_ori(outputs, input_shape, image_shape)
        # prediction_results = self._post_process(outputs, img_widths=input_shape[1], img_heights=input_shape[0], input_size=640)
        st=time.time()
        prediction_results=self.postprocess(outputs,image[0],ratio=ratio,pad_w=0,pad_h=0,conf_threshold=self.confidence_threshold,iou_threshold=self.iou_threshold,nm=nm)
        print("posr time", time.time()-st)
        self._original_predictions = prediction_results
        self._original_shape = image[0].shape
        # 输出的prediction_results要求是一个list shape(n,2), n是照片数量,  prediction_results[i]也是list,其中:prediction_results[i][0]是box,shape为(m,6), m为obj数量;prediction_results[i][1]是mask,shape(m,h,w)

    @property
    def category_names(self):
        return list(self.category_mapping.values())

    @property
    def num_categories(self):
        """
        Returns number of categories
        """
        return len(self.category_mapping)

    @property
    def has_mask(self):
        """
        Returns if model output contains segmentation mask

        Not yet supported
        """
        return True
    def _create_object_prediction_list_from_original_predictions(
        self,
        shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
        full_shape_list: Optional[List[List[int]]] = None,
    ):
        """
        self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
        self._object_prediction_list_per_image.
        Args:
            shift_amount_list: list of list
                To shift the box and mask predictions from sliced image to full sized image, should
                be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
            full_shape_list: list of list
                Size of the full image after shifting, should be in the form of
                List[[height, width],[height, width],...]
        """
        original_predictions = self._original_predictions

        # compatibility for sahi v0.8.15
        shift_amount_list = fix_shift_amount_list(shift_amount_list)
        full_shape_list = fix_full_shape_list(full_shape_list)

        # handle all predictions
        object_prediction_list_per_image = []

        for image_ind, image_predictions in enumerate(original_predictions):
            shift_amount = shift_amount_list[image_ind]
            full_shape = None if full_shape_list is None else full_shape_list[0]
            object_prediction_list = []

            # Extract boxes and optional masks/obb
            if self.has_mask or self.is_obb:
                if isinstance(image_predictions[0],torch.Tensor):
                    boxes = image_predictions[0].cpu().detach().numpy()
                    masks_or_points = image_predictions[1].cpu().detach().numpy()
                else:
                    boxes = image_predictions[0]
                    masks_or_points = image_predictions[1]

            else:
                boxes = image_predictions.data.cpu().detach().numpy()
                masks_or_points = None

            # Process each prediction
            for pred_ind, prediction in enumerate(boxes):
                # Get bbox coordinates
                bbox = prediction[:4].tolist()
                score = prediction[4]
                category_id = int(prediction[5])
                category_name = self.category_mapping[str(category_id)]

                # Fix box coordinates
                bbox = [max(0, coord) for coord in bbox]
                if full_shape is not None:
                    bbox[0] = min(full_shape[1], bbox[0])
                    bbox[1] = min(full_shape[0], bbox[1])
                    bbox[2] = min(full_shape[1], bbox[2])
                    bbox[3] = min(full_shape[0], bbox[3])

                # Ignore invalid predictions
                if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
                    logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
                    continue

                # Get segmentation or OBB points
                segmentation = None
                if masks_or_points is not None:
                    if self.has_mask:
                        bool_mask = masks_or_points[pred_ind]
                        # Resize mask to original image size
                        bool_mask = cv2.resize(
                            bool_mask.astype(np.uint8), (self._original_shape[1], self._original_shape[0])
                        )
                        segmentation = get_coco_segmentation_from_bool_mask(bool_mask)
                    else:  # is_obb
                        obb_points = masks_or_points[pred_ind]  # Get OBB points for this prediction
                        segmentation = get_coco_segmentation_from_obb_points(obb_points)

                    if len(segmentation) == 0:
                        continue

                # Create and append object prediction
                object_prediction = ObjectPrediction(
                    bbox=bbox,
                    category_id=category_id,
                    score=score,
                    segmentation=segmentation,
                    category_name=category_name,
                    shift_amount=shift_amount,
                    full_shape=self._original_shape[:2] if full_shape is None else full_shape,  # (height, width)
                )
                object_prediction_list.append(object_prediction)

            object_prediction_list_per_image.append(object_prediction_list)

        self._object_prediction_list_per_image = object_prediction_list_per_image

    # def _create_object_prediction_list_from_original_predictions(
    #     self,
    #     shift_amount_list: Optional[List[List[int]]] = [[0, 0]],
    #     full_shape_list: Optional[List[List[int]]] = None,
    # ):
    #     """
    #     self._original_predictions is converted to a list of prediction.ObjectPrediction and set to
    #     self._object_prediction_list_per_image.
    #     Args:
    #         shift_amount_list: list of list
    #             To shift the box and mask predictions from sliced image to full sized image, should
    #             be in the form of List[[shift_x, shift_y],[shift_x, shift_y],...]
    #         full_shape_list: list of list
    #             Size of the full image after shifting, should be in the form of
    #             List[[height, width],[height, width],...]
    #     """
    #     original_predictions = self._original_predictions

    #     # compatilibty for sahi v0.8.15
    #     shift_amount_list = fix_shift_amount_list(shift_amount_list)
    #     full_shape_list = fix_full_shape_list(full_shape_list)

    #     # handle all predictions
    #     object_prediction_list_per_image = []
    #     for image_ind, image_predictions_in_xyxy_format in enumerate(original_predictions):
    #         shift_amount = shift_amount_list[image_ind]
    #         full_shape = None if full_shape_list is None else full_shape_list[image_ind]
    #         object_prediction_list = []

    #         # process predictions
    #         for prediction in image_predictions_in_xyxy_format.cpu().detach().numpy():
    #             x1 = prediction[0]
    #             y1 = prediction[1]
    #             x2 = prediction[2]
    #             y2 = prediction[3]
    #             bbox = [x1, y1, x2, y2]
    #             score = prediction[4]
    #             category_id = int(prediction[5])
    #             category_name = self.category_mapping[str(category_id)]

    #             # fix negative box coords
    #             bbox[0] = max(0, bbox[0])
    #             bbox[1] = max(0, bbox[1])
    #             bbox[2] = max(0, bbox[2])
    #             bbox[3] = max(0, bbox[3])

    #             # fix out of image box coords
    #             if full_shape is not None:
    #                 bbox[0] = min(full_shape[1], bbox[0])
    #                 bbox[1] = min(full_shape[0], bbox[1])
    #                 bbox[2] = min(full_shape[1], bbox[2])
    #                 bbox[3] = min(full_shape[0], bbox[3])

    #             # ignore invalid predictions
    #             if not (bbox[0] < bbox[2]) or not (bbox[1] < bbox[3]):
    #                 logger.warning(f"ignoring invalid prediction with bbox: {bbox}")
    #                 continue

    #             object_prediction = ObjectPrediction(
    #                 bbox=bbox,
    #                 category_id=category_id,
    #                 score=score,
    #                 segmentation=None,
    #                 category_name=category_name,
    #                 shift_amount=shift_amount,
    #                 full_shape=full_shape,
    #             )
    #             object_prediction_list.append(object_prediction)
    #         object_prediction_list_per_image.append(object_prediction_list)

    #     self._object_prediction_list_per_image = object_prediction_list_per_image

# output0 = outputs[0]
# output1 = outputs[1]
# output0 = output0[0].transpose()
# output1 = output1[0]
# boxes = output0[:,0:84]
# masks = output0[:,84:]
# output1 = output1.reshape(32,160*160)
# masks = masks @ output1
# boxes = np.hstack([boxes,masks])
# for row in boxes:
#     xc,yc,w,h = row[:4]
#     x1 = (xc-w/2)/640*img_width
#     y1 = (yc-h/2)/640*img_height
#     x2 = (xc+w/2)/640*img_width
#     y2 = (yc+h/2)/640*img_height
#     prob = row[4:84].max()
#     if prob < 0.5:
#         continue
#     class_id = row[4:84].argmax()
#     label = yolo_classes[class_id]