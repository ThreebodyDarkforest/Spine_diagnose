import os, requests, torch, math, cv2
import numpy as np
import PIL
import torch.nn as nn

from components.logger import LOGGER
from yolov6.utils.events import load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import fix_non_max_suppression
from yolov6.core.inferer import Inferer

from typing import List, Optional, Union

def check_img_size(img_size, s=32, floor=0):
  def make_divisible( x, divisor):
    # Upward revision the value x to make it evenly divisible by the divisor.
    return math.ceil(x / divisor) * divisor
  """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
  if isinstance(img_size, int):  # integer i.e. img_size=640
      new_size = max(make_divisible(img_size, int(s)), floor)
  elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
      new_size = [max(make_divisible(x, int(s)), floor) for x in img_size]
  else:
      raise Exception(f"Unsupported type of img_size: {type(img_size)}")

  if new_size != img_size:
      print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
  return new_size if isinstance(img_size,list) else [new_size]*2

def process_image(path, img_size, stride, half):
  '''Process image before image inference.'''
  try:
    from PIL import Image
    img = requests.get(path, stream=True).raw if 'http' in path else path
    img_src = np.asarray(Image.open(img))
    assert img_src is not None, f'Invalid image: {path}'
  except Exception as e:
    LOGGER.warn(e)
  if isinstance(path, np.ndarray):
     img_src = path.copy()
  image = letterbox(img_src, img_size, stride=stride)[0]

  # Convert
  image = image.transpose((2, 0, 1))  # HWC to CHW
  image = torch.from_numpy(np.ascontiguousarray(image))
  image = image.half() if half else image.float()  # uint8 to fp16/32
  image /= 255  # 0 - 255 to 0.0 - 1.0

  return image, img_src

def get_yolo_model(weights_path, config_path, half: bool = False, device = 'cpu'):
    assert '.pt' in weights_path, 'Invalid model path.'
    model = DetectBackend(weights_path, device=device)
    stride = model.stride
    class_names = load_yaml(config_path)['names']

    if half & (device.type != 'cpu'):
        model.model.half()
    else:
        model.model.float()
        half = False

    #if device.type != 'cpu':
    #    model(torch.zeros(1, 3, *img_size).to(device).type_as(next(model.model.parameters())))  # warmup
    return model, stride, class_names
    
def get_rcnn_model():
   pass

def detect(model: nn.Module, img_path: Union[str, np.ndarray], class_names: List[str], model_type: str = 'yolo', hide_labels: bool = False, \
           stride = None, hide_conf: bool = False, img_size:int = 640, conf_thres: float =.25, device='cpu', \
           iou_thres: float =.45, max_det:int =  1000, agnostic_nms: bool = False, half:bool = False, plot = False):
    '''Detect the vertebrae and intervertebral discs present in the image.

    Please make sure that the image you input is a mid-sagittal 
    T2-weighted CT slice image to avoid issues with detecting the target.

    Args:
        model: (nn.Module), the model to use, note that only supports yolo/rcnn type now.
        img_path: (str), The path of the image to detect.
        class_names: (List[str]), a list of class names for model detection.
        model_type: (str), the type of detection model, support 'yolo' and 'rcnn'. Default is 'yolo'.
        hide_labels: (bool), whether to hide the label of the detected object. Default is False.
        stride: (int), the stride of the model. Default is None.
        hide_conf: (bool), whether to hide the confidence score of the detected object. Default is False.
        img_size: (int), the size of the input image. Default is 640.
        conf_thres: (float), confidence threshold for object detection. Default is 0.25.
        device: (str), the device to run the model on. Default is 'cpu'.
        iou_thres: (float), IoU threshold for non-maximum suppression. Default is 0.45.
        max_det: (int), maximum number of detections to keep. Default is 1000.
        agnostic_nms: (bool), whether to use agnostic NMS. Default is False.
        half: (bool), whether to use half precision. Default is False.
        plot: (bool), whether to plot the detected objects. Default is False.
        
    Returns:
        img_ori: (numpy.ndarray), the original image in numpy array format.
        results: (List[Dict]), a list of dictionaries containing the detected objects' bounding boxes, labels, and class numbers. 
    '''

    model_type = model_type.lower()
    assert model_type in ['yolo', 'rcnn'], f'Invalid model type {model_type}.'

    if model_type == 'yolo':
        img_size = check_img_size(img_size, s=model.stride if stride is None else stride)
        img, img_src = process_image(img_path, img_size, stride, half)
        img = img.to(device)
        if len(img.shape) == 3:
            img = img[None]
            # expand for batch dim

        pred_results = model(img)
        classes:Optional[List[int]] = None # the classes to keep
        det, det_logits = fix_non_max_suppression(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        det, det_logits = det[0], det_logits[0]

        gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        img_ori = img_src.copy()
        results = []

        if len(det):
            det[:, :4] = Inferer.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
            for logits, (*xyxy, conf, cls) in zip(reversed(det_logits), reversed(det)):
                class_num = int(cls)
                label = None if hide_labels else (class_names[class_num] if hide_conf else f'{class_names[class_num]} {conf:.2f}')
                results.append({
                    'xyxy': ((int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))), 
                    'label': class_names[class_num],
                    'class_num': class_num, 
                    'confidence': conf.item(), 
                    'logits': logits.tolist(),
                })
                if plot:
                    Inferer.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=Inferer.generate_colors(class_num, True))
        
        return img_src, results
    else:
       pass

def detect_batch():
   pass

if __name__ == '__main__':
   device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
   model, stride, class_names = get_yolo_model('../../weights/best_ckpt.pt', '../../data/spine.yaml', device=device)
   res = detect(model, '../../data/images/study1_image35.jpg', class_names, stride=stride, device=device)
   print(res)