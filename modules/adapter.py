import torch
from detector import detect, get_yolo_model
from util import filter_box, plot_boxes, crop_img, get_center
from classifier import get_resnet_model, classify
import cv2, os
from typing import Union, List
from yolov6.utils.events import LOGGER

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_disease_str(disease: Union[List, tuple]):
    disease_str = ''
    if disease[0] != 'None': disease_str += disease[0]
    if disease[1] != 'None': disease_str += f', {disease[1]}'
    return disease_str

if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model, stride, class_names = get_yolo_model(PATH + '/weights/detect.pt', PATH + '/data/spine.yaml', device=device)
    img_src, boxes = detect(model, PATH + '/data/images/study3_image7.jpg', class_names, stride=stride, device=device)
    boxes = filter_box(img_src, boxes)
    imgs = crop_img(img_src, boxes)
    classify_model, class_names_ = get_resnet_model(PATH + '/weights/classify.pt', PATH + '/data/spine.yaml', device=device)
    LOGGER.info('Detection done. start classifying now...')
    ret = [classify(classify_model, img, class_names_, device=device) for img in imgs]
    LOGGER.info('Classification done.')
    result = []
    for data, box in zip(ret, boxes):
        disease_type, conf = data[0], data[1]
        label = box['label']
        result.append((f'{label} {get_disease_str(disease_type)}', get_center(box)))
        box['label'] += ' ' + get_disease_str(disease_type)
        box['confidence'] *= conf[0] * conf[1]
        
    LOGGER.info(boxes)
    plot_boxes(img_src, 1, boxes)
    cv2.imwrite('test.jpg', img_src)
    LOGGER.info('done.')