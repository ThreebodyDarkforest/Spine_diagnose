import os, copy
import glob
import SimpleITK as sitk
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import cv2
from matplotlib.patches import Rectangle

class_names = ['L1', 'L2', 'L3' ,'L4', 'L5', 'T12-L1', 'L1-L2', 'L2-L3', 'L3-L4', 'L4-L5', 'L5-S1']

num_class = {
    "L1" : 0,
    "L2" : 1,
    "L3" : 2,
    "L4" : 3, 
    "L5" : 4,
    "T12-L1" : 5,
    "L1-L2" : 6,
    "L2-L3" : 7,
    "L3-L4" : 8,
    "L4-L5" : 9,
    "L5-S1" : 10,
    "T11-T12": 5,
    "Anchor": 11,
    "All": 12,
}

disease_class = {
    "v1": 0,
    "v2": 1,
    "v3": 2,
    "v4": 3,
    "v5": 4,
}

init_param = {
    'vertebra': ([82, 64], 0.015),
    'disc': ([64, 32], 0.01),
    'stop': 0.014,
}
scale_size = (648, 648)

def dicom_metainfo(dicm_path, list_tag):
    '''
    获取dicom的元数据信息
    dicm_path: dicom文件地址
    list_tag: 标记名称列表,比如['0008|0018',]
    '''
    reader = sitk.ImageFileReader()
    reader.LoadPrivateTagsOn()
    reader.SetFileName(dicm_path)
    reader.ReadImageInformation()
    return [reader.GetMetaData(t) for t in list_tag]

def dicom2array(dcm_path):
    '''
    读取dicom文件并把其转化为灰度图(np.array)
    https://simpleitk.readthedocs.io/en/master/link_DicomConvert_docs.html
    dcm_path: dicom文件
    '''
    image_file_reader = sitk.ImageFileReader()
    image_file_reader.SetImageIO('GDCMImageIO')
    image_file_reader.SetFileName(dcm_path)
    image_file_reader.ReadImageInformation()
    image = image_file_reader.Execute()
    if image.GetNumberOfComponentsPerPixel() == 1:
        image = sitk.RescaleIntensity(image, 0, 255)
        if image_file_reader.GetMetaData('0028|0004').strip() == 'MONOCHROME1':
            image = sitk.InvertIntensity(image, maximum=255)
        image = sitk.Cast(image, sitk.sitkUInt8)
    img_x = sitk.GetArrayFromImage(image)[0]
    return img_x

def regress(img, coord, dtype):
    height, width = img.shape
    img = np.where(cv2.resize(img, scale_size) > 0, 1, 0)
    box_wh, thresh = copy.deepcopy(init_param[dtype])
    stop_thresh = init_param['stop']
    while True:
        box_wh[0] -= 1 / scale_size[0] * width
        box_wh[1] -= 1 / scale_size[1] * height
        if(box_wh[0] * box_wh[1] <= scale_size[0] * scale_size[1] * stop_thresh): break
        if(np.sum(img[int(coord[0]-box_wh[0]/2):int(coord[0]+box_wh[0]/2), \
                      int(coord[1]-box_wh[1]/2):int(coord[1]-box_wh[1]/2)]) \
                      <= np.sum(img) * thresh): break
    return (box_wh[0] / scale_size[0] * width, box_wh[1] / scale_size[1] * height)

def preprocess(img):
    # 对图像进行锐化和gamma矫正和直方图均衡化
    gray_img = img.copy()
    blurred_img = cv2.GaussianBlur(gray_img, (5, 5), 0)
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened_img = cv2.filter2D(blurred_img, -1, kernel)
    gamma = 1.5
    gamma_img = sharpened_img ** gamma
    gamma_img = cv2.normalize(sharpened_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    equalized_img = cv2.equalizeHist(gamma_img)
    # 将图像二值化
    ret, binary_img = cv2.threshold(equalized_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 对图像进行闭运算
    kernel = np.ones((1, 1),np.uint8)
    closing_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel, iterations=3)
    return closing_img

def postprocess(img, coord: List[int], box_size: List[float]):
    return img[int(coord[1] - box_size[1] / 2) - 2 : int(coord[1] + box_size[1] / 2) + 2, \
               int(coord[0] - box_size[0] / 2) - 2 : int(coord[0] + box_size[0] / 2) + 2]

def get_disease_type(tag):
    disease = tag['vertebra'] if 'vertebra' in \
              tag.keys() else all_gts[i]['tag']['disc']
    disease = disease if disease != '' else all_gts[i]['tag']['disc']
    disease = [x for x in disease.split(',') if x != '']
    return disease

def get_rgb_image(path):
    filename = os.path.splitext(os.path.basename(path))[0]
    filedir = os.path.dirname(path)
    number = int(filename[5:])
    img = np.array([dicom2array(os.path.join(filedir, f'image{number + _}.dcm')) for _ in range(-1, 2)])
    return img.transpose((1, 2, 0))

if __name__ == '__main__':
    # 先处理标签和图像的对应关系 studyUid,seriesUid,instanceUid,annotation
    annotation_info = pd.DataFrame(columns=('studyUid', 'seriesUid', 'instanceUid', 'annotation'))
    json_df = pd.read_json('./lumbar_train51_annotation.json')
    for idx in json_df.index:
        studyUid = json_df.loc[idx, "studyUid"]
        seriesUid = json_df.loc[idx, "data"][0]['seriesUid']
        instanceUid = json_df.loc[idx, "data"][0]['instanceUid']
        annotation = json_df.loc[idx, "data"][0]['annotation']
        row = pd.Series(
            {'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid, 'annotation': annotation})
        annotation_info = annotation_info._append(row, ignore_index=True)
    dcm_paths = glob.glob(os.path.join('./train', "**", "**.dcm"))
    # 'studyUid','seriesUid','instanceUid'
    tag_list = ['0020|000d', '0020|000e', '0008|0018']  # 好像所有的图片都是这个
    dcm_info = pd.DataFrame(columns=('dcmPath', 'studyUid', 'seriesUid', 'instanceUid'))
    for dcm_path in dcm_paths:
        try:
            studyUid, seriesUid, instanceUid = dicom_metainfo(dcm_path, tag_list)
            row = pd.Series(
                {'dcmPath': dcm_path, 'studyUid': studyUid, 'seriesUid': seriesUid, 'instanceUid': instanceUid})
            dcm_info = dcm_info._append(row, ignore_index=True)
        except:
            continue
    result = pd.merge(annotation_info, dcm_info, on=['studyUid', 'seriesUid', 'instanceUid'])
    result = result.set_index('dcmPath')['annotation']
    #print(result)
    # results里的每一行，行索引即为该病人（如study41）中、被标注了的那张图像的名称（如image17.dcm）。
    # 每一行中的值，代表椎骨、椎间盘等金标准标签。具体见“标签数据结构示意图.jpg”。
    path = './images/train'
    label_path = './labels'
    view_path = './views'
    precise_label_path = './precise_labels'

    train_label_path = os.path.join(label_path, os.path.basename(path))
    precise_path = os.path.join(precise_label_path, os.path.basename(path))
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(train_label_path):
        os.makedirs(train_label_path)
    if not os.path.exists(view_path):
        os.makedirs(view_path)
    if not os.path.exists(precise_path):
        os.makedirs(precise_path)

    # 为yolov6生成标注数据

    for ind, val in zip(result.index, result.values):
        #print(ind, val)
        dcm_path = ind
        img_x = dicom2array(dcm_path)
        height, width = img_x.shape
        name = os.path.splitext(os.path.basename(dcm_path))[0]
        name = os.path.basename(os.path.dirname(dcm_path)) + '_' + name
        all_gts = val[0]['data']['point']
        plt.figure(1)
        fig, ax = plt.subplots()
        plt.imshow(img_x, cmap='gray')
        save_path = os.path.join(path, name + '.jpg')
        label_path = os.path.join(train_label_path, name + '.txt')
        label_txt = ''
        preview_path = os.path.join(view_path, name + '.jpg')
        plt.imsave(save_path, img_x, cmap='gray')
        all_gts = sorted(all_gts, key=lambda x : x['coord'][1], reverse=True)
        
        # 生成全局定位锚
        min_x, max_x = min([x['coord'][0] for x in all_gts]), max([x['coord'][0] for x in all_gts])
        min_y, max_y = all_gts[-1]['coord'][1], all_gts[0]['coord'][1]
        length = np.mean([all_gts[i]['coord'][1] - all_gts[i-1]['coord'][1] for i in range(1, len(all_gts))])
        anchor_x, anchor_y = all_gts[0]['coord'][0] - 0.7 * length, all_gts[0]['coord'][1] - 0.7 * length
        anchor_w, anchor_h = regress(preprocess(img_x), coords, 'vertebra')
        anchor_rect = Rectangle((int(anchor_x - anchor_w / 2), int(anchor_y - anchor_h / 2)), anchor_w, anchor_h, linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(anchor_rect)
        class_id, center_x, center_y, box_w, box_h = num_class['Anchor'], anchor_x / width, anchor_y / height, anchor_w / width, anchor_h / height
        label_txt += f'{class_id} {center_x} {center_y} {box_w} {box_h}\n'

        box_axis = []
        # 生成其他标注框
        for i in range(len(all_gts)):
            coords = all_gts[i]['coord']
            vertebrae_or_disc_name = all_gts[i]['tag']['identification']  # 哪块椎骨或者椎间盘？
            vertebrae_disease = all_gts[i]['tag']['vertebra'] if 'vertebra' in all_gts[i]['tag'].keys() else None  # 椎骨是否有疾病？
            disc_disease = all_gts[i]['tag']['disc'] if 'disc' in all_gts[i]['tag'].keys() else None  # 椎间盘是否有疾病？
            assert (vertebrae_disease is not None or disc_disease is not None)  # 至少有一个不是None
            disease = vertebrae_disease if vertebrae_disease is not None and len(vertebrae_disease) > 0 else disc_disease
            if disease == '': disease = disc_disease
            color = 'r' if disease == 'v1' else 'b'  # v1是没有疾病，用蓝色表示，否则用红色。
            # 但是颜色区分有可能无效，因为plt.show()似乎是读取完了所有标记，然后用最后的颜色去画的。不过，重要的是名称和疾病的标签。
            plt.plot([coords[0]], [coords[1]], marker='o', color=color, markersize=2)
            plt.text(coords[0] + 4, coords[1] + 4, vertebrae_or_disc_name + ', ' + disease, color=color, size=10)

            disease_type = [disease_class[x] for x in get_disease_type(all_gts[i]['tag'])]

            x, y = regress(preprocess(img_x), coords, 'vertebra' if vertebrae_disease is not None else 'disc')
            rect = Rectangle((int(coords[0] - x / 2), int(coords[1] - y / 2)), x, y, linewidth=1, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            # 记录所有矩形边界点坐标
            box_axis.append([coords[0] - x / 2, coords[1] - y / 2])
            box_axis.append([coords[0] + x / 2, coords[1] + y / 2])
            # 生成 yolo 数据
            class_id, center_x, center_y, box_w, box_h = num_class[vertebrae_or_disc_name], coords[0] / width, coords[1] / height, x / width, y / height
            label_txt += f'{class_id} {center_x} {center_y} {box_w} {box_h}\n'

            # 生成用于分类的图像和数据
            pos_type = 0 if vertebrae_disease is not None else 1
            precise_txt = f'{pos_type} {disease_type}'
            precise_txt_path = os.path.join(precise_path, name + f'_{i}' + '.txt')
            precise_file_path = os.path.join(precise_path, name + f'_{i}' + '.jpg')
            with open(precise_txt_path, 'w', encoding='utf-8') as f:
                f.write(precise_txt)
            plt.imsave(precise_file_path, postprocess(img_x, coords, [x, y]), cmap='gray')
            
        # 生成整体标注框
        min_x, min_y = np.min(box_axis, axis=0)
        max_x, max_y = np.max(box_axis, axis=0)
        all_w, all_h = max_x - min_x, max_y - min_y
        all_rect = Rectangle((int(min_x), int(min_y)), int(all_w), int(all_h), linewidth=1, edgecolor='g', facecolor='none')
        ax.add_patch(all_rect)
        class_id, center_x, center_y, box_w, box_h = num_class['All'], (min_x + all_w / 2) / width, (min_y + all_h / 2) / height, (max_x - min_x) / width, (max_y - min_y) / height
        label_txt += f'{class_id} {center_x} {center_y} {box_w} {box_h}\n'

        with open(label_path, 'w', encoding='utf-8') as f:
            f.write(label_txt)
        plt.savefig(preview_path)
        plt.cla()