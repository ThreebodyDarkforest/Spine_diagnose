import torch
from detector import detect, get_yolo_model
from util import filter_box, filter_box_deperacated, plot_boxes, crop_img, get_center, IMG_EXT
from classifier import get_resnet_model, classify, get_vit_model, get_swin_model, get_resnest_model
from evaler import evalutate
import cv2, os
from typing import Union, List
from components.logger import LOGGER
import argparse
import torch.distributed as dist
import yaml
import os.path as osp
from pathlib import Path
import datetime
import sys
from util import Config
import torch.nn as nn
import glob
from tqdm import tqdm
import json
import numpy as np

from yolov6.core.engine import Trainer
from yolov6.utils.events import save_yaml, load_yaml
from yolov6.utils.envs import get_envs, select_device, set_random_seed
from yolov6.utils.general import increment_name, find_latest_checkpoint, check_img_size

from resnet.core.trainer import Trainer as resn_Trainer
from vit.core.trainer import Trainer as vit_Trainer
from swin_trans.core.trainer import Trainer as swin_Trainer
from resnest.core.train import Trainer as resnest_Trainer

PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_date_str():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M')

def get_disease_str(disease: Union[List, tuple]):
    disease_str = ''
    if disease[0] != 'None': disease_str += disease[0]
    if disease[1] != 'None': disease_str += f', {disease[1]}'
    return disease_str

def predict(args):
    try:
        device = torch.device(args.device)
    except:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    detect_model, classify_model = None, None
    cfgs = load_yaml(args.yaml)
    if 'yolov6' in cfgs['detect']:
        detect_model, stride, dclass_names = get_yolo_model(args.detect_model, args.yaml, device=device)
    if 'resnet' in cfgs['classify1']:
        classify_modelA, cclass_namesA = get_resnet_model(args.vert_model, args.yaml, device=device)
        classify_modelB, cclass_namesB = get_resnest_model(args.disc_model, args.yaml, device=device)
    #if 'ViT' in cfgs['classify']:
    #    classify_model, cclass_names = get_vit_model(args.classify_model, args.yaml, device=device)
    #if 'swin' in cfgs['classify']:
    #    classify_model, cclass_names = get_swin_model(args.classify_model, args.yaml, device=device)
    
    #assert detect_model is not None and classify_model is not None, 'Invalid model path or file.'

    if args.save_dir is not None and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    results = []
    img_files = []
    if os.path.isdir(args.source):
        [img_files.extend(glob.glob(os.path.join(args.source, '**.' + ext))) for ext in IMG_EXT]
    else:
        img_files.append(args.source)
    processor = tqdm(img_files)
    processor.set_description('Processing')
    for i, path in enumerate(processor):
        img_src, boxes = detect(detect_model, path, dclass_names, stride=stride, device=device)
        boxes = filter_box(img_src, boxes)
        imgs = crop_img(img_src, boxes, 4)

        ret = [classify(classify_modelA, img, cclass_namesA, device=device) if len(box['label']) <= 2 \
               else classify(classify_modelB, img, cclass_namesB, device=device) for img, box in zip(imgs, boxes)]
        result = []
        for i, (data, box) in enumerate(zip(ret, boxes)):
            disease_type, conf, cls_logits = data[0], data[1], data[2]
            #box['label'] += ' ' + get_disease_str(disease_type)
            box['dlabel'] = get_disease_str(disease_type)
            #box['logits'] = [x * box['confidence'] for x in logits]
            class_cnt = len(cls_logits) // 2
            if class_cnt <= 3: 
                [cls_logits.insert(2, 0.) for _ in range(3)]
                [cls_logits.insert(8, 0.) for _ in range(3)]
            class_cnt = len(cls_logits) // 2
            #print(cls_logits)
            result.append({
                    "bone_type": box['label'],
                    "bone_idx": box['class_num'],
                    "disease_type": get_disease_str(disease_type), 
                    "disease_idx": (np.argmax(cls_logits[:class_cnt]), np.argmax(cls_logits[class_cnt:])),
                    "coord": get_center(box),
                    "detect_conf": box['confidence'],
                    "classify_conf": conf[0] * conf[1],
                    "detect_logits": box['logits'],
                    "classify_logits": (cls_logits[:class_cnt], cls_logits[class_cnt:]),
                })
        results.append({'file_path': path, 'result': result})

        if args.save_img:
            plot_boxes(img_src, 1, boxes)
            cv2.imwrite(os.path.join(args.save_dir, f'result_{i}.jpg'), img_src)

    LOGGER.info('Done.')
    LOGGER.debug(results)
    if args.save_txt:
        with open(os.path.join(args.save_dir, 'result.txt')) as f:
            for result in results:
                f.write(str(result))
    
    return results

def eval(args):
    results = predict(args)
    cfgs = load_yaml(args.yaml)

    # extract class_to_num_dict
    num_dict = {}
    [num_dict.update({name: i}) for i, name in enumerate(cfgs['names'])]
    dnum_dict = {}
    [dnum_dict.update({name: i}) for i, name in enumerate(cfgs['dnames2'])]

    # preprocess source data
    label_path = sorted(glob.glob(os.path.join(args.source, '**.json')))
    results = sorted(results, key=lambda x : x['file_path'])
    labels = []
    for path in label_path:
        with open(path, 'r') as f:
            labels.append(json.loads(f.read()))

    eval_res = evalutate(results, labels, num_dict, dnum_dict)
    LOGGER.info(f'Evaluation result: {eval_res}')
    return eval_res

def check_and_init(args):
    '''check config files and device.'''
    # check files
    master_process = args.rank == 0 if args.world_size > 1 else args.rank == -1
    if args.resume:
        # args.resume can be a checkpoint file path or a boolean value.
        checkpoint_path = args.resume if isinstance(args.resume, str) else find_latest_checkpoint()
        assert os.path.isfile(checkpoint_path), f'the checkpoint path is not exist: {checkpoint_path}'
        LOGGER.info(f'Resume training from the checkpoint file :{checkpoint_path}')
        resume_opt_file_path = Path(checkpoint_path).parent.parent / 'args.yaml'
        if osp.exists(resume_opt_file_path):
            with open(resume_opt_file_path) as f:
                args = argparse.Namespace(**yaml.safe_load(f))  # load args value from args.yaml
        else:
            LOGGER.warning(f'We can not find the path of {Path(checkpoint_path).parent.parent / "args.yaml"},'\
                           f' we will save exp log to {Path(checkpoint_path).parent.parent}')
            LOGGER.warning(f'In this case, make sure to provide configuration, such as data, batch size.')
            args.save_dir = str(Path(checkpoint_path).parent.parent)
        args.resume = checkpoint_path  # set the args.resume to checkpoint path.
    else:
        args.save_dir = str(increment_name(osp.join(args.output_dir, args.name)))
        if master_process:
            os.makedirs(args.save_dir)

    # check specific shape
    if args.specific_shape:
        if args.rect:
            LOGGER.warning('You set specific shape, and rect to True is needless. YOLOv6 will use the specific shape to train.')
        args.height = check_img_size(args.height, 32, floor=256)  # verify imgsz is gs-multiple
        args.width = check_img_size(args.width, 32, floor=256)
    else:
        args.img_size = check_img_size(args.img_size, 32, floor=256)

    cfg = Config.fromfile(args.conf_file)
    if not hasattr(cfg, 'training_mode'):
        setattr(cfg, 'training_mode', 'repvgg')
    # check device
    device = select_device(args.device)
    # set random seed
    set_random_seed(1+args.rank, deterministic=(args.rank == -1))
    # save args
    if master_process:
        save_yaml(vars(args), osp.join(args.save_dir, 'args.yaml'))

    return cfg, device, args

# TODO: Move this two functions to other files.
def train_yolov6(args):
    '''main function of training'''
    cfg, device, args = check_and_init(args)
    # reload envs because args was chagned in check_and_init(args)
    args.local_rank, args.rank, args.world_size = get_envs()
    LOGGER.info(f'training args are: {args}\n')
    if args.local_rank != -1: # if DDP mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        LOGGER.info('Initializing process group... ')
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", \
                init_method=args.dist_url, rank=args.local_rank, world_size=args.world_size,timeout=datetime.timedelta(seconds=7200))

    # Start
    trainer = Trainer(args, cfg, device)
    # PTQ
    if args.quant and args.calib:
        trainer.calibrate(cfg)
        return
    trainer.train()

    # End
    if args.world_size > 1 and args.rank == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()

def train_resnet(args):
    cfg = Config.fromfile(args.conf_file)
    LOGGER.info(f'training args are: {args}\n')
    try:
        device = torch.device(args.device)
    except:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_cfg = load_yaml(args.data_path)

    save_path = os.path.join(args.output_dir, f'{args.model}_{get_date_str()}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    modelA, modelB = None, None

    if args.pretrained is not None:
        modelA, class_namesA, modelB, class_namesB = get_resnet_model(args.pretrained, args.data_path, pretrained=args.pretrained)
    
    trainerA = resn_Trainer(data_cfg['ctrain1'], data_cfg['cval1'], data_cfg['dnc1'],
                           args.model, args.workers,
                           cfg.solver.optim, args.batch_size, modelA, device)

    trainerA.train(args.epochs, cfg.solver.lr, cfg.saver.save,
                  cfg.saver.save_every, cfg.saver.save_best, os.path.join(save_path, 'modelA'))
    
    trainerB = resn_Trainer(data_cfg['ctrain2'], data_cfg['cval2'], data_cfg['dnc2'],
                           args.model, args.workers,
                           cfg.solver.optim, args.batch_size, modelB, device)
    
    trainerB.train(args.epochs, cfg.solver.lr, cfg.saver.save,
                  cfg.saver.save_every, cfg.saver.save_best, os.path.join(save_path, 'modelB'))

def train_resnest(args):
    cfg = Config.fromfile(args.conf_file)
    LOGGER.info(f'training args are: {args}\n')
    try:
        device = torch.device(args.device)
    except:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_cfg = load_yaml(args.data_path)
    save_path = os.path.join(args.output_dir, f'{args.model}_{get_date_str()}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    modelA, modelB = None, None

    assert args.pretrained is None, 'resnest does not support pretrained model.'
    trainerA = resn_Trainer(data_cfg['ctrain1'], data_cfg['cval1'], data_cfg['dnc1'],
                           args.model, args.workers,
                           cfg.solver.optim, args.batch_size, modelA, device)

    trainerA.train(args.epochs, cfg.solver.lr, cfg.saver.save,
                  cfg.saver.save_every, cfg.saver.save_best, os.path.join(save_path, 'modelA'))
    
    trainerB = resnest_Trainer(data_cfg['ctrain2'], data_cfg['cval2'], data_cfg['dnc2'],
                           args.model, args.workers, cfg.solver.optim, cfg.model.final_drop, cfg.solver.last_gamma,
                           args.batch_size, modelB, device)

    trainerB.train(args.epochs, cfg.solver.lr, cfg.saver.save, cfg.solver.weight_decay,
                  cfg.solver.momentum, cfg.solver.lr_scheduler, cfg.solver.warmup_epochs,
                  cfg.saver.save_every, cfg.saver.save_best, os.path.join(save_path, 'modelB'))

def train_vit(args):
    cfg = Config.fromfile(args.conf_file)
    LOGGER.info(f'training args are: {args}\n')
    try:
        device = torch.device(args.device)
    except:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_cfg = load_yaml(args.data_path)

    save_path = os.path.join(args.output_dir, f'{args.model}_{get_date_str()}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = None

    if args.pretrained is not None:
        model, class_names = get_vit_model(args.pretrained, args.data_path, pretrained=args.pretrained)
    
    trainer = vit_Trainer(data_cfg['ctrain'], data_cfg['cval'], data_cfg['dnc'],
                           args.model, args.workers,
                           cfg.solver.optim, args.batch_size, model, device)

    trainer.train(args.epochs, cfg.solver.lr, cfg.saver.save,
                  cfg.saver.save_every, cfg.saver.save_best, save_path)

def train_swin(args):
    cfg = Config.fromfile(args.conf_file)
    LOGGER.info(f'training args are: {args}\n')
    try:
        device = torch.device(args.device)
    except:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    data_cfg = load_yaml(args.data_path)

    save_path = os.path.join(args.output_dir, f'{args.model}_{get_date_str()}')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    model = None

    if args.pretrained is not None:
        model, class_names = get_swin_model(args.pretrained, args.data_path, pretrained=args.pretrained)
    
    trainer = swin_Trainer(data_cfg['ctrain'], data_cfg['cval'], data_cfg['dnc'],
                           args.model, args.workers,
                           cfg.solver.optim, args.batch_size, model, device)

    trainer.train(args.epochs, cfg.solver.lr, cfg.saver.save,
                  cfg.saver.save_every, cfg.saver.save_best, save_path)

def train(args):
    # Setup
    args.local_rank, args.rank, args.world_size = get_envs()
    if 'yolo' in args.model:
        train_yolov6(args)
    elif 'resnet' in args.model:
        train_resnet(args)
    elif 'ViT' in args.model:
        train_vit(args)
    elif 'swin' in args.model:
        train_swin(args)
    elif 'resnest' in args.model:
        train_resnest(args)
    else:
        raise ValueError('Invalid type of model.')

if __name__ == '__main__':
    print(sys.path)
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