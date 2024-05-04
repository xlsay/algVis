# coding=utf-8
'''
@File    :   obj_det.py
@Time    :   2024/05/03
@Author  :   xlwang 
'''
import cv2
from PIL import Image, ImageDraw
from ultralytics import YOLO
import numpy as np
import torch
import random
import os
# os.environ['CUDA_VISIBLE_DEVICES']='0'

class MARK_DET(object):
    def __init__(self,weights,font,conf=0.2, iou_thres=0.7,
                color=(0,255,0)):
        self.model = YOLO(weights,task='detect')
        names = self.model.names
        self.labels = [names[i] for i in range(len(names))]
        color = [tuple([np.clip(int(ci),0,255) for ci in c]) for c in color if len(c)==3]
        if len(color)==len(names):
            self.color = color
        else:
            self.color = color+\
                        [tuple([random.randint(0, 255) for _ in range(3)]) for _1 in range(len(names)-len(color))]
        self.font = font
        self.conf,self.iou_thres = conf, iou_thres
        # e.g., 'cpu', 'cuda:0' or '0'
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.imgsz = 640
        # warmup
        self.model.predict(source=[np.random.randint(0, 255, size=(640,640,3), dtype=np.uint8)],\
                        conf=self.conf, imgsz=self.imgsz,device=self.device,iou=self.iou_thres,\
                        verbose=False)


    def do_mark(self,frame_list,show_conf=True):
        num_frame = len(frame_list)
        none_inds = [i for i,frame in enumerate(frame_list) if (frame is None)]
        frames = [frame_list[i] for i in range(num_frame) if i not in none_inds]
        assert len(frames)>0
        # Inference & Mark
        marked = []
        with torch.no_grad():
            preds = self.model.predict(source=frames,conf=self.conf,\
                    imgsz=self.imgsz,device=self.device,iou=self.iou_thres,
                    verbose=False)
            pred_ind = 0
            for i in range(num_frame):
                frame = frame_list[i]
                if i in none_inds:
                    marked.append(None)
                else:
                    pred = preds[pred_ind]; pred_ind+=1
                    if len(pred)==0:
                        marked.append(frame)
                        continue
                    xyxys = pred.boxes.xyxy.cpu().numpy().astype(int)
                    clss = pred.boxes.cls.cpu().numpy().astype(int)
                    confs = pred.boxes.conf.cpu().numpy()
                    img_pil = Image.fromarray(frame)
                    draw = ImageDraw.Draw(img_pil)
                    for bi,xyxy in enumerate(xyxys):
                        cls_ind = clss[bi]
                        draw.rectangle(xyxy.tolist(), width=2, outline=self.color[cls_ind])
                        if show_conf:
                            label = '{}_{:.2f}'.format(self.labels[cls_ind],confs[bi])
                        else:
                            label = self.labels[cls_ind]
                        txt_width, txt_height = self.font.getsize(label)
                        draw.rectangle([xyxy[0], xyxy[1]-txt_height+4, xyxy[0]+txt_width, xyxy[1]], fill=tuple(self.color[cls_ind]))
                        draw.text((xyxy[0], xyxy[1]-txt_height+1), label, fill=(255,255,255), font=self.font)
                    marked.append(np.asarray(img_pil))
                    #  marked.append(pred.plot())
        return marked

