# coding=utf-8
'''
@File    :   read_source.py
@Time    :   2024/05/03
@Author  :   xlwang 
'''
import os
import cv2
import glob
import numpy as np
from pathlib import Path

IMG_FORMATS = 'bmp', 'dng', 'jpeg', 'jpg', 'mpo', 'png', 'tif', 'tiff', 'webp'  # include image suffixes
VID_FORMATS = 'asf', 'avi', 'gif', 'm4v', 'mkv', 'mov', 'mp4', 'mpeg', 'mpg', 'ts', 'wmv'  # include video suffixes

class Frame_Reader(object):
    # refer to: YOLOv5 image/video dataloader
    def __init__(self, path,frame_skip=0):
        self.is_url = path.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        if self.is_url:
            images = []
            videos = [path]
        else:
            p = str(Path(path).resolve())  # os-agnostic absolute path
            if '*' in p:
                files = sorted(glob.glob(p, recursive=True))  # glob
            elif os.path.isdir(p):
                files = sorted(glob.glob(os.path.join(p, '*.*')))  # dir
            elif os.path.isfile(p):
                files = [p]  # files
            else:
                raise Exception(f'ERROR: {p} does not exist')
            images = [x for x in files if x.split('.')[-1].lower() in IMG_FORMATS]
            videos = [x for x in files if x.split('.')[-1].lower() in VID_FORMATS]
        ni, nv = len(images), len(videos)

        self.files = images + videos
        self.nf = ni + nv  # number of files
        self.video_flag = [False]*ni + [True]*nv
        if self.nf<=0:
            print(f'ERROR:No images or videos found in {p}. ' \
                            f'Supported formats are:\nimages: {IMG_FORMATS}\nvideos: {VID_FORMATS}')
            return None
        if any(videos):
            self.new_video(videos[0])  # new video
            self.frame_skip = 1 + int(frame_skip)
        else:
            self.cap = None
            testimg = cv2.imread(images[0])
            self.source_h,self.source_w = testimg.shape[:2]

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        if self.count==self.nf:
            raise StopIteration
        path = self.files[self.count]
        if self.video_flag[self.count]:
            # Read video
            for _ in range(self.frame_skip):
                self.cap.grab()
            ret_val, frame = self.cap.retrieve()
            while not ret_val:
                self.count = self.count if self.is_url else self.count+1
                self.cap.release()
                if self.count==self.nf:  # last video
                    raise StopIteration
                else:
                    path = self.files[self.count]
                    self.new_video(path)
                    ret_val, frame = self.cap.read()
            self.frame += 1
        else:
            # Read image
            self.count += 1
            frame = cv2.imread(path)  # BGR
            if frame is None:
                print(f'ERROR: Image Not Found {path}')
        # frame = np.ascontiguousarray(frame)
        return frame


    def new_video(self, path):
        self.frame = 0
        self.cap = cv2.VideoCapture(path)
        self.source_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
        self.source_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))


    def __len__(self):
        return self.nf  # number of files
