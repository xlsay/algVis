# coding=utf-8
'''
@File    :   main_ffpusher.py
@Time    :   2024/05/03
@Author  :   xlwang 
'''
import yaml
from PIL import ImageFont
import argparse
import os
from src.read_source import *
from src.push import *
from src.obj_det import *


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config.yaml', help='config file pah')
    params = parser.parse_args()
    with open(params.cfg) as f:
        cfgs = yaml.safe_load(f)
    params.source = cfgs['source']
    params.stream_out = cfgs['stream_out']
    params.max_side = cfgs['max_side']
    params.font = cfgs['font']
    params.det_weights = cfgs['Detetion']['weights']
    params.det_conf = cfgs['Detetion']['conf']
    params.det_color = cfgs['Detetion']['box_color']


    assert params.source!='' and params.stream_out!=''

    source_stream = Frame_Reader(params.source,frame_skip=2)
    source_h, source_w = source_stream.source_h, source_stream.source_w
    width, height = get_push_size(params.max_side,source_h,source_w)
    ffpusher = FFMPEG_PUSHER(width,height,params.stream_out)
    # process
    print(source_h, source_w)
    font = ImageFont.truetype(params.font, 15) # fontsize *max(width, height)
    det_marker = MARK_DET(params.det_weights, font, conf=params.det_conf, iou_thres=0.7, \
                        color=params.det_color)

    print('start pushing to ', params.stream_out)
    for frame in source_stream:
        if frame is None:
            print('ERROR: stream reading retrying...')
            break
        marked_frame = det_marker.do_mark([frame],show_conf=False)[0]
        push_frame = cv2.resize(marked_frame, (width,height), interpolation=cv2.INTER_LINEAR)
        if not ffpusher.pushing(push_frame):
            print('ERROR: stream pushing error!')
            break
    ffpusher.close_pipe()
