# coding=utf-8
from flask import Flask,render_template, Response
import yaml
from PIL import ImageFont
import argparse
import threading
import os
import time
import cv2
from src.read_source import *
from src.obj_det import *
from src.utils import *

app = Flask(__name__)
orig_frame, alg_frame = None, None
sreams_frames = [orig_frame, alg_frame]

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

def gen_frame(fi):
    while True:
        frame = sreams_frames[fi]
        if frame is None:
            time.sleep(0.1)
            continue
        frame = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/stream_orig')
def stream_orig():
    return Response(gen_frame(0),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/stream_alg')
def stream_alg():
    return Response(gen_frame(1),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

def stream_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='config.yaml', help='config file pah')
    params = parser.parse_args()
    with open(params.cfg) as f:
        cfgs = yaml.safe_load(f)
    params.source = cfgs['source']
    params.max_side = cfgs['max_side']
    params.font = cfgs['font']
    params.det_weights = cfgs['Detetion']['weights']
    params.det_conf = cfgs['Detetion']['conf']
    params.det_color = cfgs['Detetion']['box_color']
    assert params.source!=''


    # create processor
    font = ImageFont.truetype(params.font, 15)
    det_marker = MARK_DET(params.det_weights, font, conf=params.det_conf, iou_thres=0.7, \
                        color=params.det_color)
    
    global sreams_frames
    while True:
        source_stream = Frame_Reader(params.source)
        source_h, source_w = source_stream.source_h, source_stream.source_w
        # print('orig size(hxw): {}x{}'.format(source_h, source_w))
        width, height = get_push_size(params.max_side,source_h,source_w)
        for frame in source_stream:
            if frame is None:
                print('ERROR: stream reading retrying...')
                break
            marked_frame = det_marker.do_mark([frame],show_conf=False)[0]
            sreams_frames[0] = cv2.resize(frame, (width,height), interpolation=cv2.INTER_LINEAR)
            sreams_frames[1] = cv2.resize(marked_frame, (width,height), interpolation=cv2.INTER_LINEAR)
        time.sleep(2)


if __name__=='__main__':
    gen_thread = threading.Thread(target=stream_gen) 
    gen_thread.setDaemon(True); gen_thread.start()

    app.debug = False
    app.run(host='0.0.0.0', port=5011, threaded=True)