# coding=utf-8
'''
@File    :   utils.py
@Time    :   2024/05/03
@Author  :   xlwang 
'''
def get_push_size(max_side, frame_h, frame_w):
    if max_side>0:
            rz = max_side/max(frame_h,frame_w)
    else:
        rz = 1
    width,height = int(frame_w*rz), int(frame_h*rz) # push size
    width = width if width%2==0 else width+1
    height = height if height%2==0 else height+1
    return width, height