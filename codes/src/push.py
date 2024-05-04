# coding=utf-8
'''
@File    :   push.py
@Time    :   2024/05/03
@Author  :   xlwang 
'''
import subprocess as sp
from subprocess import TimeoutExpired

class FFMPEG_PUSHER(object):
    def __init__(self,width,height,push_url):
        self.width, self.height = width, height
        self.push_url = push_url
        self.pipe = self.new_pipe()
        self.cnt_error,self.error_thresh = 0,5


    def new_pipe(self):
        # '/usr/bin/ffmpeg'
        push_cmd = ['/usr/bin/ffmpeg', '-an',
                '-f', 'rawvideo', 
                '-vcodec','rawvideo', 
                '-pix_fmt', 'bgr24', 
                '-s', '{}x{}'.format(self.width, self.height),
                '-re',
                '-i', '-', 
                '-c:v', 'libx264', 
                '-b:v', '1M', # 平均编码码率 
                '-minrate:v', '2000k', # 最小编码码率 
                '-maxrate:v', '2M', 
                '-bufsize', '2M', # 解码缓冲大小 
                '-pix_fmt', 'yuv420p', 
                '-preset', 'ultrafast', 
                '-profile:v', 'baseline', 
                '-f', 'flv', 
                '-loglevel', 'quiet', 
                self.push_url,]
        pipe = sp.Popen(push_cmd, stdin=sp.PIPE, bufsize=0)
        # bufsize<0, bufsize=系统默认值 io.DEFAULT_BUFFER_SIZE (当前8192)
        return pipe


    def pushing(self, push_frame):
        try:
            self.pipe.stdin.write(push_frame.tobytes())
            self.cnt_error = 0
        except BaseException as e:
            if isinstance(e, KeyboardInterrupt): 
                return False
            print('WARNING: stdin exception with code:{}'.format(self.pipe.poll())) 
            self.cnt_error += 1
            if self.cnt_error > self.error_thresh:
                print('ERROR:  please check srs server.')
                return False
            self.close_pipe()
            self.pipe = self.new_pipe()
        return True


    def close_pipe(self):
        try:
            self.pipe.communicate(timeout=5)
        except TimeoutExpired:
            self.pipe.kill()
            self.pipe.communicate()
        self.pipe.stdin.close()

