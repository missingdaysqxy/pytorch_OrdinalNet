# -*- coding: utf-8 -*-
# @Time    : 2019/1/8 2:08
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : utils.py
# @Software: PyCharm

import os
import sys
import logging
import json
import numpy as np
import torch as t
from warnings import warn
from visdom import Visdom
from time import strftime as timestr
from .config import Config


# Todo: Save/Load method: download env from server to local for transmit, and upload local env to server for share
# Todo: Start server through local code
class Visualizer(object):
    def __init__(self, config: Config):
        # logging_level = logging._checkLevel("INFO")
        # logging.getLogger().setLevel(logging_level)
        # VisdomServer.start_server(port=VisdomServer.DEFAULT_PORT, env_path=config.vis_env_path)
        self.reinit(config)

    def reinit(self, config):
        self.config = config
        try:
            self.visdom = Visdom(env=config.visdom_env)
            self.connected = self.visdom.check_connection()
            if not self.connected:
                print("Visdom server hasn't started, please run command 'python -m visdom.server' in terminal.")
                # try:
                #     print("Visdom server hasn't started, do you want to start it? ")
                #     if 'y' in input("y/n: ").lower():
                #         os.popen('python -m visdom.server')
                # except Exception as e:
                #     warn(e)
        except ConnectionError as e:
            warn("Can't open Visdom because " + e.strerror)
        with open(self.config.log_file, 'a') as f:
            info = "[{time}]Initialize Visdom\n".format(time=timestr('%m-%d %H:%M:%S'))
            info += str(self.config)
            f.write(info + '\n')

    def save(self, save_path: str = None) -> str:
        retstr = self.visdom.save([self.config.visdom_env])  # return current environments name in format of json
        try:
            ret = json.loads(retstr)[0]
            if ret == self.config.visdom_env:
                if isinstance(save_path, str):
                    from shutil import copy
                    copy(self.config.vis_env_path, save_path)
                    print('Visdom Environment has saved into ' + save_path)
                else:
                    print('Visdom Environment has saved into ' + self.config.vis_env_path)
                with open(self.config.vis_env_path, 'r') as fp:
                    env_str = json.load(fp)
                    return env_str
        except:
            pass
        return None

    def clear(self):
        self.visdom.close()

    @staticmethod
    def _to_numpy(value):
        if isinstance(value, t.Tensor):
            value = value.cpu().detach().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            value = np.array(value)
        if value.ndim == 0:
            value = value[np.newaxis]
        return value

    def plot(self, y, x, line_name, win, legend=None):
        # type:(float,float,str,str,list(str))->None
        """Plot a (sequence) of y point(s) (each) with one x value(s), loop this method to draw whole plot"""
        update = None if not self.visdom.win_exists(win) else 'append'
        opts = dict(title=win)
        if legend is not None:
            opts["legend"] = legend
        y = Visualizer._to_numpy(y)
        x = Visualizer._to_numpy(x)
        return win == self.visdom.line(y, x, win=win, env=self.config.visdom_env,
                                       update=update, name=line_name, opts=opts)

    def bar(self, y, win, rowindices=None):
        opts = dict(title=win)
        y = Visualizer._to_numpy(y)
        if isinstance(rowindices, list) and len(rowindices) == len(y):
            opts["rownames"] = rowindices
        return win == self.visdom.bar(y, win=win, env=self.config.visdom_env, opts=opts)

    def log(self, msg, name, append=True, log_file=None):
        # type:(str,str,bool,bool,str)->None
        if log_file is None:
            log_file = self.config.log_file
        info = "[{time}]{msg}".format(time=timestr('%m-%d %H:%M:%S'), msg=msg)
        append = append and self.visdom.win_exists(name)
        ret = self.visdom.text(info, win=name, env=self.config.visdom_env, opts=dict(title=name), append=append)
        mode = 'a+' if append else 'w+'
        with open(log_file, mode) as f:
            f.write(info + '\n')
        return ret == name

    def log_process(self, num, total, msg, name, append=True):
        # type:(int,int,str,Visdom,str,str,dict,bool)->None
        info = "[{time}]{msg}".format(time=timestr('%m-%d %H:%M:%S'), msg=msg)
        append = append and self.visdom.win_exists(name)
        ret = self.visdom.text(info, win=(name), env=self.config.visdom_env, opts=dict(title=name), append=append)
        with open(self.config.log_file, 'a') as f:
            f.write(info + '\n')
        self.processBar(num, total, msg)
        return ret == name

    def processBar(self, num, total, msg='', length=50):
        rate = num / total
        rate_num = int(rate * 100)
        clth = int(rate * length)
        if len(msg) > 0:
            msg += ':'
        # msg = msg.replace('\n', '').replace('\r', '')
        if rate_num == 100:
            r = '\r%s[%s%d%%]\n' % (msg, '*' * length, rate_num,)
        else:
            r = '\r%s[%s%s%d%%]' % (msg, '*' * clth, '-' * (length - clth), rate_num,)
        sys.stdout.write(r)
        sys.stdout.flush
        return r.replace('\r', ':')
