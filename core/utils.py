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
from warnings import warn
from visdom import Visdom, server as VisdomServer
from time import strftime as timestr
from core.config import Config

#Todo: Save/Load method; about connected
class Visualizer(object):
    def __init__(self, config: Config):
        logging_level = logging._checkLevel("INFO")
        logging.getLogger().setLevel(logging_level)
        VisdomServer.start_server(port=VisdomServer.DEFAULT_PORT, env_path=config.vis_env_path)
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
        with open(self.config.log_path, 'a') as f:
            info = "[{time}]Initialize Visdom\n".format(time=timestr('%m-%d %H:%M:%S'))
            info+=str(self.config)
            f.write(info + '\n')

    def save(self, save_path: str = None) -> str:
        retstr = self.visdom.save([self.config.visdom_env])
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

    def plot(self, y, x, name):
        # type:(int,int,str,Config)->None
        update = None if x == 0 or not self.visdom.win_exists(name) else 'append'
        return name == self.visdom.line(np.array([y]), np.array([x]), (name), self.config.visdom_env, dict(title=name),
                                        update)

    def log(self, msg, name, append=True):
        # type:(str,str,bool,bool)->None
        info = "[{time}]{msg}".format(time=timestr('%m-%d %H:%M:%S'), msg=msg)
        append = append and self.visdom.win_exists(name)
        ret = self.visdom.text(info, win=(name), env=self.config.visdom_env, opts=dict(title=name), append=append)
        with open(self.config.log_path, 'a') as f:
            f.write(info + '\n')
        return ret == name

    def log_process(self, num, total, msg, name, append=True):
        # type:(int,int,str,Visdom,str,str,dict,bool)->None
        info = "[{time}]{msg}".format(time=timestr('%m-%d %H:%M:%S'), msg=msg)
        append = append and self.visdom.win_exists(name)
        ret = self.visdom.text(info, win=(name), env=self.config.visdom_env, opts=dict(title=name), append=append)
        with open(self.config.log_path, 'a') as f:
            f.write(info + '\n')
        self.processBar(num, total, msg)
        return ret == name

    def processBar(self, num, total, msg='', length=50):
        rate = num / total
        rate_num = int(rate * 100)
        clth = int(rate * length)
        if len(msg) > 0:
            msg += ':'
        msg = msg.replace('\n', '').replace('\r', '')
        if rate_num == 100:
            r = '\r%s[%s%d%%]\n' % (msg, '*' * length, rate_num,)
        else:
            r = '\r%s[%s%s%d%%]' % (msg, '*' * clth, '-' * (length - clth), rate_num,)
        sys.stdout.write(r)
        sys.stdout.flush
        return r.replace('\r', ':')