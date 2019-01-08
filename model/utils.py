# -*- coding: utf-8 -*-
# @Time    : 2019/1/8/008 2:08 上午
# @Author  : LQX
# @Email   : qixuan.lqx@qq.com
# @File    : utils.py
# @Software: PyCharm

import sys
import numpy as np
from visdom import Visdom
from time import strftime as timestr


def plot(y, x, visdom, name, env=None):
    # type:(int,int,Visdom,str,str)->None
    update = None if x == 0 or not visdom.win_exists(name) else 'append'
    return visdom.line(np.array([y]), np.array([x]), (name), env, dict(title=name), update)


def log(msg, visdom, name, env=None, append=True, logfile='logfile.txt'):
    # type:(str,Visdom,str,str,dict,bool)->None
    info = "[{time}]{msg}".format(time=timestr('%m-%d %H:%M:%S'), msg=msg)
    append = append and visdom.win_exists(name)
    ret = visdom.text(info, win=(name), env=env, opts=dict(title=name), append=append)
    with open(logfile, 'a') as f:
        f.write(info + '\n')
    return ret


def log_process(num, total, msg, visdom, name, env=None, append=True, logfile='logfile.txt'):
    # type:(int,int,str,Visdom,str,str,dict,bool)->None
    info = "[{time}]{msg}".format(time=timestr('%m-%d %H:%M:%S'), msg=msg)
    append = append and visdom.win_exists(name)
    ret = visdom.text(info, win=(name), env=env, opts=dict(title=name), append=append)
    with open(logfile, 'a') as f:
        f.write(info + '\n')
    processBar(num, total, msg)
    return ret


def processBar(num, total, msg='', length=50):
    rate = num / total
    rate_num = int(rate * 100)
    clth = int(rate * length)
    if len(msg) > 0:
        msg += ':'
    if rate_num == 100:
        r = '\r%s[%s%d%%]\n' % (msg, '*' * length, rate_num,)
    else:
        r = '\r%s[%s%s%d%%]\n' % (msg, '*' * clth, '-' * (length - clth), rate_num,)
    sys.stdout.write(r)
    sys.stdout.flush
    return r.replace('\r', ':')
#
# def processBar(num, total, msg='', length=50):
#     rate = num / total
#     rate_num = int(rate * 100)
#     clth = int(rate * length)
#     if len(msg) > 0:
#         msg += ':'
#     if rate_num == 100:
#         r = '\r%s[%s%d%%]\n' % (msg, '*' * length, rate_num,)
#     else:
#         r = '\r%s[%s%s%d%%]' % (msg, '*' * clth, '-' * (length - clth), rate_num,)
#     sys.stdout.write(r)
#     sys.stdout.flush
#     return r.replace('\r', ':')
