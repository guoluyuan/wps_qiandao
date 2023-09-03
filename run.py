# -*- coding: utf-8 -*-
# @Time : 2023/8/27 15:51
# @Author : 郭盖
# @Email : wayne_lau@aliyun.com
# @File : wps领空间签到.py
import json
import time

import requests
import sklearn.mixture
from PIL import Image, ImageFont, ImageDraw
import numpy

from urllib.parse import quote
import torch
import torch.nn as nn
import torch.nn.functional as F


def CAPTCHA_to_data(filename, width, height):
    '''
    convert CAPTCHA image to 7 chinese character image data.
    kind of slow because of GMM iteration.
    return a 7 * 40 * 40 array
    '''
    padding = 20
    padding_color = 249

    captcha = Image.open(filename)

    bg = numpy.full((height + padding * 2, width + padding * 2), padding_color, dtype='uint8')
    fr = numpy.asarray(captcha.convert('L'))
    bg[padding:padding + height, padding:padding + width] = fr

    black_pixel_indexes = numpy.transpose(numpy.nonzero(bg <= 150))
    gmm = sklearn.mixture.GaussianMixture(n_components=5, covariance_type='tied', reg_covar=1e2, tol=1e3, n_init=9)
    gmm.fit(black_pixel_indexes)

    indexes = gmm.means_.astype(int).tolist()
    new_indexes = []
    for [y, x] in indexes:
        new_indexes.append((y - padding, x - padding))

    data = numpy.empty((0, 40, 40), 'float32')
    full_image = data_to_image(bg)

    for [y, x] in new_indexes:
        cim = full_image.crop((x, y, x + padding * 2, y + padding * 2))
        X = numpy.asarray(cim.convert('L')).astype('float32')
        X[X <= 150] = -1
        # black
        X[X > 150] = 1
        # white
        data = numpy.append(data, X.reshape(1, 40, 40), axis=0)

    return data, new_indexes


def mark_points(image, points):
    '''
    mark locations on image
    '''

    im = image.convert("RGB")
    bgdr = ImageDraw.Draw(im)
    for [y, x] in points:
        bgdr.ellipse((x - 3, y - 3, x + 3, y + 3), fill="red", outline='red')
    return im


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 32, 3, padding=1)
        self.fc1 = nn.Linear(32 * 25, 40)
        self.fc2 = nn.Linear(40, 2)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = F.max_pool2d(F.relu(self.conv3(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


def data_to_image(d):
    '''
    convert 2darray to image object.
    '''

    return Image.fromarray(numpy.uint8(d))


# Load net from file on CPU.


def predict_result(filename, width, height):
    '''
    given a captcha image file,
    return the upside-down character indexes.
    '''

    data, indexes = CAPTCHA_to_data(filename, width, height)
    inputs = torch.from_numpy(data.reshape(5, 1, 40, 40))
    outputs = net(inputs)
    _, predicted = torch.max(outputs.data, 1)
    predicted = predicted.tolist()

    return [i for (i, p) in zip(indexes, predicted) if not p]


def get_daoli_xy_list(filename, width, height):
    ps = predict_result(filename, width, height)
    ps = [(y, x) for (x, y) in ps]
    return ps


def getnow():
    t = time.time()
    return str(int(round(t * 1000)))


def get_captcha_pos(sid):
    headers = {
        # "user-agent": "Mozilla/5.0 (Linux; Android 12; Redmi K30 Pro Build/SKQ1.211006.001; wv) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/111.0.5563.116 Mobile Safari/537.36 XWEB/5235 MMWEBSDK/20230701 MMWEBID/9650 MicroMessenger/8.0.40.2420(0x28002855) WeChat/arm64 Weixin NetType/5G Language/zh_CN ABI/arm64",
        "accept": "image/wxpic,image/tpg,image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
        "cookie": f"wps_sid={sid}"

    }

    url=f"https://vip.wps.cn/checkcode/signin/captcha.png?platform=8&encode=0&img_witdh=336&img_height=84&v={getnow()}"
    content = requests.get(url=url, headers=headers).content
    # print(content)
    with open(f"./captcha.jpg", "wb") as f:
        f.write(content)
    zuobiao_list = get_daoli_xy_list(f"./captcha.jpg", 350, 88)
    captcha_pos = '|'.join([f"{x},{y}" for x, y in zuobiao_list])
    return captcha_pos


def send(msg):
    if token=="":
        return
    #使用PUSHPLUS发送消息
    url = f"http://www.pushplus.plus/send/{token}?title=wps空间签到&content={msg}"
    requests.get(url=url)
    pass

# 签到领空间
def sign_kongjian(captcha_pos,sid,name):
    headers = {
        "cookie": f"wps_sid={sid}",
        "content-type": "application/x-www-form-urlencoded",
        "accept": "*/*",
    }
    url = "https://vip.wps.cn/sign/v2"
    body = f"platform=8&captcha_pos={quote(captcha_pos)}&img_witdh=336&img_height=84"
    print(body)
    ret = requests.post(url=url, headers=headers, data=body).json()
    print(ret)
    if ret['result'] == "ok":
        # send(f"{name}:签到成功")
        qiandao_msg.append(f"{name}:签到成功")
        print("签到成功")
    elif ret["msg"] == "已完成签到" or ret['msg']=="10003":
        qiandao_msg.append(f"{name}:已经签到成功")
        print("签到成功")

if __name__ == "__main__":
    #这里zheye.pt文件的路径填写绝对路径(看看你服务器具体情况填写)
    zheye_pt_path=""
    #使用cpu运行，方便没有gpu放入服务器运行
    net = torch.load(zheye_pt_path, map_location=torch.device('cpu'))
    net.eval()
    #push_token，可以不填
    # 签到成功列表
    qiandao_msg = []
    token = ""
    #签到字典{"备注":"wps_sid"}
    sid_list = {
        "备注":"wps_sid",#有几个写几个一行一个
    }
    #遍历sid，签到
    for name in sid_list:
        sid=sid_list[name]
        #获取识别并拼接好的captcha_pos
        captcha_pos = get_captcha_pos(sid)
        print(captcha_pos)
        #签到
        sign_kongjian(captcha_pos,sid,name)
    #不填push的注释下面一行
    send("\n".join(qiandao_msg))

