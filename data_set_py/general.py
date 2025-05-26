import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
import matplotlib


# 将灰度数组映射为直方图字典,nums表示灰度的数量级
def arrayToHist(grayArray, nums):
    if len(grayArray.shape) != 2:
        print("length error")
        return None
    w, h = grayArray.shape
    hist = {}
    for k in range(nums):
        hist[k] = 0
    for i in range(w):
        for j in range(h):
            if hist.get(grayArray[i][j]) is None:
                hist[grayArray[i][j]] = 0
            hist[grayArray[i][j]] += 1
    # normalize
    n = w * h
    for key in hist.keys():
        hist[key] = float(hist[key]) / n
    return hist


# 计算累计直方图计算出新的均衡化的图片，nums为灰度数,256
def equalization(grayArray, h_s, nums):
    # 计算累计直方图
    tmp = 0.0
    h_acc = h_s.copy()
    for i in range(256):
        tmp += h_s[i]
        h_acc[i] = tmp

    if len(grayArray.shape) != 2:
        print("length error")
        return None
    w, h = grayArray.shape
    des = np.zeros((w, h), dtype=np.uint8)
    for i in range(w):
        for j in range(h):
            des[i][j] = int((nums - 1) * h_acc[grayArray[i][j]] + 0.5)
    return des


# 传入的直方图要求是个字典，每个灰度对应着概率
def drawHist(hist, name):
    keys = hist.keys()
    values = hist.values()
    x_size = len(hist) - 1  # x轴长度，也就是灰度级别
    axis_params = [0, x_size]

    # plt.figure()
    if name is not None:
        plt.title(name)
    plt.bar(tuple(keys), tuple(values))  # 绘制直方图
    # plt.show()


# 直方图匹配函数，接受原始图像和目标灰度直方图
def histMatch(grayArray, dest_array):
    h_d = arrayToHist(dest_array, 256)
    # 计算累计直方图
    tmp = 0.0
    h_acc = h_d.copy()
    for i in range(256):
        tmp += h_d[i]
        h_acc[i] = tmp

    h1 = arrayToHist(grayArray, 256)
    tmp = 0.0
    h1_acc = h1.copy()
    for i in range(256):
        tmp += h1[i]
        h1_acc[i] = tmp
    # 计算映射
    M = np.zeros(256)
    for i in range(256):
        idx = 0
        minv = 1
        for j in h_acc:
            if np.fabs(h_acc[j] - h1_acc[i]) < minv:
                minv = np.fabs(h_acc[j] - h1_acc[i])
                idx = int(j)
        M[i] = idx
    des = M[grayArray]
    return des
