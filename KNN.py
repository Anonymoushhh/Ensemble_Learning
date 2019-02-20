# -*- coding: utf-8 -*-
from numpy import *
import operator
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir
from mpl_toolkits.mplot3d import Axes3D
import struct

#读取图片
def read_image(file_name):
    #先用二进制方式把文件都读进来
    file_handle=open(file_name,"rb")  #以二进制打开文档
    file_content=file_handle.read()   #读取到缓冲区中

    offset=0
    head = struct.unpack_from('>IIII', file_content, offset)  # 取前4个整数，返回一个元组
    offset += struct.calcsize('>IIII')
    imgNum = head[1]  #图片数
    rows = head[2]   #宽度
    cols = head[3]  #高度
    # print(imgNum)
    # print(rows)
    # print(cols)

    #测试读取一个图片是否读取成功
    #im = struct.unpack_from('>784B', file_content, offset)
    #offset += struct.calcsize('>784B')

    images=np.empty((imgNum , 784))#empty，是它所常见的数组内的所有元素均为空，没有实际意义，它是创建数组最快的方法
    image_size=rows*cols#单个图片的大小
    fmt='>' + str(image_size) + 'B'#单个图片的format

    for i in range(imgNum):
        images[i] = np.array(struct.unpack_from(fmt, file_content, offset))
        # images[i] = np.array(struct.unpack_from(fmt, file_content, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt)
    return images

    '''bits = imgNum * rows * cols  # data一共有60000*28*28个像素值
    bitsString = '>' + str(bits) + 'B'  # fmt格式：'>47040000B'
    imgs = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    imgs_array=np.array(imgs).reshape((imgNum,rows*cols))     #最后将读取的数据reshape成 【图片数，图片像素】二维数组
    return imgs_array'''

#读取标签
def read_label(file_name):
    file_handle = open(file_name, "rb")  # 以二进制打开文档
    file_content = file_handle.read()  # 读取到缓冲区中

    head = struct.unpack_from('>II', file_content, 0)  # 取前2个整数，返回一个元组
    offset = struct.calcsize('>II')

    labelNum = head[1]  # label数
    # print(labelNum)
    bitsString = '>' + str(labelNum) + 'B'  # fmt格式：'>47040000B'
    label = struct.unpack_from(bitsString, file_content, offset)  # 取data数据，返回一个元组
    return np.array(label)

#KNN算法
def KNN(test_data, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]#dataSet.shape[0]表示的是读取矩阵第一维度的长度，代表行数
    # distance1 = tile(test_data, (dataSetSize,1)) - dataSet#欧氏距离计算开始
    # print("dataSetSize:")
    # print(dataSetSize)
    distance1 = tile(test_data, (dataSetSize)).reshape((60000,784))-dataSet#tile函数在行上重复dataSetSizec次，在列上重复1次
    # print("distance1.shape")
    # print(distance1.shape)
    distance2 = distance1**2 #每个元素平方
    distance3 = distance2.sum(axis=1)#矩阵每行相加
    distances4 = distance3**0.5#欧氏距离计算结束
    # print(distances4[53843])
    # print(distances4[38620])
    # print(distances4[16186])
    sortedDistIndicies = distances4.argsort() #返回从小到大排序的索引
    classCount=np.zeros((10), np.int32)#10是代表10个类别
    for i in range(k): #统计前k个数据类的数量
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] += 1
    max = 0
    id = 0
    # print(classCount.shape[1])

    for i in range(classCount.shape[0]):
        if classCount[i] >= max:
            max = classCount[i]
            id = i

    # sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)#从大到小按类别数目排序
    return id

def test_KNN():
    # 文件获取
    #mnist数据集
    # train_image = "F:\mnist\\train-images-idx3-ubyte"
    # test_image = "F:\mnist\\t10k-images-idx3-ubyte"
    # train_label = "F:\mnist\\train-labels-idx1-ubyte"
    # test_label = "F:\mnist\\t10k-labels-idx1-ubyte"
    #fashion mnist数据集
    train_image = "train-images.idx3-ubyte"
    test_image = "t10k-images.idx3-ubyte"
    train_label = "train-labels.idx1-ubyte"
    test_label = "t10k-labels.idx1-ubyte"
    # 读取数据
    train_x = read_image(train_image)  # train_dataSet
    test_x = read_image(test_image)  # test_dataSet
    train_y = read_label(train_label)  # train_label
    test_y = read_label(test_label)  # test_label
    #调试的时候让速度快点，就先减少数据集大小
    train_x=train_x[0:60000,:]
    train_y=train_y[0:60000]
    test_x=test_x[0:10000,:]
    test_y=test_y[0:10000]

    testRatio = 1  # 取数据集的前0.1为测试数据,这个参数比重可以改变
    train_row = train_x.shape[0]  # 数据集的行数，即数据集的总的样本数
    test_row=test_x.shape[0]
    testNum = int(test_row * testRatio)
    errorCount = 0  # 判断错误的个数
    results=[]
    for i in range(testNum):
        result = KNN(test_x[i], train_x, train_y, 10)
        results.append(result)
        if result != test_y[i]:
            errorCount += 1.0# 如果mnist验证集的标签和本身标签不一样，则出错
    error_rate = errorCount / float(testNum)  # 计算出错率
    acc = 1.0 - error_rate
    print('test end')
    f = open('classificationforKNN.txt','w')
    for i in results:
        f.write(str(i)+' ')
    f.close()
    print("\nthe total number of errors is: %d" % errorCount)
    print("\nthe total error rate is: %f" % (error_rate))
    print("\nthe total accuracy rate is: %f" % (acc))

if __name__ == "__main__":
    test_KNN()#test(）函数中调用了读取数据集的函数，并调用分类函数对数据集进行分类，最后对分类情况进行计算

    
