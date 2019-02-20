from libsvm.python.svmutil import *
from libsvm.python.svm import *
import os
import struct
import numpy
dic={}
#数据加载函数，kind值标明了读取文件的类型
def loadforSVM(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = numpy.fromfile(lbpath,dtype=numpy.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = numpy.fromfile(imgpath,dtype=numpy.uint8).reshape(len(labels), 784)
    #由于源数据有些数据过大，会导致激活函数计算溢出，所以对数据集集体缩小，
    #由于图片数据每一位的值均为0-255之间，归一化处理
    if kind=='train':
    	f = open('trainforSVM.txt','w')
    if kind=='t10k':
    	f = open('testforSVM.txt','w')
    count=0
    for i in range(10):
    	for j in range(len(images)):
    		index=1
    		if labels[j]==i:
    			string=str(i)+' '
    			for k in images[j]:
    				string=string+str(index)+':'+str(k/255)+' '
    				index+=1
    			f.writelines(string+'\n')
    			dic[count]=j
    			count+=1
    f.close()
if __name__ == '__main__': 
    loadforSVM("C:\\Users\\Anonymous\\Documents\\机器学习\\作业四赵虎201600301325", kind='train')
    loadforSVM("C:\\Users\\Anonymous\\Documents\\机器学习\\作业四赵虎201600301325", kind='t10k')
    y, x = svm_read_problem('trainforSVM.txt')
    yt,xt=svm_read_problem('testforSVM.txt')
    model=svm_train(y,x,'-t 0 -m 600')
    # print('test:')
    p_label, p_acc, p_val = svm_predict(yt, xt, model)
    f = open('classificationforSVM.txt','w')
    for i in range(len(p_label)):
    	# f.write(str(int(p_label[dic[i]]))+' ')
    	f.write(str(int(p_label[i]))+' ')
    f1=open("classificationforSVM.txt")
    s=f1.read().split()
    dic1={}
    for i in range(10000):
        dic1[dic[i]]=i
    f2=open("classificationforlinearSVM.txt",'w')
    for i in range(10000):
        f2.write(s[dic1[i]]+' ')


