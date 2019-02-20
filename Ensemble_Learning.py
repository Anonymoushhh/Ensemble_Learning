import os
import struct
import numpy
#数据加载函数，kind值标明了读取文件的类型
ans=[]
def load(path, kind='train'):
    labels_path = os.path.join(path,'%s-labels.idx1-ubyte'% kind)
    images_path = os.path.join(path,'%s-images.idx3-ubyte'% kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II',lbpath.read(8))
        labels = numpy.fromfile(lbpath,dtype=numpy.uint8)
    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII',imgpath.read(16))
        images = numpy.fromfile(imgpath,dtype=numpy.uint8).reshape(len(labels), 784)
    #由于源数据有些数据过大，会导致激活函数计算溢出，所以对数据集集体缩小，
    #由于图片数据每一位的值均为0-255之间，但统一除以255后发现当神经元个数达到一定数目或层数增加时还是会计算溢出，于是决定统一除以2550
    return (images/2550), labels

def vote(lt):
	index1 = 0
	max = 0
	for i in range(len(lt)):
		flag = 0
		for j in range(i+1,len(lt)):
			if lt[j] == lt[i]:
				flag += 1
		if flag > max:
			max = flag
			index1 = i
	return index1
def Ensemble():
	f1=open('classificationforlinearSVM.txt')
	f2=open('classificationfornonlinearSVM.txt')
	f3=open('classificationforBayes.txt')
	f4=open('classificationforKNN.txt')
	f5=open('classificationforNN.txt')
	classificationforlinearSVM=f1.read().split()
	classificationfornonlinearSVM=f2.read().split()
	classificationforBayes=f3.read().split()
	classificationforKNN=f4.read().split()
	classificationforNN=f5.read().split()
	for i in range(len(classificationforBayes)):
		# print(len(classificationforBayes))
		ls=[]
		ls.append(classificationforlinearSVM[i])
		ls.append(classificationfornonlinearSVM[i])
		ls.append(classificationforBayes[i])
		ls.append(classificationforKNN[i])
		ls.append(classificationforNN[i])
		ans.append(ls[vote(ls)])
def accuracy():
	test_images_data,test_labels=load("C:\\Users\\Anonymous\\Documents\\机器学习\\作业五赵虎201600301325", kind='t10k')
	count=0
	for i in range(len(test_labels)):
		# print(len(test_labels))
		# print(type(test_labels[i]))
		if ans[i]==str(test_labels[i]):
			count+=1
			print(count)
	return count/10000
if __name__=='__main__':
	Ensemble()
	print('集成学习准确率：'+str(accuracy()))



