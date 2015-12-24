#-*-coding:utf-8-*-
__author__ = 'Qiao'


from numpy import *
import operator


class knn:

    #初始化：
    def  __init__(self,Filename,Inx,K,Filetest):
        #training数据集：
        self.filetrain=Filename
        #待检测数据：
        self.inX=Inx
        self.k=K
        #test数据集：
        self.filetest=Filetest

    #对training数据进行文本分析并返回矩阵：
    def file2matrix(self,filetrain):
        try:
            fr = open(self.filetrain)
        except:
            filetrain=raw_input("错误.输入training数据集")
            fr=open(filetrain)
        numberOfLines = len(fr.readlines())
        dataSet = zeros((numberOfLines,3))
        labels = []
        index = 0
        for line in fr.readlines():
             line = line.strip()
             listFromLine = line.split('\t')
             dataSet[index,:] = listFromLine[0:3]
             labels.append(int(listFromLine[-1]))
             index += 1
        return dataSet,labels,numberOfLines

    #归一化数据：
    def autoNorm(self,dataSet):
        minVals = dataSet.min(0)
        maxVals = dataSet.max(0)
        ranges = maxVals - minVals
        normDataSet=zeros()
        m = dataSet.shape[0]
        normDataSet = dataSet - tile(minVals, (m,1))
        normDataSet = normDataSet/tile(ranges, (m,1))
        return normDataSet, ranges, minVals


    #分类：
    def classify0(self, normdataSet, labels):
        #计算距离并距离排序：
        dataSetSize = normdataSet.shape[0]
        norminx=self.autoNorm(self,self.inX)
        diffMat = tile(norminx, (dataSetSize,1)) - normdataSet
        sqDiffMat = diffMat**2
        sqDistances = sqDiffMat.sum(axis=1)
        distances = sqDistances**0.5
        sortedDistIndicies = distances.argsort()
        #选择与待分类数据最近k个的数据点并统计其类型：
        classCount={}
        try:
            k=int(self.k)
        except:
            k=raw_input("错误。输入k值")
            k=int(k)
        for i in range(self.k):
             voteIlabel = labels[sortedDistIndicies[i]]
             classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
        #确定分类：
        sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
        return sortedClassCount[0][0]


    #算法检测：
    def datingClassTest(self):
        hoRatio = 0.50
        datingDataMat,datingLabels =self.file2matrix(self.filetest)
        normMat, ranges, minVals = self.autoNorm(datingDataMat)
        m = normMat.shape[0]
        numTestVecs = int(m*hoRatio)
        errorCount = 0.0
        for i in range(numTestVecs):
             classifierResult = self.classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
             print "分类结果为: %d, 正确结果为: %d" % (classifierResult, datingLabels[i])
             if (classifierResult != datingLabels[i]): errorCount += 1.0
        print "错误率为: %f" % (errorCount/float(numTestVecs))
        print errorCount


print "k均值聚类学习"
Filename=raw_input("输入training数据集")
Filetest=raw_input("输入test数据集")
#可以用graphlib将一个数据集进行确定比例的随机分割
K=raw_input("输入k值")
Inx=raw_input("输入待分类数据")
KNN=knn(Filename,Inx,K,Filetest)
KNN.start()
