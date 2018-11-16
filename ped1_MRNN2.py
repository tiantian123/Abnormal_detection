# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 08:30:23 2015

@author: Administrator
"""
import numpy as np
from rnn3 import MetaRNN
import theano
import theano.tensor as T
from sklearn.base import BaseEstimator
import logging
import time
import os
import datetime
import cPickle as pickle
import matplotlib.pylab as plt
import random
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,auc
from matplotlib.ticker import MultipleLocator,FormatStrFormatter

def test_binary(multiple_out=False,n_epochs=250):
    """Test RNN with binary outputs."""
    n_hidden = 2
    n_in = 1104
    if multiple_out:
        n_out =1104
    else:
        n_out = 1
    n_steps = 39
    n_seq =170

    #load data into seq and targets
    floatX=theano.config.floatX
    seq = np.zeros([n_seq,n_steps,n_in])
    target = np.zeros([n_seq,n_steps,n_out],dtype='int32')
    #test_seq = np.zeros([36,n_steps,n_in])
    #test_target = np.zeros([36,n_steps,n_out],dtype='int32')
    test_seq = np.zeros([39,39,1104])
    test_target = np.zeros([39,39,1104])
    
    f1=file("ped1train24.csv","r")
    data = np.loadtxt(f1,delimiter=",")
    f1.close()
    f0=file("newdataALL36.csv","r")
    data2 = np.loadtxt(f0,delimiter=",")

    for i in range(0,170):
        seq[i,:,:]=data[i*39:(i+1)*39,:]
        target[i,:,:]=data[i*39+1:(i+1)*39+1,:]
    for i in range(0,39):
        test_seq[i,:,:]=data2[i,:]
    #test_target=data2[]
    """for i in range(0,180):
        test_seq[i:,:]=data2[i*39:(i+1)*39,:]
        test_target[i,:,:]=data2[i*39+1:(i+1)*39+1,:]"""
            #print "targetguess's shape:",targetguess.shape 

    #path1='target_turn_row.csv' 
    #fgs=file(path1,"a")
    #np.savetxt(fgs,test_target.reshape(-1,1),delimiter=",")
    #fgs.close()
    seqs=seq.astype(floatX)
    print seq.dtype
    test_seqs=test_seq.astype(floatX)
    print test_seqs.dtype
    target.astype(floatX)
    test_target.astype(floatX)
    
    model = MetaRNN(n_in=n_in, n_mul=n_hidden, n_out = n_out,learning_rate=0.1,
                    learning_rate_decay=0.999, n_epochs =n_epochs, L1_reg=0.005,
                    activation = 'tanh',output_type='real')
    model.fit(seqs,target,validation_frequency=200)
    path2='result_MRNN.csv'
    f2=file(path2,"a")
    flag=np.zeros([39,46])
    distance=np.zeros([39,39,1104])
    test_target[0] = model.predict(test_seq[0])
    print test_target[0]
    distance[0]=(test_target[0]-test_seq[0])**2
    for idy in range(0,46):
            if sum(distance[0,i*24:(i+1)*24])>10000:
                flag[0,idy]=1
    for idx in range(1,39):
        if sum(flag[idx])>0:
            test_target[idx]=model.predict(test_target[idx-1])
        else:
            test_target[idx] = model.predict(test_seq[idx])
        distance[idx]=(test_target[idx]-test_seq[idx])**2
        for idy in range(0,46):
            if sum(distance[idx,i*24:(i+1)*24])>10000:
                flag[idx,idy]=1
        if sum(flag[idx])>0:
            test_target[idx]=model.predict(test_target[idx-1])
    #np.savetxt(f2,distance,delimiter=",")
    print "flag:",flag
    f2.close()
    path3='falg_MRNN.csv'
    f3=file(path3,"a")
    np.savetxt(f3,flag,delimiter=",")
    f3.close()

    
def aggregation():
    """turn 1104 row to 46 row"""
    
    f0 = file("distance_MRNN.csv","r")
    data = np.loadtxt(f0,delimiter=",")
    f0.close()
    data = data.reshape(-1,24)
    
    feature=map(sum,data)
    
    path="distance_row_sum.csv"
    fgs=file(path,"a")
    np.savetxt(fgs,feature,delimiter=",")
    fgs.close()
    
    f2=file("distance_row_sum.csv","r")
    fe=np.loadtxt(f2,delimiter=",")
    fe = fe.reshape(-1,46)
    
    path2="feature46_MRNN_sum.csv"
    f3=file(path2,"a")
    np.savetxt(f3,fe,delimiter=",")
    f3.close()

def threshold1(thr,step,feature,target):
    """detect the anoraml"""
    numtrue=np.zeros(np.size(feature[:,0]))
    numdetect=np.zeros(np.size(feature[:,0]))
    flag = np.zeros([7020,46])
    n_step=0
    F=[]
    tpr=[]
    fpr=[]
    for dela in np.arange(0,thr,step):
        #print float(dela)
        #numtrue=np.zeros(np.size(feature[:,0]))
        numdetect=np.zeros(np.size(feature[:,0]))
        print numdetect
        n_step += 1
        for i in range(0,7020):
            for j in range(0,46):
                if feature[i,j]>=dela:
                    flag[i,j]=1.0
                else:
                    flag[i,j]=0.0
        #print flag
        for i in range(0,7020):
            if sum(target[i,:])>0:
                numtrue[i]=1.0
            if sum(flag[i,:])>0:
                numdetect[i]=1.0
        #print numdetect
        flag1=np.logical_and(numtrue==numdetect,numdetect==1)
        #print flag1
        num11=np.size(numtrue[flag1])
        print 'num11=',num11
        flag2=np.logical_and(numtrue==numdetect,numdetect==0)
        num00=np.size(numtrue[flag2])
        print 'num00=',num00
        flag3=np.logical_and(numtrue==1,numdetect==0)
        num10=np.size(numtrue[flag3])
        print 'num10=',num10
        flag4=np.logical_and(numtrue==0,numdetect==1)            
        num01=np.size(numtrue[flag4])
        print 'num01=',num01
        tp=float(num11)/float(num11+num10)
        fp=float(num01)/float(num01+num00)
        tpr.append(tp)
        fpr.append(fp)
        print 'tp=%f,fp=%f'% (tp,fp)
        precision = tp/(tp+fp)
        f=float(2/((1/precision)+(1/tp)))
        F.append(f)
        #print 'when n_step=',n_step
        #print 'threshold=',step
        
        #print 'F=%f '%(float(2/((1/precision)+(1/tp)))
    b=max(F)
    print 'F=',F
    print 'tpr:',tpr
    print 'fpr:',fpr
    #print 'the max f:',b
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=1,label='ROC fold (area =%0.2f)'% (roc_auc))
    #plt.title("异常检测")
    return b,F.index(b)*dela

def threshold2(thr,dela,feature,target):
    
    flag = np.zeros([7020,46])
    n_step=0
    F=[]
    tpr=[]
    fpr=[]
    for step in np.arange(0,thr,dela):
        numtrue=np.zeros(np.size(feature[:,0]))
        numdetect=np.zeros(np.size(feature[:,0]))
        print float(step)
        n_step += 1
        for i in range(0,7020):
            for j in range(0,46):
                if feature[i,j]>=step:
                    flag[i,j]=1.0
                else:
                    flag[i,j]=0.0
            if sum(target[i,:])==0:
                numtrue[i]=0
                if sum(flag[i,:])==0:
                    numdetect[i]=0
                else:
                    numdetect[i]=1
            else:
                numtrue[i]=1
                if sum(np.logical_and(flag[i],target[i])):
                    numdetect[i]=1
                else:
                    numdetect[i]=0
        
        print "flag:",flag
        print "numtrue:",numtrue
        print "numdetect:",numdetect
        flag1=np.logical_and(numtrue==numdetect,numdetect==1)
        print flag1
        num11=np.size(numtrue[flag1])
        print 'num11=',num11
        flag2=np.logical_and(numtrue==numdetect,numdetect==0)
        num00=np.size(numtrue[flag2])
        print 'num00=',num00
        flag3=np.logical_and(numtrue==1,numdetect==0)
        num10=np.size(numtrue[flag3])
        print 'num10=',num10
        flag4=np.logical_and(numtrue==0,numdetect==1)            
        num01=np.size(numtrue[flag4])
        print 'num01=',num01
        tp=float(num11)/float(num11+num10)
        fp=float(num01)/float(num01+num00)
        tpr.append(tp)
        fpr.append(fp)
        print 'tp=%f,fp=%f'% (tp,fp)
        precision = tp/(tp+fp)
        f=float(2/((1/precision)+(1/tp)))
        F.append(f)
        #print 'when n_step=',n_step
        #print 'threshold=',step
        #print 'F=',F
        #print 'F=%f '%(float(2/((1/precision)+(1/tp)))
    b=max(F)
    roc_auc = auc(fpr,tpr)
    plt.plot(fpr,tpr,lw=1,label='ROC fold (area =%0.2f)'% (roc_auc))
    print "F=",F
    print "tpr:",tpr
    print "fpr:",fpr
    #print 'the max f:',b
    return b,F.index(b)*dela
      
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    t0=time.time()
    #test_real(n_updates=20)
    test_binary(multiple_out=True,n_epochs=2)
    """aggregation()
    f0=file("feature46_MRNN_sum.csv",'r')
    distance=np.loadtxt(f0,delimiter=',')
    f0.close()
    #feature = distance[0:30,0:10]
    f1=file("locate36.csv",'r')
    data=np.loadtxt(f1,delimiter=',')
    f1.close()
    target = data[1:7021,:]
    #print 'target:',target
    F1,threshold1=threshold1(100000,1000,distance,target)
    print F1,threshold2
    F2,threshold2=threshold2(100000,10000,distance,target)
    print F2,threshold2
    #test_softmax(n_updates=20)"""
    print "Elapsed time:%f" % (time.time()-t0)
