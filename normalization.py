# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 08:56:09 2015

@author: Administrator
"""
import numpy as np

def LoadData(fileName1,fileName2):
    seq1=np.zeros([6631,1104])
    seq2=np.zeros([7021,1104])
    f0=file(fileName1)
    data1=np.loadtxt(f0,delimiter=',')
    seq1=data1[0:6630,:]
    f0.close()
    f1=file(fileName2)
    data2=np.loadtxt(f1,delimiter=',')
    seq2=data2[0:7020,:]
    f1.close()
    
    seq=np.append(seq1,seq2,axis=0)
    print seq.shape
    seqs=np.zeros([46,13652,24])
    for i in range(0,46):
        seqs[i,:,:]=seq[:,i*24:(i+1)*24]
        
    seqmin=[]
    seqmax=[]
    seqmean=[]
    for i in xrange(0,46):
        seqmin.append(np.min(seqs[i]))
        seqmax.append(np.max(seqs[i]))
        seqmean.append(np.mean(seqs[i]))
        seqs[i]=(seqs[i]-seqmin[i])/(seqmax[i]-seqmin[i])
        
    for i in range(0,46):
        seq[:,i*24:(i+1)*24]=seqs[i]
    
    path0='normalization_data.csv'
    f2 = file(path0,'a')
    np.savetxt(f2,seq,delimiter=',')
    f2.close()

def splitData(fileName)    :
    f0=file(fileName)
    data = np.loadtxt(f0,delimiter=',')
    seq1 = data[:6630,:]
    seq2 = data[6630:,:]
    path0 = 'trainData.csv'
    f1=file(path0,'a')
    np.savetxt(f1,seq1,delimiter=',')
    f1.close()
    path1 = 'testData.csv'
    f2=file(path1,'a')
    np.savetxt(f2,seq2,delimiter=',')
    f2.close()
    
    
if __name__=='__main__':
    LoadData("ped1train24.csv","newdataALL36.csv")
    splitData("normalization_data.csv")
    
    
    