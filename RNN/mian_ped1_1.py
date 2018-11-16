# -*- coding: utf-8 -*-
"""
Created on Mon May 23 15:13:22 2016

@author: TianChen
"""
import numpy as np
import theano
from rnn import MetaRNN
from hf import SequenceDataset, hf_optimizer
import logging
import time
import os

if __name__=='__main__': 
    logging.basicConfig(level=logging.INFO)
    t0=time.time()
    print  "start time:",time.ctime
    f1=file("ped1train24.csv","r")
    data = np.loadtxt(f1,delimiter=',')
    f2=file("ped1test24.csv","r")
    test_data = np.loadtxt(f2,delimiter=',')
    
    floatX = theano.config.floatX

    n_in=72
    n_out=24
    n_seq=34
    n_steps=200
    n_grid=46
    
    n_seqs=n_grid*n_seq
    
    seq = np.zeros([n_seqs,n_steps,n_in])
    target = np.zeros([n_seqs,n_steps,n_out])
    test_seq = np.zeros([n_seqs,n_steps,n_in])
    test_target = np.zeros([312800,n_out])

    for idx in range(0,7):
        for idy in range(0,8):
            index = idx*8 + idy
            if index in [0,1,2,3,8,9,10,16,17,24]:
                break
            elif index in range(4,8):
                index-=4
            elif index in range(11,16):
                index-=7
            elif index in range(18,24):
                index-=9
            else:
                index-=10
            col = index
            row = index    
            if index in range(4,9):
                col+=3
                row+=1
            elif index in range(9,15):
                col+=2
                row+=2
            elif index in range(15,22):
                col+=1
                row+=3          
            for i in range(0,n_seq):
                seq[index*n_seq+i,:,0:24]=data[i*n_steps:(i+1)*n_steps,index*24:(index+1)*24]
                test_seq[index*n_seq+i,:,0:24]=test_data[i*n_steps:(i+1)*n_steps,index*24:(index+1)*24]
                target[index*n_seq+i,:,0:24]=data[i*n_steps+1:(i+1)*n_steps+1,index*24:(index+1)*24]
                if idx-1>=0 and idy-1>=0 and index not in [4,5,9,10,15,16,23]:
                    seq[index*n_seq+i,:,24:26]=data[i*n_steps:(i+1)*n_steps,(col-9)*24+6:(col-9)*24+8]
                    seq[index*n_seq+i,:,26:28]=data[i*n_steps:(i+1)*n_steps,(col-9)*24+14:(col-9)*24+16]
                    seq[index*n_seq+i,:,28:30]=data[i*n_steps:(i+1)*n_steps,(col-9)*24+22:(col-9)*24+24]
                    test_seq[index*n_seq+i,:,24:26]=test_data[i*n_steps:(i+1)*n_steps,(col-9)*24+6:(col-9)*24+8]
                    test_seq[index*n_seq+i,:,26:28]=test_data[i*n_steps:(i+1)*n_steps,(col-9)*24+14:(col-9)*24+16]
                    test_seq[index*n_seq+i,:,28:30]=test_data[i*n_steps:(i+1)*n_steps,(col-9)*24+22:(col-9)*24+24]
                    
                if idx-1>=0 and index not in [4,9,15,22]:				
                    seq[index*n_seq+i,:,30:32]=data[i*n_steps:(i+1)*n_steps,(col-8)*24+5:(col-8)*24+7]
                    seq[index*n_seq+i,:,32:34]=data[i*n_steps:(i+1)*n_steps,(col-8)*24+13:(col-8)*24+15]
                    seq[index*n_seq+i,:,34:36]=data[i*n_steps:(i+1)*n_steps,(col-8)*24+21:(col-8)*24+23]
                    test_seq[index*n_seq+i,:,30:32]=test_data[i*n_steps:(i+1)*n_steps,(col-8)*24+5:(col-8)*24+7]
                    test_seq[index*n_seq+i,:,32:34]=test_data[i*n_steps:(i+1)*n_steps,(col-8)*24+13:(col-8)*24+15]
                    test_seq[index*n_seq+i,:,34:36]=test_data[i*n_steps:(i+1)*n_steps,(col-8)*24+21:(col-8)*24+23]
                if idx-1>=0 and idy+1<8:    				
                    seq[index*n_seq+i,:,36:38]=data[i*n_steps:(i+1)*n_steps,(col-7)*24+4:(col-7)*24+6]
                    seq[index*n_seq+i,:,38:40]=data[i*n_steps:(i+1)*n_steps,(col-7)*24+12:(col-7)*24+14]
                    seq[index*n_seq+i,:,40:42]=data[i*n_steps:(i+1)*n_steps,(col-7)*24+20:(col-7)*24+22]
                    test_seq[index*n_seq+i,:,36:38]=test_data[i*n_steps:(i+1)*n_steps,(col-7)*24+4:(col-7)*24+6]
                    test_seq[index*n_seq+i,:,38:40]=test_data[i*n_steps:(i+1)*n_steps,(col-7)*24+12:(col-7)*24+14]
                    test_seq[index*n_seq+i,:,40:42]=test_data[i*n_steps:(i+1)*n_steps,(col-7)*24+20:(col-7)*24+22]
                    
                if idy-1>=0 and index not in [0,4,9,15]:
                    seq[index*n_seq+i,:,42]=data[i*n_steps:(i+1)*n_steps,(index-1)*24]
                    seq[index*n_seq+i,:,43:45]=data[i*n_steps:(i+1)*n_steps,(index-1)*24+7:(index-1)*24+9]
                    seq[index*n_seq+i,:,45:47]=data[i*n_steps:(i+1)*n_steps,(index-1)*24+15:(index-1)*24+17]
                    seq[index*n_seq+i,:,48]=data[i*n_steps:(i+1)*n_steps,(index-1)*24+23]
                    test_seq[index*n_seq+i,:,42]=test_data[i*n_steps:(i+1)*n_steps,(index-1)*24]
                    test_seq[index*n_seq+i,:,43:45]=test_data[i*n_steps:(i+1)*n_steps,(index-1)*24+7:(index-1)*24+9]
                    test_seq[index*n_seq+i,:,45:47]=test_data[i*n_steps:(i+1)*n_steps,(index-1)*24+15:(index-1)*24+17]
                    test_seq[index*n_seq+i,:,48]=test_data[i*n_steps:(i+1)*n_steps,(index-1)*24+23]
                    
                if idy+1<8:
                    seq[index*n_seq+i,:,48:50]=data[i*n_steps:(i+1)*n_steps,(index+1)*24+3:(index+1)*24+5]
                    seq[index*n_seq+i,:,50:52]=data[i*n_steps:(i+1)*n_steps,(index+1)*24+11:(index+1)*24+13]
                    seq[index*n_seq+i,:,52:54]=data[i*n_steps:(i+1)*n_steps,(index+1)*24+19:(index+1)*24+21]
                    test_seq[index*n_seq+i,:,48:50]=test_data[i*n_steps:(i+1)*n_steps,(index+1)*24+3:(index+1)*24+5]
                    test_seq[index*n_seq+i,:,50:52]=test_data[i*n_steps:(i+1)*n_steps,(index+1)*24+11:(index+1)*24+13]
                    test_seq[index*n_seq+i,:,52:54]=test_data[i*n_steps:(i+1)*n_steps,(index+1)*24+19:(index+1)*24+21]
                if idx+1<7 and idy-1>=0:
                    seq[index*n_seq+i,:,54:56]=data[i*n_steps:(i+1)*n_steps,(row+4)*24:(row+4)*24+2]
                    seq[index*n_seq+i,:,56:58]=data[i*n_steps:(i+1)*n_steps,(row+4)*24+8:(row+4)*24+10]
                    seq[index*n_seq+i,:,58:60]=data[i*n_steps:(i+1)*n_steps,(row+4)*24+16:(row+4)*24+18]
                    test_seq[index*n_seq+i,:,54:56]=test_data[i*n_steps:(i+1)*n_steps,(row+4)*24:(row+4)*24+2]
                    test_seq[index*n_seq+i,:,56:58]=test_data[i*n_steps:(i+1)*n_steps,(row+4)*24+8:(row+4)*24+10]
                    test_seq[index*n_seq+i,:,58:60]=test_data[i*n_steps:(i+1)*n_steps,(row+4)*24+16:(row+4)*24+18]
                if idx+1<7:    				
                    seq[index*n_seq+i,:,60:62]=data[i*n_steps:(i+1)*n_steps,(row+5)*24+1:(row+5)*24+3]
                    seq[index*n_seq+i,:,62:64]=data[i*n_steps:(i+1)*n_steps,(row+5)*24+9:(row+5)*24+11]
                    seq[index*n_seq+i,:,64:66]=data[i*n_steps:(i+1)*n_steps,(row+5)*24+17:(row+5)*24+19]
                    test_seq[index*n_seq+i,:,60:62]=test_data[i*n_steps:(i+1)*n_steps,(row+5)*24+1:(row+5)*24+3]
                    test_seq[index*n_seq+i,:,62:64]=test_data[i*n_steps:(i+1)*n_steps,(row+5)*24+9:(row+5)*24+11]
                    test_seq[index*n_seq+i,:,64:66]=test_data[i*n_steps:(i+1)*n_steps,(row+5)*24+17:(row+5)*24+19]
                if idx+1<7 and idy+1<8:                    
                    seq[index*n_seq+i,:,66:68]=data[i*n_steps:(i+1)*n_steps,(row+6)*24+2:(row+6)*24+4]
                    seq[index*n_seq+i,:,68:70]=data[i*n_steps:(i+1)*n_steps,(row+6)*24+10:(row+6)*24+12]
                    seq[index*n_seq+i,:,70:72]=data[i*n_steps:(i+1)*n_steps,(row+6)*24+18:(row+6)*24+20]
                    test_seq[index*n_seq+i,:,66:68]=test_data[i*n_steps:(i+1)*n_steps,(row+6)*24+2:(row+6)*24+4]
                    test_seq[index*n_seq+i,:,68:70]=test_data[i*n_steps:(i+1)*n_steps,(row+6)*24+10:(row+6)*24+12]
                    test_seq[index*n_seq+i,:,70:72]=test_data[i*n_steps:(i+1)*n_steps,(row+6)*24+18:(row+6)*24+20]
                    
            test_target[index*6800:(index+1)*6800,0:24]=test_data[:6800,index*24:(index+1)*24]
            
    seqs = seq.astype(floatX)
    targets = target.astype(floatX)
    test_seqs = test_seq.astype(floatX)
    test_targets = test_target.astype(floatX)
    
    [n_steps,n_seq,n_in]=seq.shape
    
    n_hidden=2*n_in
    n_out=24
        
    seqs = [i for i in seqs]
    targets = [i for i in target]
    gradient_dataset = SequenceDataset([seqs, targets], batch_size=None,
										   number_batches=50)
    cg_dataset = SequenceDataset([seqs, targets], batch_size=None,
                             number_batches=20)
		
    model = MetaRNN(n_in=n_in, n_mul=n_hidden, n_out=n_out,
                activation='tanh')
                
    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
						   s=model.rnn.y_pred,
						   costs=[model.rnn.loss(model.y)], h=model.rnn.h)
    
    opt.train(gradient_dataset, cg_dataset, num_updates=300)

    print "traning over"

    path='/root/chentian/share/201605/ped1'
#    os.makedirs(path)
    path0 = path+'/predict.csv'
    f3 = file(path0,"a")
    for k in range(0,n_seqs):
        guess = model.predict(test_seqs[k])
        np.savetxt(f3,guess,delimiter=',')
    f3.close()
    
    error1=np.zeros([6800,1104])
    pre_f=file(path0,"r")
    pre=np.loadtxt(pre_f,delimiter=',')
    pre_f.close()
    path3= path+"/error.csv"
    f8 = file(path3,"a")
    error=(pre-test_targets)**2
    for i in range(0,46):
        error1[:,i*24:(i+1)*24]=error[i*6800:(i+1)*6800,:]
    np.savetxt(f8,error1,delimiter=',')
    f8.close()

    print "Elapsed time: %f" % (time.time()-t0)       