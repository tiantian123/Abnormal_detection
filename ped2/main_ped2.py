# -*- coding: utf-8 -*-
"""
Created on Tue May 17 09:50:51 2016

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
    f1=file("ped2train_raw.csv","r")
    data = np.loadtxt(f1,delimiter=',')
    f2=file("ped2test_raw.csv","r")
    test_data = np.loadtxt(f2,delimiter=',')
    
    floatX = theano.config.floatX
    
    seq = np.zeros([4200,30,72])
    target = np.zeros([4200,30,24])
    test_seq = np.zeros([3000,30,72])
    test_target = np.zeros([90000,24])
    
   
    for idx in range(0,5):
        for idy in range(0,10):
            index=idx*10+idy
            for i in range(0,84):
                seq[index*84+i,:,0:24]=data[i*30:(i+1)*30,index*24:(index+1)*24]
                target[index*84+i,:,0:24]=data[i*30+1:(i+1)*30+1,index*24:(index+1)*24]
                if idx-1>=0 and idy-1>=0:
                    seq[index*84+i,:,24:26]=data[i*30:(i+1)*30,(index-11)*24+6:(index-11)*24+8]
                    seq[index*84+i,:,26:28]=data[i*30:(i+1)*30,(index-11)*24+14:(index-11)*24+16]
                    seq[index*84+i,:,28:30]=data[i*30:(i+1)*30,(index-11)*24+22:(index-11)*24+24]
                
                if idx-1>=0:				
                    seq[index*84+i,:,30:32]=data[i*30:(i+1)*30,(index-10)*24+5:(index-10)*24+7]
                    seq[index*84+i,:,32:34]=data[i*30:(i+1)*30,(index-10)*24+13:(index-10)*24+15]
                    seq[index*84+i,:,34:36]=data[i*30:(i+1)*30,(index-10)*24+21:(index-10)*24+23]
                
                if idx-1>=0 and idy+1<10:    				
                    seq[index*84+i,:,36:38]=data[i*30:(i+1)*30,(index-9)*24+4:(index-9)*24+6]
                    seq[index*84+i,:,38:40]=data[i*30:(i+1)*30,(index-9)*24+12:(index-9)*24+14]
                    seq[index*84+i,:,40:42]=data[i*30:(i+1)*30,(index-9)*24+20:(index-9)*24+22]
                    
                if idy-1>=0:
                    seq[index*84+i,:,42]=data[i*30:(i+1)*30,(index-1)*24]
                    seq[index*84+i,:,43:45]=data[i*30:(i+1)*30,(index-1)*24+7:(index-1)*24+9]
                    seq[index*84+i,:,45:47]=data[i*30:(i+1)*30,(index-1)*24+15:(index-1)*24+17]
                    seq[index*84+i,:,48]=data[i*30:(i+1)*30,(index-1)*24+23]
                    
                if idy+1<10:
                    seq[index*84+i,:,48:50]=data[i*30:(i+1)*30,(index+1)*24+3:(index+1)*24+5]
                    seq[index*84+i,:,50:52]=data[i*30:(i+1)*30,(index+1)*24+11:(index+1)*24+13]
                    seq[index*84+i,:,52:54]=data[i*30:(i+1)*30,(index+1)*24+19:(index+1)*24+21]
                     
                if idx+1<5 and idy-1>=0:
                    seq[index*84+i,:,54:56]=data[i*30:(i+1)*30,(index+9)*24:(index+9)*24+2]
                    seq[index*84+i,:,56:58]=data[i*30:(i+1)*30,(index+9)*24+8:(index+9)*24+10]
                    seq[index*84+i,:,58:60]=data[i*30:(i+1)*30,(index+9)*24+16:(index+9)*24+18]
                
                if idx+1<5:    				
                    seq[index*84+i,:,60:62]=data[i*30:(i+1)*30,(index+10)*24+1:(index+10)*24+3]
                    seq[index*84+i,:,62:64]=data[i*30:(i+1)*30,(index+10)*24+9:(index+10)*24+11]
                    seq[index*84+i,:,64:66]=data[i*30:(i+1)*30,(index+10)*24+17:(index+10)*24+19]
                
                if idx+1<5 and idy+1<10:                    
                    seq[index*84+i,:,66:68]=data[i*30:(i+1)*30,(index+11)*24+2:(index+11)*24+4]
                    seq[index*84+i,:,68:70]=data[i*30:(i+1)*30,(index+11)*24+10:(index+11)*24+12]
                    seq[index*84+i,:,70:72]=data[i*30:(i+1)*30,(index+11)*24+18:(index+11)*24+20]
            
            test_target[index*1800:(index+1)*1800,0:24]=test_data[:1800,index*24:(index+1)*24]
            for i in range(0,60):
                test_seq[index*60+i,:,0:24]=test_data[i*30:(i+1)*30,index*24:(index+1)*24]
                
                if idx-1>=0 and idy-1>=0:
                    seq[index*60+i,:,24:26]=data[i*30:(i+1)*30,(index-11)*24+6:(index-11)*24+8]
                    seq[index*60+i,:,26:28]=data[i*30:(i+1)*30,(index-11)*24+14:(index-11)*24+16]
                    seq[index*60+i,:,28:30]=data[i*30:(i+1)*30,(index-11)*24+22:(index-11)*24+24]
                
                if idx-1>=0:				
                    seq[index*60+i,:,30:32]=data[i*30:(i+1)*30,(index-10)*24+5:(index-10)*24+7]
                    seq[index*60+i,:,32:34]=data[i*30:(i+1)*30,(index-10)*24+13:(index-10)*24+15]
                    seq[index*60+i,:,34:36]=data[i*30:(i+1)*30,(index-10)*24+21:(index-10)*24+23]
                
                if idx-1>=0 and idy+1<10:    				
                    seq[index*60+i,:,36:38]=data[i*30:(i+1)*30,(index-9)*24+4:(index-9)*24+6]
                    seq[index*60+i,:,38:40]=data[i*30:(i+1)*30,(index-9)*24+12:(index-9)*24+14]
                    seq[index*60+i,:,40:42]=data[i*30:(i+1)*30,(index-9)*24+20:(index-9)*24+22]
                    
                if idy-1>=0:
                    seq[index*60+i,:,42]=data[i*30:(i+1)*30,(index-1)*24]
                    seq[index*60+i,:,43:45]=data[i*30:(i+1)*30,(index-1)*24+7:(index-1)*24+9]
                    seq[index*60+i,:,45:47]=data[i*30:(i+1)*30,(index-1)*24+15:(index-1)*24+17]
                    seq[index*60+i,:,48]=data[i*30:(i+1)*30,(index-1)*24+23]
                    
                if idy+1<10:
                    seq[index*60+i,:,48:50]=data[i*30:(i+1)*30,(index+1)*24+3:(index+1)*24+5]
                    seq[index*60+i,:,50:52]=data[i*30:(i+1)*30,(index+1)*24+11:(index+1)*24+13]
                    seq[index*60+i,:,52:54]=data[i*30:(i+1)*30,(index+1)*24+19:(index+1)*24+21]
                     
                if idx+1<5 and idy-1>=0:
                    seq[index*60+i,:,54:56]=data[i*30:(i+1)*30,(index+9)*24:(index+9)*24+2]
                    seq[index*60+i,:,56:58]=data[i*30:(i+1)*30,(index+9)*24+8:(index+9)*24+10]
                    seq[index*60+i,:,58:60]=data[i*30:(i+1)*30,(index+9)*24+16:(index+9)*24+18]
                
                if idx+1<5:    				
                    seq[index*60+i,:,60:62]=data[i*30:(i+1)*30,(index+10)*24+1:(index+10)*24+3]
                    seq[index*60+i,:,62:64]=data[i*30:(i+1)*30,(index+10)*24+9:(index+10)*24+11]
                    seq[index*60+i,:,64:66]=data[i*30:(i+1)*30,(index+10)*24+17:(index+10)*24+19]
                
                if idx+1<5 and idy+1<10:                    
                    seq[index*60+i,:,66:68]=data[i*30:(i+1)*30,(index+11)*24+2:(index+11)*24+4]
                    seq[index*60+i,:,68:70]=data[i*30:(i+1)*30,(index+11)*24+10:(index+11)*24+12]
                    seq[index*60+i,:,70:72]=data[i*30:(i+1)*30,(index+11)*24+18:(index+11)*24+20]
    
    seqs = seq.astype(floatX)
    targets = target.astype(floatX)
    test_seqs = test_seq.astype(floatX)
    test_targets = test_target.astype(floatX)
    
    [n_steps,n_seq,n_in]=seq.shape
    
    n_hidden=2*n_in
    n_out=24
    n_epochs=1500
    
    '''seqs = [i for i in seqs]
    targets = [i for i in targets]
    gradient_dataset = SequenceDataset([seqs, targets], batch_size=None,
										   number_batches=50)
    cg_dataset = SequenceDataset([seqs, targets], batch_size=None,
                             number_batches=20)
		
    model = MetaRNN(n_in=n_in, n_mul=n_hidden, n_out=n_out,
                activation='tanh')
                
    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
						   s=model.rnn.y_pred,
						   costs=[model.rnn.loss(model.y)], h=model.rnn.h)
    
    opt.train(gradient_dataset, cg_dataset, num_updates=300)'''
    model = MetaRNN(n_in=n_in, n_mul=n_hidden, n_out=n_out, learning_rate=0.001,
                        learning_rate_decay=0.999, n_epochs=n_epochs, L1_reg=0.005,
                        activation = 'tanh', output_type='real',index=index)
    model.fit(seqs,targets,validation_frequency=1000)
    print "traning over"

    path='/root/chentian/share/201605/ped2'
#    os.makedirs(path)
    path0 = path+'/predict.csv'
    f3 = file(path0,"a")
    for k in range(0,3000):
        guess = model.predict(test_seqs[k])
        np.savetxt(f3,guess,delimiter=',')
    f3.close()
    
    error1=np.zeros([1800,1200])
    pre_f=file(path0,"r")
    pre=np.loadtxt(pre_f,delimiter=',')
    pre_f.close()
    path3= path+"/error.csv"
    f8 = file(path3,"a")
    error=(pre-test_targets)**2
    for i in range(0,50):
        error1[:,i*24:(i+1)*24]=error[i*1800:(i+1)*1800,:]
    np.savetxt(f8,error1,delimiter=',')
    f8.close()

    print "Elapsed time: %f" % (time.time()-t0)
        