# -*- coding: utf-8 -*-
"""
Created on Mon May 16 15:32:56 2016

@author: TianChen
"""
import numpy as np
import theano.tensor as T
import theano
from rnn import MetaRNN
from hf import SequenceDataset,hf_optimizer
import logging
import time
import os

if __name__=='__main__':
    logging.basicConfig(level=logging.INFO)
    t0=time.time()
    f1=file("ped1train24.csv","r")
    data = np.loadtxt(f1,delimiter=',')
    f2=file("ped1test24.csv","r")
    test_data = np.loadtxt(f2,delimiter=',')
	sub=50
	sample=6800/sub
	total_sample=sample*46
    
    floatX = theano.config.floatX
    
    seq = np.zeros([total_sample,sub,72])
    target = np.zeros([total_sample,sub,24])
    test_seq = np.zeros([total_sample,sub,72])
    test_target = np.zeros([312800,24])
    
    for index in range(0,46):
        if index==0:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+4)*24:(index+4)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+4)*24+8:(index+4)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+4)*24+16:(index+4)*24+18]
                
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+5)*24+1:(index+5)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+5)*24+9:(index+5)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+5)*24+17:(index+5)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+6)*24+2:(index+6)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+6)*24+10:(index+6)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+6)*24+18:(index+6)*24+20]
                
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
                
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+4)*24:(index+4)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+4)*24+8:(index+4)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+4)*24+16:(index+4)*24+18]
                
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+5)*24+1:(index+5)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+5)*24+9:(index+5)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+5)*24+17:(index+5)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+6)*24+2:(index+6)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+6)*24+10:(index+6)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+6)*24+18:(index+6)*24+20]
				
        elif index in [1,2]:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
                
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
                
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+4)*24:(index+4)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+4)*24+8:(index+4)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+4)*24+16:(index+4)*24+18]
                
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+5)*24+1:(index+5)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+5)*24+9:(index+5)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+5)*24+17:(index+5)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+6)*24+2:(index+6)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+6)*24+10:(index+6)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+6)*24+18:(index+6)*24+20]
                
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
                
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+4)*24:(index+4)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+4)*24+8:(index+4)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+4)*24+16:(index+4)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+5)*24+1:(index+5)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+5)*24+9:(index+5)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+5)*24+17:(index+5)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+6)*24+2:(index+6)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+6)*24+10:(index+6)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+6)*24+18:(index+6)*24+20]
        elif index==3:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+4)*24:(index+4)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+4)*24+8:(index+4)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+4)*24+16:(index+4)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+5)*24+1:(index+5)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+5)*24+9:(index+5)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+5)*24+17:(index+5)*24+19]
                
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
						
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+4)*24:(index+4)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+4)*24+8:(index+4)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+4)*24+16:(index+4)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+5)*24+1:(index+5)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+5)*24+9:(index+5)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+5)*24+17:(index+5)*24+19]
        elif index==4:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-4)*24+4:(index-4)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-4)*24+12:(index-4)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-4)*24+20:(index-4)*24+22]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+5)*24:(index+5)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+5)*24+8:(index+5)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+5)*24+16:(index+5)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+6)*24+1:(index+6)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+6)*24+9:(index+6)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+6)*24+17:(index+6)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+7)*24+2:(index+7)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+7)*24+10:(index+7)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+7)*24+18:(index+7)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-4)*24+4:(index-4)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-4)*24+12:(index-4)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-4)*24+20:(index-4)*24+22]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+5)*24:(index+5)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+5)*24+8:(index+5)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+5)*24+16:(index+5)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+6)*24+1:(index+6)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+6)*24+9:(index+6)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+6)*24+17:(index+6)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+7)*24+2:(index+7)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+7)*24+10:(index+7)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+7)*24+18:(index+7)*24+20]
        
        elif index==5:
            for i in range(0,sample):          
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-5)*24+5:(index-5)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-5)*24+13:(index-5)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-5)*24+21:(index-5)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-4)*24+4:(index-4)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-4)*24+12:(index-4)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-4)*24+20:(index-4)*24+22]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+5)*24:(index+5)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+5)*24+8:(index+5)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+5)*24+16:(index+5)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+6)*24+1:(index+6)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+6)*24+9:(index+6)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+6)*24+17:(index+6)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+7)*24+2:(index+7)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+7)*24+10:(index+7)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+7)*24+18:(index+7)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-5)*24+5:(index-5)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-5)*24+13:(index-5)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-5)*24+21:(index-5)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-4)*24+4:(index-4)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-4)*24+12:(index-4)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-4)*24+20:(index-4)*24+22]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+5)*24:(index+5)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+5)*24+8:(index+5)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+5)*24+16:(index+5)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+6)*24+1:(index+6)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+6)*24+9:(index+6)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+6)*24+17:(index+6)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+7)*24+2:(index+7)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+7)*24+10:(index+7)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+7)*24+18:(index+7)*24+20]
        elif index in [6,7]:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-6)*24+6:(index-6)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-6)*24+14:(index-6)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-6)*24+22:(index-6)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-5)*24+5:(index-5)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-5)*24+13:(index-5)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-5)*24+21:(index-5)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-4)*24+4:(index-4)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-4)*24+12:(index-4)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-4)*24+20:(index-4)*24+22]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+5)*24:(index+5)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+5)*24+8:(index+5)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+5)*24+16:(index+5)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+6)*24+1:(index+6)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+6)*24+9:(index+6)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+6)*24+17:(index+6)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+7)*24+2:(index+7)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+7)*24+10:(index+7)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+7)*24+18:(index+7)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-6)*24+6:(index-6)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-6)*24+14:(index-6)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-6)*24+22:(index-6)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-5)*24+5:(index-5)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-5)*24+13:(index-5)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-5)*24+21:(index-5)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-4)*24+4:(index-4)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-4)*24+12:(index-4)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-4)*24+20:(index-4)*24+22]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+5)*24:(index+5)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+5)*24+8:(index+5)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+5)*24+16:(index+5)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+6)*24+1:(index+6)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+6)*24+9:(index+6)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+6)*24+17:(index+6)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+7)*24+2:(index+7)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+7)*24+10:(index+7)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+7)*24+18:(index+7)*24+20]
        elif index==8:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-6)*24+6:(index-6)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-6)*24+14:(index-6)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-6)*24+22:(index-6)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-5)*24+5:(index-5)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-5)*24+13:(index-5)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-5)*24+21:(index-5)*24+23]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
								
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+5)*24:(index+5)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+5)*24+8:(index+5)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+5)*24+16:(index+5)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+6)*24+1:(index+6)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+6)*24+9:(index+6)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+6)*24+17:(index+6)*24+19]
                				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-6)*24+6:(index-6)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-6)*24+14:(index-6)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-6)*24+22:(index-6)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-5)*24+5:(index-5)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-5)*24+13:(index-5)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-5)*24+21:(index-5)*24+23]
								
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+5)*24:(index+5)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+5)*24+8:(index+5)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+5)*24+16:(index+5)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+6)*24+1:(index+6)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+6)*24+9:(index+6)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+6)*24+17:(index+6)*24+19]
        elif index==9:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
								
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-5)*24+4:(index-5)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-5)*24+12:(index-5)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-5)*24+20:(index-5)*24+22]
								
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+6)*24:(index+6)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+6)*24+8:(index+6)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+6)*24+16:(index+6)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+7)*24+1:(index+7)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+7)*24+9:(index+7)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+7)*24+17:(index+7)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+8)*24+2:(index+8)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+8)*24+10:(index+8)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+8)*24+18:(index+8)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
							
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-5)*24+4:(index-5)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-5)*24+12:(index-5)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-5)*24+20:(index-5)*24+22]
							
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+6)*24:(index+6)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+6)*24+8:(index+6)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+6)*24+16:(index+6)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+7)*24+1:(index+7)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+7)*24+9:(index+7)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+7)*24+17:(index+7)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+8)*24+2:(index+8)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+8)*24+10:(index+8)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+8)*24+18:(index+8)*24+20]
        elif index==10:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
								
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-6)*24+5:(index-6)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-6)*24+13:(index-6)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-6)*24+21:(index-6)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-5)*24+4:(index-5)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-5)*24+12:(index-5)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-5)*24+20:(index-5)*24+22]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+6)*24:(index+6)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+6)*24+8:(index+6)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+6)*24+16:(index+6)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+7)*24+1:(index+7)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+7)*24+9:(index+7)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+7)*24+17:(index+7)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+8)*24+2:(index+8)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+8)*24+10:(index+8)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+8)*24+18:(index+8)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-6)*24+5:(index-6)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-6)*24+13:(index-6)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-6)*24+21:(index-6)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-5)*24+4:(index-5)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-5)*24+12:(index-5)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-5)*24+20:(index-5)*24+22]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+6)*24:(index+6)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+6)*24+8:(index+6)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+6)*24+16:(index+6)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+7)*24+1:(index+7)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+7)*24+9:(index+7)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+7)*24+17:(index+7)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+8)*24+2:(index+8)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+8)*24+10:(index+8)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+8)*24+18:(index+8)*24+20]
        elif index in [11,12,13]:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-7)*24+6:(index-7)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-7)*24+14:(index-7)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-7)*24+22:(index-7)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-6)*24+5:(index-6)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-6)*24+13:(index-6)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-6)*24+21:(index-6)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-5)*24+4:(index-5)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-5)*24+12:(index-5)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-5)*24+20:(index-5)*24+22]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+6)*24:(index+6)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+6)*24+8:(index+6)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+6)*24+16:(index+6)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+7)*24+1:(index+7)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+7)*24+9:(index+7)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+7)*24+17:(index+7)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+8)*24+2:(index+8)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+8)*24+10:(index+8)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+8)*24+18:(index+8)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-7)*24+6:(index-7)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-7)*24+14:(index-7)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-7)*24+22:(index-7)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-6)*24+5:(index-6)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-6)*24+13:(index-6)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-6)*24+21:(index-6)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-5)*24+4:(index-5)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-5)*24+12:(index-5)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-5)*24+20:(index-5)*24+22]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+6)*24:(index+6)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+6)*24+8:(index+6)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+6)*24+16:(index+6)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+7)*24+1:(index+7)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+7)*24+9:(index+7)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+7)*24+17:(index+7)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+8)*24+2:(index+8)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+8)*24+10:(index+8)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+8)*24+18:(index+8)*24+20]
        elif index==14:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-7)*24+6:(index-7)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-7)*24+14:(index-7)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-7)*24+22:(index-7)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-6)*24+5:(index-6)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-6)*24+13:(index-6)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-6)*24+21:(index-6)*24+23]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+6)*24:(index+6)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+6)*24+8:(index+6)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+6)*24+16:(index+6)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+7)*24+1:(index+7)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+7)*24+9:(index+7)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+7)*24+17:(index+7)*24+19]
                
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-7)*24+6:(index-7)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-7)*24+14:(index-7)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-7)*24+22:(index-7)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-6)*24+5:(index-6)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-6)*24+13:(index-6)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-6)*24+21:(index-6)*24+23]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+6)*24:(index+6)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+6)*24+8:(index+6)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+6)*24+16:(index+6)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+7)*24+1:(index+7)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+7)*24+9:(index+7)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+7)*24+17:(index+7)*24+19]
        elif index==15:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-6)*24+4:(index-6)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-6)*24+12:(index-6)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-6)*24+20:(index-6)*24+22]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-6)*24+4:(index-6)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-6)*24+12:(index-6)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-6)*24+20:(index-6)*24+22]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
        elif index==16:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-7)*24+5:(index-7)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-7)*24+13:(index-7)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-7)*24+21:(index-7)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-6)*24+4:(index-6)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-6)*24+12:(index-6)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-6)*24+20:(index-6)*24+22]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-7)*24+5:(index-7)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-7)*24+13:(index-7)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-7)*24+21:(index-7)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-6)*24+4:(index-6)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-6)*24+12:(index-6)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-6)*24+20:(index-6)*24+22]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
        elif index in [17,18,19,20]:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-8)*24+6:(index-8)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-8)*24+14:(index-8)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-8)*24+22:(index-8)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-7)*24+5:(index-7)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-7)*24+13:(index-7)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-7)*24+21:(index-7)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-6)*24+4:(index-6)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-6)*24+12:(index-6)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-6)*24+20:(index-6)*24+22]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-8)*24+6:(index-8)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-8)*24+14:(index-8)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-8)*24+22:(index-8)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-7)*24+5:(index-7)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-7)*24+13:(index-7)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-7)*24+21:(index-7)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-6)*24+4:(index-6)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-6)*24+12:(index-6)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-6)*24+20:(index-6)*24+22]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
        elif index==21:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-8)*24+6:(index-8)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-8)*24+14:(index-8)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-8)*24+22:(index-8)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-7)*24+5:(index-7)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-7)*24+13:(index-7)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-7)*24+21:(index-7)*24+23]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-8)*24+6:(index-8)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-8)*24+14:(index-8)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-8)*24+22:(index-8)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-7)*24+5:(index-7)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-7)*24+13:(index-7)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-7)*24+21:(index-7)*24+23]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
        elif index==22:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-6)*24+4:(index-6)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-6)*24+12:(index-6)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-6)*24+20:(index-6)*24+22]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
                
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-6)*24+4:(index-6)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-6)*24+12:(index-6)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-6)*24+20:(index-6)*24+22]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
        elif index==23:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
        elif index in [24,25,26,27,28,31,32,33,sample,35,36]:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-9)*24+6:(index-9)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-9)*24+14:(index-9)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-9)*24+22:(index-9)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-9)*24+6:(index-9)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-9)*24+14:(index-9)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-9)*24+22:(index-9)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
        elif index in [29,37]:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-9)*24+6:(index-9)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-9)*24+14:(index-9)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-9)*24+22:(index-9)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,54:56]=data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                seq[index*sample+i,:,56:58]=data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                seq[index*sample+i,:,58:60]=data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-9)*24+6:(index-9)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-9)*24+14:(index-9)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-9)*24+22:(index-9)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,54:56]=test_data[i*sub:(i+1)*sub,(index+7)*24:(index+7)*24+2]
                test_seq[index*sample+i,:,56:58]=test_data[i*sub:(i+1)*sub,(index+7)*24+8:(index+7)*24+10]
                test_seq[index*sample+i,:,58:60]=test_data[i*sub:(i+1)*sub,(index+7)*24+16:(index+7)*24+18]
				
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
        elif index==30:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
                
                seq[index*sample+i,:,60:62]=data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                seq[index*sample+i,:,62:64]=data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                seq[index*sample+i,:,64:66]=data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                seq[index*sample+i,:,66:68]=data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                seq[index*sample+i,:,68:70]=data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                seq[index*sample+i,:,70:72]=data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]

                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
                
                test_seq[index*sample+i,:,60:62]=test_data[i*sub:(i+1)*sub,(index+8)*24+1:(index+8)*24+3]
                test_seq[index*sample+i,:,62:64]=test_data[i*sub:(i+1)*sub,(index+8)*24+9:(index+8)*24+11]
                test_seq[index*sample+i,:,64:66]=test_data[i*sub:(i+1)*sub,(index+8)*24+17:(index+8)*24+19]
                
                test_seq[index*sample+i,:,66:68]=test_data[i*sub:(i+1)*sub,(index+9)*24+2:(index+9)*24+4]
                test_seq[index*sample+i,:,68:70]=test_data[i*sub:(i+1)*sub,(index+9)*24+10:(index+9)*24+12]
                test_seq[index*sample+i,:,70:72]=test_data[i*sub:(i+1)*sub,(index+9)*24+18:(index+9)*24+20]
                
        elif index==38:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]

            	test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
        elif index in [39,40,41,42,43,44]:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-9)*24+6:(index-9)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-9)*24+14:(index-9)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-9)*24+22:(index-9)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                seq[index*sample+i,:,36:38]=data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                seq[index*sample+i,:,38:40]=data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                seq[index*sample+i,:,40:42]=data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                seq[index*sample+i,:,48:50]=data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                seq[index*sample+i,:,50:52]=data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                seq[index*sample+i,:,52:54]=data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-9)*24+6:(index-9)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-9)*24+14:(index-9)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-9)*24+22:(index-9)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                test_seq[index*sample+i,:,36:38]=test_data[i*sub:(i+1)*sub,(index-7)*24+4:(index-7)*24+6]
                test_seq[index*sample+i,:,38:40]=test_data[i*sub:(i+1)*sub,(index-7)*24+12:(index-7)*24+14]
                test_seq[index*sample+i,:,40:42]=test_data[i*sub:(i+1)*sub,(index-7)*24+20:(index-7)*24+22]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,48:50]=test_data[i*sub:(i+1)*sub,(index+1)*24+3:(index+1)*24+5]
                test_seq[index*sample+i,:,50:52]=test_data[i*sub:(i+1)*sub,(index+1)*24+11:(index+1)*24+13]
                test_seq[index*sample+i,:,52:54]=test_data[i*sub:(i+1)*sub,(index+1)*24+19:(index+1)*24+21]
        elif index==45:
            for i in range(0,sample):
                seq[index*sample+i,:,0:24]=data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                seq[index*sample+i,:,24:26]=data[i*sub:(i+1)*sub,(index-9)*24+6:(index-9)*24+8]
                seq[index*sample+i,:,26:28]=data[i*sub:(i+1)*sub,(index-9)*24+14:(index-9)*24+16]
                seq[index*sample+i,:,28:30]=data[i*sub:(i+1)*sub,(index-9)*24+22:(index-9)*24+24]
				
                seq[index*sample+i,:,30:32]=data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                seq[index*sample+i,:,32:sample]=data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                seq[index*sample+i,:,sample:36]=data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                seq[index*sample+i,:,42]=data[i*sub:(i+1)*sub,(index-1)*24]
                seq[index*sample+i,:,43:45]=data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                seq[index*sample+i,:,45:47]=data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                seq[index*sample+i,:,48]=data[i*sub:(i+1)*sub,(index-1)*24+23]
				
                test_seq[index*sample+i,:,0:24]=test_data[i*sub:(i+1)*sub,index*24:(index+1)*24]
				
                test_seq[index*sample+i,:,24:26]=test_data[i*sub:(i+1)*sub,(index-9)*24+6:(index-9)*24+8]
                test_seq[index*sample+i,:,26:28]=test_data[i*sub:(i+1)*sub,(index-9)*24+14:(index-9)*24+16]
                test_seq[index*sample+i,:,28:30]=test_data[i*sub:(i+1)*sub,(index-9)*24+22:(index-9)*24+24]
				
                test_seq[index*sample+i,:,30:32]=test_data[i*sub:(i+1)*sub,(index-8)*24+5:(index-8)*24+7]
                test_seq[index*sample+i,:,32:sample]=test_data[i*sub:(i+1)*sub,(index-8)*24+13:(index-8)*24+15]
                test_seq[index*sample+i,:,sample:36]=test_data[i*sub:(i+1)*sub,(index-8)*24+21:(index-8)*24+23]
				
                test_seq[index*sample+i,:,42]=test_data[i*sub:(i+1)*sub,(index-1)*24]
                test_seq[index*sample+i,:,43:45]=test_data[i*sub:(i+1)*sub,(index-1)*24+7:(index-1)*24+9]
                test_seq[index*sample+i,:,45:47]=test_data[i*sub:(i+1)*sub,(index-1)*24+15:(index-1)*24+17]
                test_seq[index*sample+i,:,48]=test_data[i*sub:(i+1)*sub,(index-1)*24+23]
		for i in range(0,sample):
			target[index*sample+i,:,:]=data[i*sub+1:(i+1)*sub+1,index*24:(index+1)*24]
		test_target[index*6800:(index+1)*6800,:]=test_data[:6800,index*24:(index+1)*24]
  
    print "seq:",seq[sample,:20,:6]
    seqs=seq.astype(floatX)
    targets=target.astype(floatX)
    [n_steps,n_seq,n_in]=seq.shape
	
    n_hidden=2*n_in
    n_out=24
    n_epochs=2500
	
    '''seqs = [i for i in seqs]
    targets = [i for i in targets]
	
    gradient_dataset = SequenceDataset([seqs, targets], batch_size=None,
    						number_batches=50)
    cg_dataset = SequenceDataset([seq, targets], batch_size=None,
							 number_batches=20)	
    
    model = MetaRNN(n_in=n_in, n_mul=n_hidden, n_out=n_out,
			activation='tanh')		
			
    opt = hf_optimizer(p=model.rnn.params, inputs=[model.x, model.y],
					   s=model.rnn.y_pred,
					   costs=[model.rnn.loss(model.y)], h=model.rnn.h)
					   
    opt.train(gradient_dataset, cg_dataset,num_updates=300)'''
    model = MetaRNN(n_in=n_in, n_mul=n_hidden, n_out=n_out, learning_rate=0.001,
                        learning_rate_decay=0.999, n_epochs=n_epochs, L1_reg=0.005,
                        activation = 'tanh', output_type='real',index=index)
    model.fit(seqs,targets,validation_frequency=500)
    print "traning over"
	
    path='/root/chentian/share/201605/ped1'
    path0 = path+'/predict.csv'
    f7 = file(path0,"a")
    for n_seq in range(0,1564):
        guess = model.predict(test_seq[n_seq])
        np.savetxt(f7,guess,delimiter=',')
    f7.close()

    error1=np.zeros([6800,1104])
    pre_f=file(path0,"r")
    pre=np.loadtxt(pre_f,delimiter=',')
    pre_f.close()
    path3= path+"/error.csv"
    f8 = file(path3,"a")
    error=(pre-test_target)**2
    for i in range(0,46):
        error1[:,i*24:(i+1)*24]=error[i*6800:(i+1)*6800,:]
    np.savetxt(f8,error1,delimiter=',')
    f8.close()
    print "Elapsed time: %f" % (time.time()-t0)            