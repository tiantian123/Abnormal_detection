import numpy as np



def threshold1(flag,target):
    """detect the anoraml"""
    numtrue=np.zeros(np.size(target[:,0]))
    numdetect=np.zeros(np.size(flag[:,0]))
        
    for i in range(0,1800):
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
    
    precision = tp/(tp+fp)
    f=float(2/((1/precision)+(1/tp)))
    '''print "precision",precision
    print 'F=',f
    print 'tpr:',tp
    print 'fpr:',fp
    #print 'the max f:',b'''
    return tp,fp,f,precision
    
def threshold2(flag,target):
    """detect the anoraml"""
    numtrue=np.zeros(np.size(target[:,0]))
    numdetect=np.zeros(np.size(flag[:,0]))
    
    for i in xrange(0,1800):
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
    
    precision = tp/(tp+fp)
    f=float(2/((1/precision)+(1/tp)))
    '''print "precision",precision
    print 'F=',f
    print 'tpr:',tp
    print 'fpr:',fp
    #print 'the max f:',b ''' 
    return tp,fp,f,precision
if __name__ == "__main__"    :
    f1=file("error.csv","r")
    d = np.loadtxt(f1,delimiter=',')
    f1.close()
    f2=file("ped2_locate.csv","r")
    tag= np.loadtxt(f2,delimiter=',')
    f1.close()
    tp_1=[]
    fp_1=[]
    f_1=[]
    pre_1=[]
    
    tp_2=[]
    fp_2=[]
    f_2=[]
    pre_2=[]
    
    for pre in xrange(0,1000000,50000):
        print "Pre=%f" % pre
        feature=np.zeros((1800,50))
        flag = np.zeros((1800,50))
        for i in range(0,1800):     
            for j in range (0,50):
                feature[i,j]=sum(d[i,j*24:(j+1)*24])
                if feature[i,j]>pre:
                    flag[i,j]=1
       
        target=tag[1:1801,:50]
        tp1,fp1,F1,pre1=threshold1(flag,target)
        tp2,fp2,F2,pre2=threshold2(flag,target)
        tp_1.append(tp1)
        fp_1.append(fp1)
        f_1.append(F1)
        pre_1.append(pre1)
        
        tp_2.append(tp2)
        fp_2.append(fp2)
        f_2.append(F2)
        pre_2.append(pre2)
    print "tp1:",tp_1
    print "fp1:",fp_1
    print "F1:",f_1
    print "pre1:",pre_1
    
    print "tp2:",tp_2
    print "fp2:",fp_2
    print "F2:",f_2
    print "pre2:",pre_2
    print "ENDING"