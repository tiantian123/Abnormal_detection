#include "stdafx.h"
#include <iostream>  
#include "opencv2/opencv.hpp"  
#include<fstream>
#include<math.h>
#include<cmath>    
using namespace cv;  
using namespace std;  
#define UNKNOWN_FLOW_THRESH 1e9  

void staticOpticalFlow(Mat flow){ 
    // determine motion range:  
    float maxrad = -1;  
    int rowup=0;
	int rowdown=0;
	int colup=0;
	int coldown=0;
	int statics[1104]={0};
	
    // Find max flow to normalize fx and fy  
	/*
	for (int i= 0; i < flow.rows; ++i){  
        for (int j = 0; j < flow.cols; ++j){  
            Vec2f flow_at_point = flow.at<Vec2f>(i, j);  
            float fx = flow_at_point[0];  
            float fy = flow_at_point[1];  
            if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH))  
                continue;  
            float rad = sqrt(fx * fx + fy * fy);  //光流大小
			if(rad<0.3){//将小于0.3的光流值变成0.
				flow_at_point[0]=0;
				flow_at_point[1]=0;
				flow.at<Vec2f>(i,j)=flow_at_point;
		    }
			rad=rad<0.3?0:rad;
            maxrad = maxrad > rad ? maxrad : rad;  
        }  		
    }  
	*/
	int nn=-1;
	for(int k=0;k<56;k++){
		if(!(k==0||k==1||k==2||k==3||k==8||k==9||k==10||k==16||k==17||k==24)){
			nn++;
		    rowdown=18+(int(k/8)*20);//记录网格上边的位置
		    rowup=38+(int(k/8)*20);  //记录网格下边的位置
            coldown=(k%8)*30;        //记录网格左边的位置
			colup=(k%8)*30+30;       //记录网格右边的位置
			if(k==7||k==15||k==23||k==31||k==39||k==47||k==55)
				colup-=2;
			for (int i= rowdown; i <rowup; ++i)//对当前网格中的每个像素进行处理
			{  
				for (int j = coldown; j < colup; ++j)   
				{  
                 
					Vec2f flow_at_point = flow.at<Vec2f>(i, j);  //光流是一个向量，有两个分量
					float fx = flow_at_point[0]; //读取x轴的坐标值
					float fy = flow_at_point[1]; //读取y轴的坐标值
					if ((fabs(fx) >  UNKNOWN_FLOW_THRESH) || (fabs(fy) >  UNKNOWN_FLOW_THRESH)){//当光流值超出预定的最大值时，改为maxrad  
						if (fabs(fx)>UNKNOWN_FLOW_THRESH)
							fx=maxrad;
						if (fabs(fy)>UNKNOWN_FLOW_THRESH)
							fy=maxrad;
					}  
					float rad = sqrt(fx * fx + fy * fy);  //光流大小
					float angle = atan2(fy, fx);          //光流的角度
					if(rad<1&&rad>=0.3)
					{   //统计大小大于0.3小于1的光流
						int tem1=0;
						int b=8*angle/(2*CV_PI);
						int tem=floor(8*angle/(2*CV_PI));
						if((8*angle/(2*CV_PI))>=tem+0.5)
							tem1=tem+1;
						else 
							tem1=tem;
						statics[((tem1+8)%8)+nn*24]++;
					}
				
					else if(rad>=1&&rad<3)
					{//统计大小大于1小于3的光流
						int tem1=0;
						int tem=floor(8*angle/(2*CV_PI));
						if((8*angle/(2*CV_PI))>=tem+0.5)
							tem1=tem+1;
						else 
							tem1=tem;
						statics[((tem1+8)%8)+8+nn*24]++;
					}
					else if(rad>=3)
					{//统计大小大于3的光流
						int tem1=0;
						int tem=floor(8*angle/(2*CV_PI));
						if((8*angle/(2*CV_PI))>=tem+0.5)
							tem1=tem+1;
						else 
							tem1=tem;
						statics[((tem1+8)%8)+16+nn*24]++;
					}
				}  
			}
		}  
	}	
	ofstream out("ped1test24.csv",std::ios_base::app);//保存到ped1test24中
	for(int i=0;i<1104;i++){
		if(i!=1103)
		out<<statics[i]<<",";
		else
			out<<statics[i];
	}
	out<<endl;
	out.close();
		
}  


int main(int, char**){  
    Mat prevgray, gray, flow, cflow, frame,flow1,flow2;  
    namedWindow("flow", 1);  
    namedWindow("original1",1);
	namedWindow("original",1);
	int num=200;
	for (int j=1;j<=36;j++){
		string pathroot="E:\\项目\\code\\UCSD_Anomaly_Dataset\\UCSD_Anomaly_Dataset.v1p2\\UCSDped1\\Train\\Train001";//你的数据的位置
		char b[10];
		sprintf_s(b,"%d",j);
		string path=pathroot+b;
		string path1=path+"\\001.tif";
        frame=imread(path1);
        imshow("original1",frame);  //显示图像
        cvtColor(frame, prevgray, CV_BGR2GRAY);  
        Ptr<DenseOpticalFlow>tvl1 = createOptFlow_DualTVL1();   
        for(int i=2;i<=num;i++){ 
		    double t = (double)cvGetTickCount();  
		    char a[10];
		    string temp =path;
		    if(i<=9){
				sprintf_s(a,"%d",i);
				temp+="\\00";
				temp+=a;
				temp+=".tif";
			}
			else if(i>9&&i<=99){
				sprintf_s(a,"%d",i);
				temp+="\\0";
				temp+=a;
				temp+=".tif";
			}
			else{
				sprintf_s(a,"%d",i);
				temp+="\\";
				temp+=a;
				temp+=".tif";
			}
			frame=imread(temp);
			imshow("original",frame);
			cvtColor(frame, gray, CV_BGR2GRAY);//转化为灰度图像
			if( prevgray.data )  
			{  
			// calcOpticalFlowFarneback(prevgray, gray, flow, 0.5, 3, 15, 3, 5, 1.2, 0);  
			 //cvCalcOpticalFlowLK(prevgray,gray,(1,1),flow1,flow2);
			 tvl1->calc(prevgray, gray, flow);//计算光流值，保存到flow中，其中prevgray为当前帧，gray为下一帧
			 //calcOpticalFlowSF(prevgray, gray, flow,3, 2, 4, 4.1, 25.5, 18, 55.0, 25.5, 0.35, 18, 55.0, 25.5, 10);
			 staticOpticalFlow(flow); //进行统计
			 //imshow("flow", motion2color);  
			}  
        std::swap(prevgray, gray);  //交换
        t = (double)cvGetTickCount() - t;  
        cout << "cost time: " << t / ((double)cvGetTickFrequency()*1000.) << endl;  //输出时间
		//waitKey();
	   }
	}
        return 0;  
    }  