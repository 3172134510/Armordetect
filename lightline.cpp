#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;
using namespace ml;
Mat videoimage;

int value = 1;
int value2 = -5;
void onmouse2(int, void *)
{
}
void onmouse(int, void *)
{
}

double getDistance(Point pointO, Point pointA) // 计算距离

{

    double distance;

    distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);

    distance = sqrtf(distance);

    return distance;
}

Mat transform_Mat(Point point1,Point point2,Point point3,Point point4)//透视变换
{
    Mat img; 
Point2f t1[4]={point1,point2,point3,point4};
Point2f t2[4] = {Point(0,0),Point(0,100),Point(100,100),Point(100,0)};
  Mat transforMatrix = getPerspectiveTransform(t1,t2);
warpPerspective(videoimage,img,transforMatrix,Size(100,100));

return img;

}




void svm_predict(Mat detect) // svm检测
{
    Ptr<ml::SVM> svm = ml::SVM::load("/home/ljj/Desktop/3_svm.xml");
    Mat src = detect.clone();
    Mat input;
    resize(src, src, Size(20, 20));
    src = src.reshape(1, 1); // 输入图片序列化
    input.push_back(src);
    input.convertTo(input, CV_32FC1); // 更改图片数据的类型，必要，不然会出错  CV_32FC1
    float r = svm->predict(input);
    // 对所有行进行预测
    if (r == 1)
        cout << "3" << endl;
    else
        cout << "4" << endl;
}

int main()
{
    VideoCapture video("test.avi");
    namedWindow("canny阈值", WINDOW_FREERATIO);
    namedWindow("阈值2", WINDOW_FREERATIO);
    createTrackbar("canny阈值", "canny阈值", &value, 20, onmouse);
    createTrackbar("阈值2", "阈值2", &value2, 20, onmouse2);
    while (1)
    {

        video.read(videoimage);
        if (videoimage.empty())
            break;

        onmouse2(value2, 0);
        onmouse(value, 0);
        Mat videoimage_1;
        vector<Mat> hsvsplits;
        cvtColor(videoimage, videoimage_1, COLOR_BGR2HSV);
        split(videoimage_1, hsvsplits);
        
        equalizeHist(hsvsplits[2], hsvsplits[2]); // 均质化图像(消除地面反光)
        merge(hsvsplits, videoimage_1);

        Mat threSHOLD;
        threshold(hsvsplits[2], threSHOLD, 250, 255, THRESH_BINARY);
        blur(threSHOLD, videoimage_1, Size(3, 3));
        Mat kennel1 = getStructuringElement(MORPH_ELLIPSE, Size(3 ,3));
        morphologyEx(videoimage_1,videoimage_1,MORPH_OPEN,kennel1);

        vector<vector<Point>> contours;
        vector<Point> hierarchy;

        findContours(videoimage_1, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);


        vector<RotatedRect> light;                // 灯条信息
        for (int i = 0; i < contours.size(); i++) // 椭圆拟合
        {

            if (contours[i].size() > 5)
            {
                RotatedRect elipse__light = fitEllipse(contours[i]);
                float w = elipse__light.size.width;  // 椭圆短轴
                float h = elipse__light.size.height; // 椭圆长轴
                float arear = w * h;
                Point elipse__light_center = elipse__light.center; // 中心点
                if (w / h < 0.4)                                   // 筛选

                    if (arear > 200 && arear < 800)

                        if (elipse__light.angle > 95 || elipse__light.angle < 15)

                            if (elipse__light.center.y > 100)

                            {
                                ellipse(videoimage, elipse__light, Scalar(255, 0, 0), 2);//框灯条
                                light.push_back(elipse__light);
                            }
            }
        }

        // 灯条排序
        for (int i = 0; i < light.size(); i++)
        {
            for (int j = i + 1; j < light.size(); j++)
            {
                if (light[i].center.x > light[j].center.x)
                {
                    RotatedRect temp;
                    temp.center = light[i].center;
                    light[i].center = light[j].center;
                    light[j].center = temp.center;
                }
            }
        }



for(int i=0;i<light.size();i++)
{
    light[i].size.height = light[i].size.height*2;//延长灯条
}


for(int i=0;i<light.size();i++)
{
    float x1,y1,x2,y2,x3,y3,x4,y4;//(逆时针)四个装甲板顶点
    Rect temp = light[i].boundingRect2f();
    x1 = temp.x + temp.width;
    y1 = temp.y;
    x2 = x1;
    y2 = y1 + temp.height;

        Rect temp2 = light[i+1].boundingRect2f();
        x3 = temp2.x;
        y3 = temp2.y + temp2.height;
        x4 = x3;
        y4 = temp2.y;
double dis = getDistance(Point(x1,y1),Point(x4,y4));//两顶点距离
    //rectangle(videoimage,temp,Scalar(255,200,100,2));
    if(dis<120&&dis>50)//筛选装甲板
    {
        line(videoimage,Point(x1,y1),Point(x2,y2),Scalar(0,255,0),2);
        line(videoimage,Point(x2,y2),Point(x3,y3),Scalar(0,255,0),2);
        line(videoimage,Point(x3,y3),Point(x4,y4),Scalar(0,255,0),2);
        line(videoimage,Point(x4,y4),Point(x1,y1),Scalar(0,255,0),2);
        Mat transfored_img = transform_Mat(Point(x1,y1),Point(x2,y2),Point(x3,y3),Point(x4,y4));//透视变换
        imshow("transform",transfored_img);
        svm_predict(transfored_img);//预测
    }


   i++;
}



        for (int i = 0; i < light.size(); i++)
        {
            
            circle(videoimage, light[i].center, 1, Scalar(25, 30, 10), 2);
        }
        resize(videoimage,videoimage,Size(500,500));
        imshow("test", videoimage);
        
        imshow("video", videoimage_1);
        //if(waitKey(0) == ' ')break;
        waitKey(1);
    }

    return 0;

}
