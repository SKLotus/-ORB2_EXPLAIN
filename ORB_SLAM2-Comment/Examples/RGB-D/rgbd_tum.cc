/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/


#include<iostream>
#include<algorithm>
#include<fstream>
#include<chrono>

#include<opencv2/core/core.hpp>

#include<System.h>

using namespace std;

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps);//调用的文件名，rgb、d、时间戳

int main(int argc, char **argv)
{
    if(argc != 5)
    {
		//判断输入参数个数
        cerr << endl << "Usage: ./rgbd_tum path_to_vocabulary path_to_settings path_to_sequence path_to_association" << endl;
        return 1;
    }

    // Retrieve paths to images，加载图片
    vector<string> vstrImageFilenamesRGB;
    vector<string> vstrImageFilenamesD;
    vector<double> vTimestamps;
    string strAssociationFilename = string(argv[4]);//从传递给程序的第四个参数指定的文本文件中加载RGB和深度图以及时间戳
    LoadImages(strAssociationFilename, vstrImageFilenamesRGB, vstrImageFilenamesD, vTimestamps);//LoadImages是输入，用空向量储存文件名和时间戳，再储存在相应向量中

    // Check consistency in the number of images and depthmaps，检查图片和输入深度图的一致性
    int nImages = vstrImageFilenamesRGB.size();//将rgb文件名的数量赋给nimage数组
    if(vstrImageFilenamesRGB.empty())//如果RGB文件里面是空的，返回错误值
    {
        cerr << endl << "No images found in provided path." << endl;
        return 1;
    }
    else if(vstrImageFilenamesD.size()!=vstrImageFilenamesRGB.size())//如果深度图的数量和RGB数量不同，返回错误值
    {
        cerr << endl << "Different number of images for rgb and depth." << endl;
        return 1;
    }

    // Create SLAM system. It initializes all system threads and gets ready to process frames.
    ORB_SLAM2::System SLAM(argv[1],argv[2],ORB_SLAM2::System::RGBD,true);//SLAM对象是一个 ORB_SLAM2::System 类型变量

    // Vector for tracking time statistics
    vector<float> vTimesTrack; //此代码片段创建一个名为“vTimesTrack”的浮点向量,该矢量的目的是存储用于跟踪的图像的时间戳
    vTimesTrack.resize(nImages); //向量的大小被设置为“nImages”，以确保它有足够的空间来存储所有图像的时间戳

    cout << endl << "-------" << endl;
    cout << "Start processing sequence ..." << endl;
    cout << "Images in the sequence: " << nImages << endl << endl;

    // Main loop
    cv::Mat imRGB, imD;
	//遍历图片，进行SLAM
    for(int ni=0; ni<nImages; ni++)//ni是循环到第几张图片
    {
        // Read image and depthmap from file
        imRGB = cv::imread(string(argv[3])+"/"+vstrImageFilenamesRGB[ni],CV_LOAD_IMAGE_UNCHANGED);
        imD = cv::imread(string(argv[3])+"/"+vstrImageFilenamesD[ni],CV_LOAD_IMAGE_UNCHANGED);//使用“+”运算符与目录路径连接。然后将加载的图像分别存储在“imRGB”和“imD”变量中
        double tframe = vTimestamps[ni];//当前帧的时间戳也是从“vTimestamps”向量中获得的，并存储在“tframe”变量中。该时间戳稍后用于帧到帧的运动估计和用于计算相机的轨迹。

        if(imRGB.empty())
        {
            cerr << endl << "Failed to load image at: "
                 << string(argv[3]) << "/" << vstrImageFilenamesRGB[ni] << endl;
            return 1;
        }

#ifdef COMPILEDWITHC11//满足C11标准库，执行以下语句
        std::chrono::steady_clock::time_point t1 = std::chrono::steady_clock::now();
#else//否则执行以下语句
        std::chrono::monotonic_clock::time_point t1 = std::chrono::monotonic_clock::now();//计时开始
#endif

        // Pass the image to the SLAM system
        SLAM.TrackRGBD(imRGB,imD,tframe);//对当前帧执行跟踪线程

#ifdef COMPILEDWITHC11
        std::chrono::steady_clock::time_point t2 = std::chrono::steady_clock::now();
#else
        std::chrono::monotonic_clock::time_point t2 = std::chrono::monotonic_clock::now();
#endif

        double ttrack= std::chrono::duration_cast<std::chrono::duration<double> >(t2 - t1).count();//计算跟踪线程所需时间

        vTimesTrack[ni]=ttrack;//把跟踪当前帧的时间赋值给vTimesTrack[ni]

        // Wait to load the next frame
        double T=0;
        if(ni<nImages-1)
            T = vTimestamps[ni+1]-tframe;//即vTimestamps[ni+1]-vTimestamps[ni]，两个时间戳相减
        else if(ni>0)
            T = tframe-vTimestamps[ni-1];//得到的T

        if(ttrack<T)//如果跟踪当前帧所需的时间小于实际拍摄经历的时间
            usleep((T-ttrack)*1e6);//把调用该函数的线程挂起一段时间，等待下一帧
    }

    // Stop all threads
    SLAM.Shutdown();

    // Tracking time statistics
    sort(vTimesTrack.begin(),vTimesTrack.end());//排序，从小到大，计算中位数和平均跟踪时间
    float totaltime = 0;
    for(int ni=0; ni<nImages; ni++)
    {
        totaltime+=vTimesTrack[ni];//计算总时间
    }
    cout << "-------" << endl << endl;
    cout << "median tracking time: " << vTimesTrack[nImages/2] << endl;
    cout << "mean tracking time: " << totaltime/nImages << endl;

    // Save camera trajectory
    SLAM.SaveTrajectoryTUM("CameraTrajectory.txt");
    SLAM.SaveKeyFrameTrajectoryTUM("KeyFrameTrajectory.txt");   

    return 0;
}

void LoadImages(const string &strAssociationFilename, vector<string> &vstrImageFilenamesRGB,
                vector<string> &vstrImageFilenamesD, vector<double> &vTimestamps)
{
    ifstream fAssociation;
    fAssociation.open(strAssociationFilename.c_str());
    while(!fAssociation.eof())
    {
        string s;
        getline(fAssociation,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB, sD;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenamesRGB.push_back(sRGB);
            ss >> t;
            ss >> sD;
            vstrImageFilenamesD.push_back(sD);//该函数从打开文本文件开始，然后读取每一行，直到到达文件的末尾。对于每一行，它使用字符串流提取时间戳、RGB图像文件名和深度图像文件名。然后，它将这些值添加到相应的向量中。

        }
    }
}
