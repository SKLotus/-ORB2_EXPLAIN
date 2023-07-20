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



#include "System.h"
#include "Converter.h"
#include <thread>
#include <pangolin/pangolin.h>
#include <iomanip>

namespace ORB_SLAM2
{
	
System::System(const string &strVocFile, //词典文件路径
	           const string &strSettingsFile, //配置文件路径
	           const eSensor sensor,//传感器类型
               const bool bUseViewer)://是否使用可视化界面
	           mSensor(sensor), //初始化传感器类型
	           mpViewer(static_cast<Viewer*>(NULL)), //空。。。对象指针？  TODO 
	           mbReset(false),//无复位标志
	           mbActivateLocalizationMode(false),//没有这个模式转换标志
               mbDeactivateLocalizationMode(false)//没有这个模式转换标志

{
    // Output welcome message
    cout << endl <<
    "ORB-SLAM2 Copyright (C) 2014-2016 Raul Mur-Artal, University of Zaragoza." << endl <<
    "This program comes with ABSOLUTELY NO WARRANTY;" << endl  <<
    "This is free software, and you are welcome to redistribute it" << endl <<
    "under certain conditions. See LICENSE.txt." << endl << endl;

    cout << "Input sensor was set to: ";

	// 输出当前传感器类型
    if(mSensor==MONOCULAR)
        cout << "Monocular" << endl;
    else if(mSensor==STEREO)
        cout << "Stereo" << endl;
    else if(mSensor==RGBD)
        cout << "RGB-D" << endl;

	// step1. 初始化各成员变量
    // step1.1. 读取配置文件信息
    //Check settings file
    cv::FileStorage fsSettings(strSettingsFile.c_str(), cv::FileStorage::READ);//将配置文件名转换成为字符串，只读
    if(!fsSettings.isOpened()) //如果打开失败，就输出调试信息
    {
       cerr << "Failed to open settings file at: " << strSettingsFile << endl;
       exit(-1); //然后退出
    }


    //Load ORB Vocabulary
    cout << endl << "Loading ORB Vocabulary. This could take a while..." << endl;

	// step1.2. 创建ORB词袋
    mpVocabulary = new ORBVocabulary();
    bool bVocLoad = mpVocabulary->loadFromTextFile(strVocFile); //获取字典加载状态，strVocFile 就是命令行传入的参数 Vocabulary/ORBvoc.txt，是离线训练而来的文件
	/*
		首先图像提取ORB 特征点，将描述子通过 k - means 进行聚类，根据设定的树的分支数和深度，
		从叶子节点开始聚类一直到根节点，最后得到一个非常大的 vocabulary tree

		1、遍历所有的训练图像，对每幅图像提取ORB特征点。
		2、设定vocabulary tree的分支数K和深度L。将特征点的每个描述子用 K - means聚类，变成 K个集合，
		作为vocabulary tree 的第1层级，然后对每个集合重复该聚类操作，就得到了vocabulary tree的第2层级，
		继续迭代最后得到满足条件的vocabulary tree，它的规模通常比较大，比如ORB - SLAM2使用的离线字典就有108万 + 个节点。
		3、离根节点最远的一层节点称为叶子或者单词 Word。根据每个Word 在训练集中的相关
		程度给定一个权重weight，训练集里出现的次数越多，说明辨别力越差，给与的权重越低。

    */
    if(!bVocLoad)//如果加载失败，就输出调试信息
    {
        cerr << "Wrong path to vocabulary. " << endl;
        cerr << "Falied to open at: " << strVocFile << endl;
        exit(-1); //然后退出
    }
    cout << "Vocabulary loaded!" << endl << endl; //否则则说明加载成功

    //Create KeyFrame Database
	// step1.3. 创建关键帧数据库,主要保存ORB描述子倒排索引(即根据描述子查找拥有该描述子的关键帧)
    mpKeyFrameDatabase = new KeyFrameDatabase(*mpVocabulary);

    //Create the Map
	// step1.4. 创建地图
    mpMap = new Map();

	
    //Create Drawers. These are used by the Viewer，这里的帧绘制器和地图绘制器将会被可视化的Viewer所使用
	mpFrameDrawer = new FrameDrawer(mpMap);
    mpMapDrawer = new MapDrawer(mpMap, strSettingsFile);

	// step2. 创建3大线程: Tracking、LocalMapping和LoopClosing
	// step2.1. 主线程就是Tracking线程,只需创建Tracking对象即可
    //Initialize the Tracking thread
    //(it will live in the main thread of execution, the one that called this constructor)
    mpTracker = new Tracking(this, //现在还不是很明白为什么这里还需要一个this指针  TODO  
		                     mpVocabulary, //字典
		                     mpFrameDrawer, //帧绘制器
		                     mpMapDrawer,//地图绘制器
		                     mpMap, //地图
		                     mpKeyFrameDatabase, //关键帧地图
		                     strSettingsFile, //设置文件路径
		                     mSensor);//传感器类型，主线程，下面是子线程
	/*
	上面主要初始化了追踪线程（或者说主线程），同时我们可以看到其初始化参数中包含了 mpFrameDrawer ， mpMapDrawer 。这两个对象主要是负责对帧与地图的绘画。
	下面是运行局部建图和闭环线程
	*/

	// step2.2. 创建LocalMapping线程及mpLocalMapper
    //Initialize the Local Mapping thread and launch
    mpLocalMapper = new LocalMapping(mpMap, mSensor==MONOCULAR);//指定使iomanip，判断传感器是否为单目
    mptLocalMapping = new thread(&ORB_SLAM2::LocalMapping::Run,mpLocalMapper);//这个线程会调用的函数LocalMapping，以及该函数的参数mpLocalMapper

	// step2.3. 创建LoopClosing线程及mpLoopCloser
    //Initialize the Loop Closing thread and launch
    mpLoopCloser = new LoopClosing(mpMap, mpKeyFrameDatabase, mpVocabulary, mSensor!=MONOCULAR);//地图、关键帧数据库、ORB字典、当前传感器是否不为单目
    mptLoopClosing = new thread(&ORB_SLAM2::LoopClosing::Run, mpLoopCloser);//这个线程会调用的函数LoopClosing，以及该函数的参数mpLoopCloser

    //Initialize the Viewer thread and launch，这里主要是可视化线程的运行
    if(bUseViewer)
		//如果指定了，程序的运行过程中需要运行可视化部分
    	//新建viewer
    {
        mpViewer = new Viewer(this, mpFrameDrawer,mpMapDrawer,mpTracker,strSettingsFile);//this指针、帧绘制器、地图绘制器、追踪器、配置文件的访问路径
        mptViewer = new thread(&Viewer::Run, mpViewer);//新建viewer线程
        mpTracker->SetViewer(mpViewer); //给运动追踪器设置其查看器
    }


	/*构造函数的主要流程
	1、加载ORB词汇表，构建Vocabulary，以及关键帧数据集库
	2、初始化追踪主线程，但是未运行
	3、初始化局部建图，回环闭合线程，且运行
	4、创建可视化线程，并且与追踪主线程关联起来。
	*/

	// step3. 设置线程间通信指针
    //Set pointers between threads  建立组件连接，实现相互通信和共享信息
    mpTracker->SetLocalMapper(mpLocalMapper);
    mpTracker->SetLoopClosing(mpLoopCloser);

    mpLocalMapper->SetTracker(mpTracker);
    mpLocalMapper->SetLoopCloser(mpLoopCloser);

    mpLoopCloser->SetTracker(mpTracker);
    mpLoopCloser->SetLocalMapper(mpLocalMapper);
}

cv::Mat System::TrackStereo(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timestamp)//调用双目跟踪函数
{
    if(mSensor!=STEREO)
    {
        cerr << "ERROR: you called TrackStereo but input sensor was not set to STEREO." << endl;
        exit(-1);
    }   

    // Check mode change，加锁
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)//判断是否处于纯定位
        {
            mpLocalMapper->RequestStop();//停止局部建图

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);//执行仅跟踪
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);//不仅仅执行跟踪
            mpLocalMapper->Release();//释放恢复局部建图线程
            mbDeactivateLocalizationMode = false;
        }
    }//释放锁

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageStereo(imLeft,imRight,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackRGBD(const cv::Mat &im, const cv::Mat &depthmap, const double &timestamp)//调用rgbd跟踪函数
{
    if(mSensor!=RGBD)
    {
        cerr << "ERROR: you called TrackRGBD but input sensor was not set to RGBD." << endl;
        exit(-1);
    }    

    // Check mode change
    {
        unique_lock<mutex> lock(mMutexMode);
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }

            mpTracker->InformOnlyTracking(true);
            mbActivateLocalizationMode = false;
        }
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }

    cv::Mat Tcw = mpTracker->GrabImageRGBD(im,depthmap,timestamp);

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;
    return Tcw;
}

cv::Mat System::TrackMonocular(const cv::Mat &im, const double &timestamp)//调用单目跟踪函数
{
    if(mSensor!=MONOCULAR)
    {
        cerr << "ERROR: you called TrackMonocular but input sensor was not set to Monocular." << endl;
        exit(-1);
    }

    // Check mode change
    {
		// 独占锁，主要是为了mbActivateLocalizationMode和mbDeactivateLocalizationMode不会发生混乱
        unique_lock<mutex> lock(mMutexMode);
		// mbActivateLocalizationMode为true会关闭局部地图线程
        if(mbActivateLocalizationMode)
        {
            mpLocalMapper->RequestStop();

            // Wait until Local Mapping has effectively stopped
            while(!mpLocalMapper->isStopped())
            {
                usleep(1000);
            }
			// 局部地图关闭以后，只进行追踪的线程，只计算相机的位姿，没有对局部地图进行更新
		    // 设置mbOnlyTracking为真
            mpTracker->InformOnlyTracking(true);
			//关闭线程可以使别的线程拿到更多资源
            mbActivateLocalizationMode = false;
        }
		// 如果mbDeactivateLocalizationMode是true，局部地图线程就被释放, 关键帧从局部地图中删除.
        if(mbDeactivateLocalizationMode)
        {
            mpTracker->InformOnlyTracking(false);
            mpLocalMapper->Release();
            mbDeactivateLocalizationMode = false;
        }
    }

    // Check reset
    {
    unique_lock<mutex> lock(mMutexReset);
    if(mbReset)
    {
        mpTracker->Reset();
        mbReset = false;
    }
    }
	//获取相机位姿的估计结果
    cv::Mat Tcw = mpTracker->GrabImageMonocular(im,timestamp);//此处调用了灰度图处理程序

    unique_lock<mutex> lock2(mMutexState);
    mTrackingState = mpTracker->mState;
    mTrackedMapPoints = mpTracker->mCurrentFrame.mvpMapPoints;
    mTrackedKeyPointsUn = mpTracker->mCurrentFrame.mvKeysUn;

    return Tcw;
}

/*
	1、判断传感器的类型是否为单目模式，如果不是，则表示设置错误，函数直接返回

	2、上锁 模式锁(mMutexMode):
		(1)如果目前需要激活定位模式，则请求停止局部建图，并且等待局部建图线程停止，设置为仅追踪模式。
		(2)如果目前需要取消定位模式，则通知局部建图可以工作了，关闭仅追踪模式

	3、上锁 复位锁(mMutexReset): 检查是否存在复位请求，如果有，则进行复位操作

	4、核心部分: 根据输入的图像获得相机位姿态（其中包含了特征提取匹配，地图初始化，关键帧查询等操作）

	5、进行数据更新，如追踪状态、当前帧的地图点、当前帧矫正之后的关键点等。

*/

void System::ActivateLocalizationMode()//激活纯定位模式
{
    unique_lock<mutex> lock(mMutexMode);
    mbActivateLocalizationMode = true;
}

void System::DeactivateLocalizationMode()//不激活纯定位模式
{
    unique_lock<mutex> lock(mMutexMode);
    mbDeactivateLocalizationMode = true;
}

bool System::MapChanged()
{
    static int n=0;
    int curn = mpMap->GetLastBigChangeIdx();//检查地图最后一次大更改的索引
    if(n<curn)//如果索引大于n，说明自上次调用此函数以来，地图已经发生更改，更新索引
    {
        n=curn;
        return true;
    }
    else
        return false;
}

void System::Reset()
{
    unique_lock<mutex> lock(mMutexReset);
    mbReset = true;
}

void System::Shutdown()
{
    mpLocalMapper->RequestFinish();
    mpLoopCloser->RequestFinish();
    if(mpViewer)
    {
        mpViewer->RequestFinish();
        while(!mpViewer->isFinished())
            usleep(5000);
    }

    // Wait until all thread have effectively stopped
    while(!mpLocalMapper->isFinished() || !mpLoopCloser->isFinished() || mpLoopCloser->isRunningGBA())
    {
        usleep(5000);
    }

    if(mpViewer)
        pangolin::BindToContext("ORB-SLAM2: Map Viewer");
}

void System::SaveTrajectoryTUM(const string &filename)//以TUM格式保存相机运动轨迹和关键帧位姿
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryTUM cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    list<bool>::iterator lbL = mpTracker->mlbLost.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(),
        lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++, lbL++)
    {
        if(*lbL)
            continue;//检查是否丢失，如果是就进行下一次迭代

        KeyFrame* pKF = *lRit;//否的话，使用迭代器lRit检索引用关键帧

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        // If the reference keyframe was culled, traverse the spanning tree to get a suitable keyframe.
        while(pKF->isBad())//如果关键帧不好，已经被剔除
        {
            Trw = Trw*pKF->mTcp;//遍历关键帧的生成树
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;//找到已知位姿的合适关键帧

        cv::Mat Tcw = (*lit)*Trw;//通过关键帧相对位姿和绝对位姿相乘来计算相机的绝对位姿
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();//时间、平移和旋转
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);//

        vector<float> q = Converter::toQuaternion(Rwc);//转换为四元数

        f << setprecision(6) << *lT << " " <<  setprecision(9) << twc.at<float>(0) << " " << twc.at<float>(1) << " " << twc.at<float>(2) << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}


void System::SaveKeyFrameTrajectoryTUM(const string &filename)//以TUM格式保存相机运动轨迹和关键帧位姿
{
    cout << endl << "Saving keyframe trajectory to " << filename << " ..." << endl;

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    //cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    for(size_t i=0; i<vpKFs.size(); i++)
    {
        KeyFrame* pKF = vpKFs[i];

       // pKF->SetPose(pKF->GetPose()*Two);

        if(pKF->isBad())
            continue;

        cv::Mat R = pKF->GetRotation().t();
        vector<float> q = Converter::toQuaternion(R);
        cv::Mat t = pKF->GetCameraCenter();
        f << setprecision(6) << pKF->mTimeStamp << setprecision(7) << " " << t.at<float>(0) << " " << t.at<float>(1) << " " << t.at<float>(2)
          << " " << q[0] << " " << q[1] << " " << q[2] << " " << q[3] << endl;

    }

    f.close();
    cout << endl << "trajectory saved!" << endl;
}

void System::SaveTrajectoryKITTI(const string &filename)//以KITTI格式保存相机运动轨迹和关键帧位姿
{
    cout << endl << "Saving camera trajectory to " << filename << " ..." << endl;
    if(mSensor==MONOCULAR)
    {
        cerr << "ERROR: SaveTrajectoryKITTI cannot be used for monocular." << endl;
        return;
    }

    vector<KeyFrame*> vpKFs = mpMap->GetAllKeyFrames();
    sort(vpKFs.begin(),vpKFs.end(),KeyFrame::lId);

    // Transform all keyframes so that the first keyframe is at the origin.
    // After a loop closure the first keyframe might not be at the origin.
    cv::Mat Two = vpKFs[0]->GetPoseInverse();

    ofstream f;
    f.open(filename.c_str());
    f << fixed;

    // Frame pose is stored relative to its reference keyframe (which is optimized by BA and pose graph).
    // We need to get first the keyframe pose and then concatenate the relative transformation.
    // Frames not localized (tracking failure) are not saved.

    // For each frame we have a reference keyframe (lRit), the timestamp (lT) and a flag
    // which is true when tracking failed (lbL).
    list<ORB_SLAM2::KeyFrame*>::iterator lRit = mpTracker->mlpReferences.begin();
    list<double>::iterator lT = mpTracker->mlFrameTimes.begin();
    for(list<cv::Mat>::iterator lit=mpTracker->mlRelativeFramePoses.begin(), lend=mpTracker->mlRelativeFramePoses.end();lit!=lend;lit++, lRit++, lT++)
    {
        ORB_SLAM2::KeyFrame* pKF = *lRit;

        cv::Mat Trw = cv::Mat::eye(4,4,CV_32F);

        while(pKF->isBad())
        {
          //  cout << "bad parent" << endl;
            Trw = Trw*pKF->mTcp;
            pKF = pKF->GetParent();
        }

        Trw = Trw*pKF->GetPose()*Two;

        cv::Mat Tcw = (*lit)*Trw;
        cv::Mat Rwc = Tcw.rowRange(0,3).colRange(0,3).t();
        cv::Mat twc = -Rwc*Tcw.rowRange(0,3).col(3);

        f << setprecision(9) << Rwc.at<float>(0,0) << " " << Rwc.at<float>(0,1)  << " " << Rwc.at<float>(0,2) << " "  << twc.at<float>(0) << " " <<
             Rwc.at<float>(1,0) << " " << Rwc.at<float>(1,1)  << " " << Rwc.at<float>(1,2) << " "  << twc.at<float>(1) << " " <<
             Rwc.at<float>(2,0) << " " << Rwc.at<float>(2,1)  << " " << Rwc.at<float>(2,2) << " "  << twc.at<float>(2) << endl;
    }
    f.close();
    cout << endl << "trajectory saved!" << endl;
}

int System::GetTrackingState()
{
    unique_lock<mutex> lock(mMutexState);//上锁
    return mTrackingState;
}

vector<MapPoint*> System::GetTrackedMapPoints()
{
    unique_lock<mutex> lock(mMutexState);//上锁
    return mTrackedMapPoints;
}

vector<cv::KeyPoint> System::GetTrackedKeyPointsUn()
{
    unique_lock<mutex> lock(mMutexState);//上锁
    return mTrackedKeyPointsUn;
}

} //namespace ORB_SLAM
