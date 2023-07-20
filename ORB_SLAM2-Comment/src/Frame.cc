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

#include "Frame.h"
#include "Converter.h"
#include "ORBmatcher.h"
#include <thread>

namespace ORB_SLAM2
{

long unsigned int Frame::nNextId=0;
bool Frame::mbInitialComputations=true;
float Frame::cx, Frame::cy, Frame::fx, Frame::fy, Frame::invfx, Frame::invfy;
float Frame::mnMinX, Frame::mnMinY, Frame::mnMaxX, Frame::mnMaxY;
float Frame::mfGridElementWidthInv, Frame::mfGridElementHeightInv;

Frame::Frame()
{}

//Copy Constructor
Frame::Frame(const Frame &frame)
    :mpORBvocabulary(frame.mpORBvocabulary), mpORBextractorLeft(frame.mpORBextractorLeft), mpORBextractorRight(frame.mpORBextractorRight),
     mTimeStamp(frame.mTimeStamp), mK(frame.mK.clone()), mDistCoef(frame.mDistCoef.clone()),
     mbf(frame.mbf), mb(frame.mb), mThDepth(frame.mThDepth), N(frame.N), mvKeys(frame.mvKeys),
     mvKeysRight(frame.mvKeysRight), mvKeysUn(frame.mvKeysUn),  mvuRight(frame.mvuRight),
     mvDepth(frame.mvDepth), mBowVec(frame.mBowVec), mFeatVec(frame.mFeatVec),
     mDescriptors(frame.mDescriptors.clone()), mDescriptorsRight(frame.mDescriptorsRight.clone()),
     mvpMapPoints(frame.mvpMapPoints), mvbOutlier(frame.mvbOutlier), mnId(frame.mnId),
     mpReferenceKF(frame.mpReferenceKF), mnScaleLevels(frame.mnScaleLevels),
     mfScaleFactor(frame.mfScaleFactor), mfLogScaleFactor(frame.mfLogScaleFactor),
     mvScaleFactors(frame.mvScaleFactors), mvInvScaleFactors(frame.mvInvScaleFactors),
     mvLevelSigma2(frame.mvLevelSigma2), mvInvLevelSigma2(frame.mvInvLevelSigma2)
{
    for(int i=0;i<FRAME_GRID_COLS;i++)
        for(int j=0; j<FRAME_GRID_ROWS; j++)
            mGrid[i][j]=frame.mGrid[i][j];

    if(!frame.mTcw.empty())
        SetPose(frame.mTcw);
}


Frame::Frame(const cv::Mat &imLeft, const cv::Mat &imRight, const double &timeStamp, ORBextractor* extractorLeft, ORBextractor* extractorRight, ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractorLeft),mpORBextractorRight(extractorRight), mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth),
     mpReferenceKF(static_cast<KeyFrame*>(NULL))//双目构建帧
{
    // Frame ID
	// Step 1 帧的ID 自增
    mnId=nNextId++;
	
	// Scale Level Info
	// Step 1 帧的ID 自增
	//获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
	//获取每层的缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
	//计算每层缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
	//获取各层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	//获取各层图像的缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	//获取sigma^2
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	//获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
	// Step 3 对这个单目图像进行提取特征点, 第一个参数0-左图， 1-右图
    thread threadLeft(&Frame::ExtractORB,this,0,imLeft);
    thread threadRight(&Frame::ExtractORB,this,1,imRight);
    threadLeft.join();
    threadRight.join();

	//求出特征点的个数
    N = mvKeys.size();
	//如果没有能够成功提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;

	// Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正 
    UndistortKeyPoints();

	//计算匹配子
    ComputeStereoMatches();
	
	// 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
	// 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N,false);


    // This is done only for the first Frame (or after a change in the calibration)
	// Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
		// 计算去畸变后图像的边界
        ComputeImageBounds(imLeft);

		// 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/(mnMaxX-mnMinX);
		// 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/(mnMaxY-mnMinY);

		//给类的静态成员变量复制
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		// 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

		//特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }

	//计算 basline
    mb = mbf/fx;

	// 将特征点分配到图像网格中 
    AssignFeaturesToGrid();
}

Frame::Frame(const cv::Mat &imGray, const cv::Mat &imDepth, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)//建立深度帧
{
    // Frame ID
	// Step 1 帧的ID 自增
    mnId=nNextId++;

    // Scale Level Info
	// Step 2 计算图像金字塔的参数 
	//获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
	//获取每层的缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
	//计算每层缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
	//获取各层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	//获取各层图像的缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	//获取sigma^2
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	//获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
	// Step 3 对这个单目图像进行提取特征点, 第一个参数0-左图， 1-右图，调用了orbextractor.cc程序
    ExtractORB(0,imGray);

	//求出特征点的个数
    N = mvKeys.size();

	//如果没有能够成功提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;

	// Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正 
    UndistortKeyPoints();

	//计算rgbd的深度
    ComputeStereoFromRGBD(imDepth);

	// 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
	// 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
	//  Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
		// 计算去畸变后图像的边界
        ComputeImageBounds(imGray);

		// 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
		// 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

		//给类的静态成员变量复制
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		// 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

		//特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }

	//计算 basline
    mb = mbf/fx;

	// 将特征点分配到图像网格中 
    AssignFeaturesToGrid();
}


Frame::Frame(const cv::Mat &imGray, const double &timeStamp, ORBextractor* extractor,ORBVocabulary* voc, cv::Mat &K, cv::Mat &distCoef, const float &bf, const float &thDepth)
    :mpORBvocabulary(voc),mpORBextractorLeft(extractor),mpORBextractorRight(static_cast<ORBextractor*>(NULL)),
     mTimeStamp(timeStamp), mK(K.clone()),mDistCoef(distCoef.clone()), mbf(bf), mThDepth(thDepth)//建立单目帧
{
    // Frame ID
	// Step 1 帧的ID 自增
    mnId=nNextId++;

    // Scale Level Info
	// Step 2 计算图像金字塔的参数 
	//获取图像金字塔的层数
    mnScaleLevels = mpORBextractorLeft->GetLevels();
	//获取每层的缩放因子
    mfScaleFactor = mpORBextractorLeft->GetScaleFactor();
	//计算每层缩放因子的自然对数
    mfLogScaleFactor = log(mfScaleFactor);
	//获取各层图像的缩放因子
    mvScaleFactors = mpORBextractorLeft->GetScaleFactors();
	//获取各层图像的缩放因子的倒数
    mvInvScaleFactors = mpORBextractorLeft->GetInverseScaleFactors();
	//获取sigma^2
    mvLevelSigma2 = mpORBextractorLeft->GetScaleSigmaSquares();
	//获取sigma^2的倒数
    mvInvLevelSigma2 = mpORBextractorLeft->GetInverseScaleSigmaSquares();

    // ORB extraction
	// Step 3 对这个单目图像进行提取特征点, 第一个参数0-左图， 1-右图
    ExtractORB(0,imGray);

	//求出特征点的个数
    N = mvKeys.size();

	//如果没有能够成功提取出特征点，那么就直接返回了
    if(mvKeys.empty())
        return;

	// Step 4 用OpenCV的矫正函数、内参对提取到的特征点进行矫正 
    UndistortKeyPoints();

    // Set no stereo information
	// 由于单目相机无法直接获得立体信息，所以这里要给右图像对应点和深度赋值-1表示没有相关信息
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

	// 初始化本帧的地图点
    mvpMapPoints = vector<MapPoint*>(N,static_cast<MapPoint*>(NULL));
	// 记录地图点是否为外点，初始化均为外点false
    mvbOutlier = vector<bool>(N,false);

    // This is done only for the first Frame (or after a change in the calibration)
	//  Step 5 计算去畸变后图像边界，将特征点分配到网格中。这个过程一般是在第一帧或者是相机标定参数发生变化之后进行
    if(mbInitialComputations)
    {
		// 计算去畸变后图像的边界
        ComputeImageBounds(imGray);

		// 表示一个图像像素相当于多少个图像网格列（宽）
        mfGridElementWidthInv=static_cast<float>(FRAME_GRID_COLS)/static_cast<float>(mnMaxX-mnMinX);
		// 表示一个图像像素相当于多少个图像网格行（高）
        mfGridElementHeightInv=static_cast<float>(FRAME_GRID_ROWS)/static_cast<float>(mnMaxY-mnMinY);

		//给类的静态成员变量复制
        fx = K.at<float>(0,0);
        fy = K.at<float>(1,1);
        cx = K.at<float>(0,2);
        cy = K.at<float>(1,2);
		// 猜测是因为这种除法计算需要的时间略长，所以这里直接存储了这个中间计算结果
        invfx = 1.0f/fx;
        invfy = 1.0f/fy;

		//特殊的初始化过程完成，标志复位
        mbInitialComputations=false;
    }

	//计算 basline
    mb = mbf/fx;
	
	// 将特征点分配到图像网格中 
    AssignFeaturesToGrid();
}

/*
最重要最复杂的函数是 Tracking::GrabImageMonocular() 中调用的 Track() 函数，
在他的前面的进行了 Frame 对象的创建，也是非常重要的，主要是因为他的构造函数之中做了图像金字塔，以及ORB特征提取等操作。
这是追踪过程中必不可少的预备工作。
*/

/**
 * @brief 将提取的ORB特征点分配到图像网格中
 *
 */
void Frame::AssignFeaturesToGrid()
{
	// 每个格子分配的特征点数，将图像分成格子，保证提取的特征点比较均匀
	// FRAME_GRID_ROWS 48
	// FRAME_GRID_COLS 64
	///这个向量中存储的是每个图像网格内特征点的id（左图）

	// Step 1  给存储特征点的网格数组 Frame::mGrid 预分配空间
	// ? 这里0.5 是为什么？节省空间？
	// FRAME_GRID_COLS = 64，FRAME_GRID_ROWS=48
    int nReserve = 0.5f*N/(FRAME_GRID_COLS*FRAME_GRID_ROWS);
	//开始对mGrid这个二维数组中的每一个vector元素遍历并预分配空间
    for(unsigned int i=0; i<FRAME_GRID_COLS;i++)
        for (unsigned int j=0; j<FRAME_GRID_ROWS;j++)
            mGrid[i][j].reserve(nReserve);

	// Step 2 遍历每个特征点，将每个特征点在mvKeysUn中的索引值放到对应的网格mGrid中
    for(int i=0;i<N;i++)
    {
		//从类的成员变量中获取已经去畸变后的特征点
        const cv::KeyPoint &kp = mvKeysUn[i];
		//存储某个特征点所在网格的网格坐标，nGridPosX范围：[0,FRAME_GRID_COLS], nGridPosY范围：[0,FRAME_GRID_ROWS]
        int nGridPosX, nGridPosY;
		// 计算某个特征点所在网格的网格坐标，如果找到特征点所在的网格坐标，记录在nGridPosX,nGridPosY里，返回true，没找到返回false
        if(PosInGrid(kp,nGridPosX,nGridPosY))
			//如果找到特征点所在网格坐标，将这个特征点的索引添加到对应网格的数组mGrid中
            mGrid[nGridPosX][nGridPosY].push_back(i);
    }
}

void Frame::ExtractORB(int flag, const cv::Mat &im)
{
	//判断左目还是右目
	if (flag == 0)
		// 左图的话就套使用左图指定的特征点提取器，并将提取结果保存到对应的变量中 
		// 这里使用了仿函数来完成，重载了括号运算符 ORBextractor::operator() 
		(*mpORBextractorLeft)(//核心关键，本质上调用的是 src\ORBextractor.cc 文件中的 ORBextractor::operator() 函数
			im,				//待提取特征点的图像
			cv::Mat(),		//掩摸图像, 实际没有用到
			mvKeys,			//输出变量，用于保存提取后的特征点
			mDescriptors);	//输出变量，用于保存特征点的描述子
	else
		// 右图的话就需要使用右图指定的特征点提取器，并将提取结果保存到对应的变量中 
		(*mpORBextractorRight)(im, cv::Mat(), mvKeysRight, mDescriptorsRight);//核心关键，本质上调用的是 src\ORBextractor.cc 文件中的 ORBextractor::operator() 函数
}

void Frame::SetPose(cv::Mat Tcw)//计算帧位姿
{
    mTcw = Tcw.clone();
    UpdatePoseMatrices();
}

void Frame::UpdatePoseMatrices()
{ 
    mRcw = mTcw.rowRange(0,3).colRange(0,3);
    mRwc = mRcw.t();
    mtcw = mTcw.rowRange(0,3).col(3);
    mOw = -mRcw.t()*mtcw;
}

/**
 * @brief 判断地图点是否在视野中
 * 步骤
 * Step 1 获得这个地图点的世界坐标，经过以下层层关卡的判断，通过的地图点才认为是在视野中
 * Step 2 关卡一：将这个地图点变换到当前帧的相机坐标系下，如果深度值为正才能继续下一步。
 * Step 3 关卡二：将地图点投影到当前帧的像素坐标，如果在图像有效范围内才能继续下一步。
 * Step 4 关卡三：计算地图点到相机中心的距离，如果在有效距离范围内才能继续下一步。
 * Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角，小于60°才能进入下一步。
 * Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
 * Step 7 记录计算得到的一些参数
 * @param[in] pMP                       当前地图点
 * @param[in] viewingCosLimit           当前相机指向地图点向量和地图点的平均观测方向夹角余弦阈值
 * @return true                         地图点合格，且在视野内
 * @return false                        地图点不合格，抛弃
 */
bool Frame::isInFrustum(MapPoint *pMP, float viewingCosLimit)
{
	// mbTrackInView是决定一个地图点是否进行重投影的标志
	// 这个标志的确定要经过多个函数的确定，isInFrustum()只是其中的一个验证关卡。这里默认设置为否
	pMP->mbTrackInView = false;

	// 3D in absolute coordinates
	// Step 1 获得这个地图点的世界坐标
	cv::Mat P = pMP->GetWorldPos();

	// 3D in camera coordinates
	// 根据当前帧(粗糙)位姿转化到当前相机坐标系下的三维点Pc
	const cv::Mat Pc = mRcw * P + mtcw;
	const float &PcX = Pc.at<float>(0);
	const float &PcY = Pc.at<float>(1);
	const float &PcZ = Pc.at<float>(2);

	// Check positive depth
	// Step 2 关卡一：将这个地图点变换到当前帧的相机坐标系下，如果深度值为正才能继续下一步。
	if (PcZ < 0.0f)
		return false;

	// Project in image and check it is not outside
	// Step 3 关卡二：将地图点投影到当前帧的像素坐标，如果在图像有效范围内才能继续下一步。
	const float invz = 1.0f / PcZ;
	const float u = fx * PcX*invz + cx;
	const float v = fy * PcY*invz + cy;

	// 判断是否在图像边界中，只要不在那么就说明无法在当前帧下进行重投影
	if (u<mnMinX || u>mnMaxX)
		return false;
	if (v<mnMinY || v>mnMaxY)
		return false;

	// Check distance is in the scale invariance region of the MapPoint
	// Step 4 关卡三：计算地图点到相机中心的距离，如果在有效距离范围内才能继续下一步。
	 // 得到认为的可靠距离范围:[0.8f*mfMinDistance, 1.2f*mfMaxDistance]
	const float maxDistance = pMP->GetMaxDistanceInvariance();
	const float minDistance = pMP->GetMinDistanceInvariance();

	// 得到当前地图点距离当前帧相机光心的距离,注意P，mOw都是在同一坐标系下才可以
	//  mOw：当前相机光心在世界坐标系下坐标
	const cv::Mat PO = P - mOw;
	//取模就得到了距离
	const float dist = cv::norm(PO);

	//如果不在有效范围内，认为投影不可靠
	if (dist<minDistance || dist>maxDistance)
		return false;

	// Check viewing angle
	// Step 5 关卡四：计算当前相机指向地图点向量和地图点的平均观测方向夹角，小于60°才能进入下一步。
	cv::Mat Pn = pMP->GetNormal();

	// 计算当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值，注意平均观测方向为单位向量
	const float viewCos = PO.dot(Pn) / dist;

	//夹角要在60°范围内，否则认为观测方向太偏了，重投影不可靠，返回false
	if (viewCos < viewingCosLimit)
		return false;

	// Predict scale in the image
	// Step 6 根据地图点到光心的距离来预测一个尺度（仿照特征点金字塔层级）
	const int nPredictedLevel = pMP->PredictScale(dist,		//这个点到光心的距离
		this);	//给出这个帧
// Step 7 记录计算得到的一些参数
// Data used by the tracking	
// 通过置位标记 MapPoint::mbTrackInView 来表示这个地图点要被投影 
	pMP->mbTrackInView = true;

	// 该地图点投影在当前图像（一般是左图）的像素横坐标
	pMP->mTrackProjX = u;

	// bf/z其实是视差，相减得到右图（如有）中对应点的横坐标
	pMP->mTrackProjXR = u - mbf * invz;

	// 该地图点投影在当前图像（一般是左图）的像素纵坐标									
	pMP->mTrackProjY = v;

	// 根据地图点到光心距离，预测的该地图点的尺度层级
	pMP->mnTrackScaleLevel = nPredictedLevel;

	// 保存当前相机指向地图点向量和地图点的平均观测方向夹角的余弦值
	pMP->mTrackViewCos = viewCos;

	//执行到这里说明这个地图点在相机的视野中并且进行重投影是可靠的，返回true
	return true;
}


/**
 * @brief 找到在 以x,y为中心,半径为r的圆形内且金字塔层级在[minLevel, maxLevel]的特征点
 *
 * @param[in] x                     特征点坐标x
 * @param[in] y                     特征点坐标y
 * @param[in] r                     搜索半径
 * @param[in] minLevel              最小金字塔层级
 * @param[in] maxLevel              最大金字塔层级
 * @return vector<size_t>           返回搜索到的候选匹配点id
 */
vector<size_t> Frame::GetFeaturesInArea(const float &x, const float  &y, const float  &r, const int minLevel, const int maxLevel) const
{
	// 存储搜索结果的vector
    vector<size_t> vIndices;
    vIndices.reserve(N);

	// Step 1 计算半径为r圆左右上下边界所在的网格列和行的id
   // 查找半径为r的圆左侧边界所在网格列坐标。这个地方有点绕，慢慢理解下：
   // (mnMaxX-mnMinX)/FRAME_GRID_COLS：表示列方向每个网格可以平均分得几个像素（肯定大于1）
   // mfGridElementWidthInv=FRAME_GRID_COLS/(mnMaxX-mnMinX) 是上面倒数，表示每个像素可以均分几个网格列（肯定小于1）
   // (x-mnMinX-r)，可以看做是从图像的左边界mnMinX到半径r的圆的左边界区域占的像素列数
   // 两者相乘，就是求出那个半径为r的圆的左侧边界在哪个网格列中
   // 保证nMinCellX 结果大于等于0
    const int nMinCellX = max(0,(int)floor((x-mnMinX-r)*mfGridElementWidthInv));

	// 如果最终求得的圆的左边界所在的网格列超过了设定了上限，那么就说明计算出错，找不到符合要求的特征点，返回空vector
    if(nMinCellX>=FRAME_GRID_COLS)
        return vIndices;

	// 计算圆所在的右边界网格列索引
    const int nMaxCellX = min((int)FRAME_GRID_COLS-1,(int)ceil((x-mnMinX+r)*mfGridElementWidthInv));
	// 如果计算出的圆右边界所在的网格不合法，说明该特征点不好，直接返回空vector
    if(nMaxCellX<0)
        return vIndices;

	//后面的操作也都是类似的，计算出这个圆上下边界所在的网格行的id
    const int nMinCellY = max(0,(int)floor((y-mnMinY-r)*mfGridElementHeightInv));
    if(nMinCellY>=FRAME_GRID_ROWS)
        return vIndices;

    const int nMaxCellY = min((int)FRAME_GRID_ROWS-1,(int)ceil((y-mnMinY+r)*mfGridElementHeightInv));
    if(nMaxCellY<0)
        return vIndices;

	// 检查需要搜索的图像金字塔层数范围是否符合要求
	//? 疑似bug。(minLevel>0) 后面条件 (maxLevel>=0)肯定成立
	//? 改为 const bool bCheckLevels = (minLevel>=0) || (maxLevel>=0);
    const bool bCheckLevels = (minLevel>0) || (maxLevel>=0);

	// Step 2 遍历圆形区域内的所有网格，寻找满足条件的候选特征点，并将其index放到输出里
    for(int ix = nMinCellX; ix<=nMaxCellX; ix++)
    {
        for(int iy = nMinCellY; iy<=nMaxCellY; iy++)
        {
			// 获取这个网格内的所有特征点在 Frame::mvKeysUn 中的索引
            const vector<size_t> vCell = mGrid[ix][iy];
			// 如果这个网格中没有特征点，那么跳过这个网格继续下一个
            if(vCell.empty())
                continue;
			
			// 如果这个网格中有特征点，那么遍历这个图像网格中所有的特征点
            for(size_t j=0, jend=vCell.size(); j<jend; j++)
            {
				// 根据索引先读取这个特征点 
                const cv::KeyPoint &kpUn = mvKeysUn[vCell[j]];
				// 保证给定的搜索金字塔层级范围合法
                if(bCheckLevels)
                {
					// cv::KeyPoint::octave中表示的是从金字塔的哪一层提取的数据
					// 保证特征点是在金字塔层级minLevel和maxLevel之间，不是的话跳过
                    if(kpUn.octave<minLevel)
                        continue;
                    if(maxLevel>=0)     //? 为何特意又强调？感觉多此一举
                        if(kpUn.octave>maxLevel)
                            continue;
                }

				// 通过检查，计算候选特征点到圆中心的距离，查看是否是在这个圆形区域之内
                const float distx = kpUn.pt.x-x;
                const float disty = kpUn.pt.y-y;

				// 如果x方向和y方向的距离都在指定的半径之内，存储其index为候选特征点
                if(fabs(distx)<r && fabs(disty)<r)
                    vIndices.push_back(vCell[j]);
            }
        }
    }

    return vIndices;//主要是选出候选的特征点
}

bool Frame::PosInGrid(const cv::KeyPoint &kp, int &posX, int &posY)
{
    posX = round((kp.pt.x-mnMinX)*mfGridElementWidthInv);
    posY = round((kp.pt.y-mnMinY)*mfGridElementHeightInv);

    //Keypoint's coordinates are undistorted, which could cause to go out of the image
    if(posX<0 || posX>=FRAME_GRID_COLS || posY<0 || posY>=FRAME_GRID_ROWS)
        return false;

    return true;
}

/**
 * @brief 计算当前帧特征点对应的词袋Bow，主要是mBowVec 和 mFeatVec
 *
 */
void Frame::ComputeBoW()
{
	// 判断是否以前已经计算过了，计算过了就跳过
    if(mBowVec.empty())
    {
		// 将描述子mDescriptors转换为DBOW要求的输入格式
        vector<cv::Mat> vCurrentDesc = Converter::toDescriptorVector(mDescriptors);
		// 将特征点的描述子转换成词袋向量mBowVec以及特征向量mFeatVec
		mpORBvocabulary->transform(vCurrentDesc,	//当前的描述子vector
			mBowVec,			//输出，词袋向量，记录的是单词的id及其对应权重TF-IDF值
			mFeatVec,		//输出，记录node id及其对应的图像 feature对应的索引
			4);				//4表示从叶节点向前数的层数
    }
}

/**
 * @brief 用内参对特征点去畸变，结果报存在mvKeysUn中
 *
 */

void Frame::UndistortKeyPoints()
{
	// Step 1 如果第一个畸变参数为0，不需要矫正。第一个畸变参数k1是最重要的，一般不为0，为0的话，说明畸变参数都是0
	//变量mDistCoef中存储了opencv指定格式的去畸变参数，格式为：(k1,k2,p1,p2,k3)
    if(mDistCoef.at<float>(0)==0.0)
    {
        mvKeysUn=mvKeys;
        return;
    }

    // Fill matrix with points
	// Step 2 如果畸变参数不为0，用OpenCV函数进行畸变矫正
	// N为提取的特征点数量，为满足OpenCV函数输入要求，将N个特征点保存在N*2的矩阵中
    cv::Mat mat(N,2,CV_32F);
	//遍历每个特征点，并将它们的坐标保存到矩阵中
    for(int i=0; i<N; i++)
    {
		//然后将这个特征点的横纵坐标分别保存
        mat.at<float>(i,0)=mvKeys[i].pt.x;
        mat.at<float>(i,1)=mvKeys[i].pt.y;
    }

    // Undistort points
	// 函数reshape(int cn,int rows=0) 其中cn为更改后的通道数，rows=0表示这个行将保持原来的参数不变
	//为了能够直接调用opencv的函数来去畸变，需要先将矩阵调整为2通道（对应坐标x,y）
    mat=mat.reshape(2);
	cv::undistortPoints(//调用OpenCV的undistortPoints函数来进行矫正
		mat,				//输入的特征点坐标
		mat,				//输出的校正后的特征点坐标覆盖原矩阵
		mK,					//相机的内参数矩阵
		mDistCoef,			//相机畸变参数矩阵
		cv::Mat(),			//一个空矩阵，对应为函数原型中的R
		mK); 				//新内参数矩阵，对应为函数原型中的P
	//调整回只有一个通道，回归我们正常的处理方式
    mat=mat.reshape(1);

    // Fill undistorted keypoint vector
	// Step 3 存储校正后的特征点
    mvKeysUn.resize(N);
	//遍历每一个特征点
    for(int i=0; i<N; i++)
    {
		//根据索引获取这个特征点
		//注意之所以这样做而不是直接重新声明一个特征点对象的目的是，能够得到源特征点对象的其他属性
        cv::KeyPoint kp = mvKeys[i];
		//读取矫正坐标后覆盖老坐标
        kp.pt.x=mat.at<float>(i,0);
        kp.pt.y=mat.at<float>(i,1);
        mvKeysUn[i]=kp;
    }
}

void Frame::ComputeImageBounds(const cv::Mat &imLeft)//计算畸变后的图像边界
{
    if(mDistCoef.at<float>(0)!=0.0)
    {
        cv::Mat mat(4,2,CV_32F);
        mat.at<float>(0,0)=0.0; mat.at<float>(0,1)=0.0;
        mat.at<float>(1,0)=imLeft.cols; mat.at<float>(1,1)=0.0;
        mat.at<float>(2,0)=0.0; mat.at<float>(2,1)=imLeft.rows;
        mat.at<float>(3,0)=imLeft.cols; mat.at<float>(3,1)=imLeft.rows;

        // Undistort corners
        mat=mat.reshape(2);
        cv::undistortPoints(mat,mat,mK,mDistCoef,cv::Mat(),mK);
        mat=mat.reshape(1);

        mnMinX = min(mat.at<float>(0,0),mat.at<float>(2,0));
        mnMaxX = max(mat.at<float>(1,0),mat.at<float>(3,0));
        mnMinY = min(mat.at<float>(0,1),mat.at<float>(1,1));
        mnMaxY = max(mat.at<float>(2,1),mat.at<float>(3,1));

    }
    else
    {
        mnMinX = 0.0f;
        mnMaxX = imLeft.cols;
        mnMinY = 0.0f;
        mnMaxY = imLeft.rows;
    }
}

/*
 * 双目匹配函数
 *
 * 为左图的每一个特征点在右图中找到匹配点 \n
 * 根据基线(有冗余范围)上描述子距离找到匹配, 再进行SAD精确定位 \n ‘
 * 这里所说的SAD是一种双目立体视觉匹配算法，可参考[https://blog.csdn.net/u012507022/article/details/51446891]
 * 最后对所有SAD的值进行排序, 剔除SAD值较大的匹配对，然后利用抛物线拟合得到亚像素精度的匹配 \n
 * 这里所谓的亚像素精度，就是使用这个拟合得到一个小于一个单位像素的修正量，这样可以取得更好的估计结果，计算出来的点的深度也就越准确
 * 匹配成功后会更新 mvuRight(ur) 和 mvDepth(Z)
 */
void Frame::ComputeStereoMatches()
{
	// 为匹配结果预先分配内存，数据类型为float型
    // mvuRight存储右图匹配点索引
    // mvDepth存储特征点的深度信息
    mvuRight = vector<float>(N,-1.0f);
    mvDepth = vector<float>(N,-1.0f);

	// orb特征相似度阈值  -> mean ～= (max  + min) / 2
    const int thOrbDist = (ORBmatcher::TH_HIGH+ORBmatcher::TH_LOW)/2;

	// 金字塔底层（0层）图像高 nRows
    const int nRows = mpORBextractorLeft->mvImagePyramid[0].rows;

    //Assign keypoints to row table
	// 二维vector存储每一行的orb特征点的列坐标，为什么是vector，因为每一行的特征点有可能不一样，例如
	// vRowIndices[0] = [1，2，5，8, 11]   第1行有5个特征点,他们的列号（即x坐标）分别是1,2,5,8,11
	// vRowIndices[1] = [2，6，7，9, 13, 17, 20]  第2行有7个特征点.etc
    vector<vector<size_t> > vRowIndices(nRows,vector<size_t>());

    for(int i=0; i<nRows; i++)
        vRowIndices[i].reserve(200); //储存每一行orb特征点的列坐标

	// 右图特征点数量，N表示数量 r表示右图，且不能被修改
    const int Nr = mvKeysRight.size();

	// Step 1. 行特征点统计。 考虑用图像金字塔尺度作为偏移，左图中对应右图的一个特征点可能存在于多行，而非唯一的一行
    for(int iR=0; iR<Nr; iR++)
    {
		// 获取特征点ir的y坐标，即行号
        const cv::KeyPoint &kp = mvKeysRight[iR];
        const float &kpY = kp.pt.y;

		// 计算特征点ir在行方向上，可能的偏移范围r，即可能的行号为[kpY + r, kpY -r]
		// 2 表示在全尺寸(scale = 1)的情况下，假设有2个像素的偏移，随着尺度变化，r也跟着变化
        const float r = 2.0f*mvScaleFactors[mvKeysRight[iR].octave];
        const int maxr = ceil(kpY+r);
        const int minr = floor(kpY-r);

		// 将特征点ir保证在可能的行号中
        for(int yi=minr;yi<=maxr;yi++)
            vRowIndices[yi].push_back(iR);
    }

	// 下面是 粗匹配 + 精匹配的过程
    // 对于立体矫正后的两张图，在列方向(x)存在最大视差maxd和最小视差mind
    // 也即是左图中任何一点p，在右图上的匹配点的范围为应该是[p - maxd, p - mind], 而不需要遍历每一行所有的像素
    // maxd = baseline * length_focal / minZ
    // mind = baseline * length_focal / maxZ

    // Set limits for search
    const float minZ = mb;
    const float minD = 0;               //最小视差为0，对应无穷远
    const float maxD = mbf/minZ;        //最大视差对应的距离是相机的基线

    // For each left keypoint search a match in the right image
	// 保存sad块匹配相似度和左图特征点索引
    vector<pair<int, int> > vDistIdx;
    vDistIdx.reserve(N);

	// 为左图每一个特征点il，在右图搜索最相似的特征点ir
    for(int iL=0; iL<N; iL++)
    {
        const cv::KeyPoint &kpL = mvKeys[iL];
        const int &levelL = kpL.octave;
        const float &vL = kpL.pt.y;
        const float &uL = kpL.pt.x;

		// 获取左图特征点il所在行，以及在右图对应行中可能的匹配点
        const vector<size_t> &vCandidates = vRowIndices[vL];

        if(vCandidates.empty())
            continue;

		// 计算理论上的最佳搜索范围
        const float minU = uL-maxD;
        const float maxU = uL-minD;

		// 最大搜索范围小于0，说明无匹配点
        if(maxU<0)
            continue;

		// 初始化最佳相似度，用最大相似度，以及最佳匹配点索引
        int bestDist = ORBmatcher::TH_HIGH;
        size_t bestIdxR = 0;

        const cv::Mat &dL = mDescriptors.row(iL);

		// Step 2. 粗配准。左图特征点il与右图中的可能的匹配点进行逐个比较,得到最相似匹配点的描述子距离和索引
        // Compare descriptor to right keypoints
        for(size_t iC=0; iC<vCandidates.size(); iC++)
        {
            const size_t iR = vCandidates[iC];
            const cv::KeyPoint &kpR = mvKeysRight[iR];
			
			// 左图特征点il与待匹配点ic的空间尺度差超过2，放弃
            if(kpR.octave<levelL-1 || kpR.octave>levelL+1)
                continue;

			// 使用列坐标(x)进行匹配，和stereomatch一样
            const float &uR = kpR.pt.x;

			// 超出理论搜索范围[minU, maxU]，可能是误匹配，放弃
            if(uR>=minU && uR<=maxU)
            {
				// 计算匹配点il和待匹配点ic的相似度dist
                const cv::Mat &dR = mDescriptorsRight.row(iR);
                const int dist = ORBmatcher::DescriptorDistance(dL,dR);

				//统计最小相似度及其对应的列坐标(x)
                if(dist<bestDist)
                {
                    bestDist = dist;
                    bestIdxR = iR;
                }
            }
        }

        // Subpixel match by correlation
		// Step 3. 图像块滑动窗口用SAD(Sum of absolute differences，差的绝对和)实现精确匹配. 
        if(bestDist<thOrbDist)
        {
            // coordinates in image pyramid at keypoint scale
			// 如果刚才匹配过程中的最佳描述子距离小于给定的阈值
			// 计算右图特征点x坐标和对应的金字塔尺度
            const float uR0 = mvKeysRight[bestIdxR].pt.x;//uR0是什么？mvKeysRight是右图特征点，bestIdxR是最小相似度对应的列坐标的索引？这里表示的是特征点的x坐标
            const float scaleFactor = mvInvScaleFactors[kpL.octave];
			// 尺度缩放后的左右图特征点坐标
            const float scaleduL = round(kpL.pt.x*scaleFactor);
            const float scaledvL = round(kpL.pt.y*scaleFactor);
            const float scaleduR0 = round(uR0*scaleFactor);//尺度缩放后右图特征点的x坐标

            // sliding window search
			// 滑动窗口搜索, 类似模版卷积或滤波
			// w表示sad相似度的窗口半径
            const int w = 5;
			// 提取左图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像块patch
            cv::Mat IL = mpORBextractorLeft->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduL-w,scaleduL+w+1);
            IL.convertTo(IL,CV_32F);
			// 图像块均值归一化，降低亮度变化对相似度计算的影响
            IL = IL - IL.at<float>(w,w) *cv::Mat::ones(IL.rows,IL.cols,CV_32F);

			//初始化最佳相似度
            int bestDist = INT_MAX;
			// 通过滑动窗口搜索优化，得到的列坐标偏移量
            int bestincR = 0;
			//滑动窗口的滑动范围为（-L, L）
            const int L = 5;
			// 初始化存储图像块相似度
            vector<float> vDists;
            vDists.resize(2*L+1);

			// 计算滑动窗口滑动范围的边界，因为是块匹配，还要算上图像块的尺寸
			// 列方向起点 iniu = r0 - 最大窗口滑动范围 - 图像块尺寸
			// 列方向终点 eniu = r0 + 最大窗口滑动范围 + 图像块尺寸 + 1
			// 此次 + 1 和下面的提取图像块是列坐标+1是一样的，保证提取的图像块的宽是2 * w + 1
			// ! 源码： const float iniu = scaleduR0+L-w; 错误
			// scaleduR0：右图特征点x坐标
            const float iniu = scaleduR0+L-w;//这里是减法，-L-w
            const float endu = scaleduR0+L+w+1;
			// 判断搜索是否越界
            if(iniu<0 || endu >= mpORBextractorRight->mvImagePyramid[kpL.octave].cols)
                continue;

			// 在搜索范围内从左到右滑动，并计算图像块相似度
            for(int incR=-L; incR<=+L; incR++)
            {
				// 提取右图中，以特征点(scaleduL,scaledvL)为中心, 半径为w的图像快patch
                cv::Mat IR = mpORBextractorRight->mvImagePyramid[kpL.octave].rowRange(scaledvL-w,scaledvL+w+1).colRange(scaleduR0+incR-w,scaleduR0+incR+w+1);
                IR.convertTo(IR,CV_32F);
				// 图像块均值归一化，降低亮度变化对相似度计算的影响
                IR = IR - IR.at<float>(w,w) *cv::Mat::ones(IR.rows,IR.cols,CV_32F);

				// sad 计算，值越小越相似
                float dist = cv::norm(IL,IR,cv::NORM_L1);
				// 统计最小sad和偏移量
                if(dist<bestDist)
                {
                    bestDist =  dist;
                    bestincR = incR;
                }
				//L+incR 为refine后的匹配点列坐标(x)
                vDists[L+incR] = dist;
            }

			// 搜索窗口越界判断
            if(bestincR==-L || bestincR==L)
                continue;

            // Sub-pixel match (Parabola fitting)
			// Step 4. 亚像素插值, 使用最佳匹配点及其左右相邻点构成抛物线来得到最小sad的亚像素坐标
			// 使用3点拟合抛物线的方式，用极小值代替之前计算的最优是差值
			//    \                 / <- 由视差为14，15，16的相似度拟合的抛物线
			//      .             .(16)
			//         .14     .(15) <- int/uchar最佳视差值
			//              . 
			//           （14.5）<- 真实的视差值
			//   deltaR = 15.5 - 16 = -0.5
			// 公式参考opencv sgbm源码中的亚像素插值公式
			// 或论文<<On Building an Accurate Stereo Matching System on Graphics Hardware>> 公式7
            const float dist1 = vDists[L+bestincR-1];
            const float dist2 = vDists[L+bestincR];
            const float dist3 = vDists[L+bestincR+1];

            const float deltaR = (dist1-dist3)/(2.0f*(dist1+dist3-2.0f*dist2));

			// 亚像素精度的修正量应该是在[-1,1]之间，否则就是误匹配
            if(deltaR<-1 || deltaR>1)
                continue;

            // Re-scaled coordinate
			// 根据亚像素精度偏移量delta调整最佳匹配索引
            float bestuR = mvScaleFactors[kpL.octave]*((float)scaleduR0+(float)bestincR+deltaR);

            float disparity = (uL-bestuR);

            if(disparity>=minD && disparity<maxD)
            {
				// 如果存在负视差，则约束为0.01
                if(disparity<=0)
                {
                    disparity=0.01;
                    bestuR = uL-0.01;
                }
				// 根据视差值计算深度信息
				// 保存最相似点的列坐标(x)信息
				// 保存归一化sad最小相似度
				// Step 5. 最优视差值/深度选择.
                mvDepth[iL]=mbf/disparity;
                mvuRight[iL] = bestuR;
                vDistIdx.push_back(pair<int,int>(bestDist,iL));
            }
        }
    }

	// Step 6. 删除离群点(outliers)
	// 块匹配相似度阈值判断，归一化sad最小，并不代表就一定是匹配的，比如光照变化、弱纹理、无纹理等同样会造成误匹配
	// 误匹配判断条件  norm_sad > 1.5 * 1.4 * median
    sort(vDistIdx.begin(),vDistIdx.end());
    const float median = vDistIdx[vDistIdx.size()/2].first;
    const float thDist = 1.5f*1.4f*median;

    for(int i=vDistIdx.size()-1;i>=0;i--)
    {
        if(vDistIdx[i].first<thDist)
            break;
        else
        {
			// 误匹配点置为-1，和初始化时保持一致，作为error code
            mvuRight[vDistIdx[i].second]=-1;
            mvDepth[vDistIdx[i].second]=-1;
        }
    }
}


void Frame::ComputeStereoFromRGBD(const cv::Mat &imDepth)
{
    mvuRight = vector<float>(N,-1);
    mvDepth = vector<float>(N,-1);

    for(int i=0; i<N; i++)
    {
        const cv::KeyPoint &kp = mvKeys[i];
        const cv::KeyPoint &kpU = mvKeysUn[i];

        const float &v = kp.pt.y;
        const float &u = kp.pt.x;

        const float d = imDepth.at<float>(v,u);

        if(d>0)
        {
            mvDepth[i] = d;
            mvuRight[i] = kpU.pt.x-mbf/d;
        }
    }
}

cv::Mat Frame::UnprojectStereo(const int &i)
{
    const float z = mvDepth[i];
    if(z>0)
    {
        const float u = mvKeysUn[i].pt.x;
        const float v = mvKeysUn[i].pt.y;
        const float x = (u-cx)*z*invfx;
        const float y = (v-cy)*z*invfy;
        cv::Mat x3Dc = (cv::Mat_<float>(3,1) << x, y, z);
        return mRwc*x3Dc+mOw;
    }
    else
        return cv::Mat();
}

} //namespace ORB_SLAM
