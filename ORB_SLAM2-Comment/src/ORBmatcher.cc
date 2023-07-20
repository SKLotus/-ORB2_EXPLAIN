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

#include "ORBmatcher.h"

#include<limits.h>

#include<opencv2/core/core.hpp>
#include<opencv2/features2d/features2d.hpp>

#include "Thirdparty/DBoW2/DBoW2/FeatureVector.h"

#include<stdint-gcc.h>

using namespace std;

namespace ORB_SLAM2
{

const int ORBmatcher::TH_HIGH = 100;
const int ORBmatcher::TH_LOW = 50;
const int ORBmatcher::HISTO_LENGTH = 30;

ORBmatcher::ORBmatcher(float nnratio, bool checkOri): mfNNratio(nnratio), mbCheckOrientation(checkOri)
{
}

int ORBmatcher::SearchByProjection(Frame &F, const vector<MapPoint*> &vpMapPoints, const float th)
{
    int nmatches=0;

    const bool bFactor = th!=1.0;

    for(size_t iMP=0; iMP<vpMapPoints.size(); iMP++)
    {
        MapPoint* pMP = vpMapPoints[iMP];
        if(!pMP->mbTrackInView)
            continue;

        if(pMP->isBad())
            continue;

        const int &nPredictedLevel = pMP->mnTrackScaleLevel;

        // The size of the window will depend on the viewing direction
        float r = RadiusByViewingCos(pMP->mTrackViewCos);

        if(bFactor)
            r*=th;

        const vector<size_t> vIndices =
                F.GetFeaturesInArea(pMP->mTrackProjX,pMP->mTrackProjY,r*F.mvScaleFactors[nPredictedLevel],nPredictedLevel-1,nPredictedLevel);

        if(vIndices.empty())
            continue;

        const cv::Mat MPdescriptor = pMP->GetDescriptor();

        int bestDist=256;
        int bestLevel= -1;
        int bestDist2=256;
        int bestLevel2 = -1;
        int bestIdx =-1 ;

        // Get best and second matches with near keypoints
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            if(F.mvpMapPoints[idx])
                if(F.mvpMapPoints[idx]->Observations()>0)
                    continue;

            if(F.mvuRight[idx]>0)
            {
                const float er = fabs(pMP->mTrackProjXR-F.mvuRight[idx]);
                if(er>r*F.mvScaleFactors[nPredictedLevel])
                    continue;
            }

            const cv::Mat &d = F.mDescriptors.row(idx);

            const int dist = DescriptorDistance(MPdescriptor,d);

            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestLevel2 = bestLevel;
                bestLevel = F.mvKeysUn[idx].octave;
                bestIdx=idx;
            }
            else if(dist<bestDist2)
            {
                bestLevel2 = F.mvKeysUn[idx].octave;
                bestDist2=dist;
            }
        }

        // Apply ratio to second match (only if best and second are in the same scale level)
        if(bestDist<=TH_HIGH)
        {
            if(bestLevel==bestLevel2 && bestDist>mfNNratio*bestDist2)
                continue;

            F.mvpMapPoints[bestIdx]=pMP;
            nmatches++;
        }
    }

    return nmatches;
}

float ORBmatcher::RadiusByViewingCos(const float &viewCos)
{
    if(viewCos>0.998)
        return 2.5;
    else
        return 4.0;
}


/*
@brief 用基础矩阵检查极线距离是否符合要求

@param[in] kp1  KF1中的特征点
@param[in] kp1  KF2中的特征点
@param[in] F12  从KF2到KF1的基础矩阵
@param[in] pKF2 关键帧KF2
@return true
@return false

*/
bool ORBmatcher::CheckDistEpipolarLine(const cv::KeyPoint &kp1,const cv::KeyPoint &kp2,const cv::Mat &F12,const KeyFrame* pKF2)
{
    // Epipolar line in second image l = x1'F12 = [a b c]
	// step1 求出kp1在kp2上对应的极线
    const float a = kp1.pt.x*F12.at<float>(0,0)+kp1.pt.y*F12.at<float>(1,0)+F12.at<float>(2,0);
    const float b = kp1.pt.x*F12.at<float>(0,1)+kp1.pt.y*F12.at<float>(1,1)+F12.at<float>(2,1);
    const float c = kp1.pt.x*F12.at<float>(0,2)+kp1.pt.y*F12.at<float>(1,2)+F12.at<float>(2,2);

	//step2 计算kp2特征点到极线L2的距离
	//极线L2：ax+by+c=0
	//（u，v）到L2的距离为：|au+bv+c|/sqrt(a^2+b^2)
    const float num = a*kp2.pt.x+b*kp2.pt.y+c;

    const float den = a*a+b*b;

	//距离无穷大
    if(den==0)
        return false;

	//距离的平方
    const float dsqr = num*num/den;

	//step3 判断误差是否满足条件，尺度越大，误差范围应该越大
	//金字塔最底层一个像素就占一个像素，在倒数第二层，一个像素等于最底层的1.2个像素（假设金字塔尺度为1.2）
	//3.84是自由度为1时，服从高斯分布的一个平方项（也就是这里的误差），小于一个像素，这件事情发生概率超过95%的概率，卡方分布
    return dsqr<3.84*pKF2->mvLevelSigma2[kp2.octave];
}

/*
 * @brief 通过词袋，对关键帧的特征点进行跟踪
 * 步骤
 * Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
 * Step 2：遍历KF中属于该node的特征点
 * Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
 * Step 4：根据阈值 和 角度投票剔除误匹配
 * Step 5：根据方向剔除误匹配的点
 * @param  pKF               关键帧
 * @param  F                 当前普通帧
 * @param  vpMapPointMatches F中地图点对应的匹配，NULL表示未匹配
 * @return                   成功匹配的数量
 */
int ORBmatcher::SearchByBoW(KeyFrame* pKF,Frame &F, vector<MapPoint*> &vpMapPointMatches)
{
	// 获取该关键帧的地图点
    const vector<MapPoint*> vpMapPointsKF = pKF->GetMapPointMatches();

	// 和普通帧F特征点的索引一致
    vpMapPointMatches = vector<MapPoint*>(F.N,static_cast<MapPoint*>(NULL));

	// 取出关键帧的词袋特征向量
    const DBoW2::FeatureVector &vFeatVecKF = pKF->mFeatVec;

    int nmatches=0;

	// 特征点角度旋转差统计用的直方图
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

	// 将0~360的数转换到0~HISTO_LENGTH的系数
	// !原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码  
    const float factor = 1.0f/HISTO_LENGTH;

    // We perform the matching over ORB that belong to the same vocabulary node (at a certain level)
	// 将属于同一节点(特定层)的ORB特征进行匹配
    DBoW2::FeatureVector::const_iterator KFit = vFeatVecKF.begin();
    DBoW2::FeatureVector::const_iterator Fit = F.mFeatVec.begin();
    DBoW2::FeatureVector::const_iterator KFend = vFeatVecKF.end();
    DBoW2::FeatureVector::const_iterator Fend = F.mFeatVec.end();

    while(KFit != KFend && Fit != Fend)
    {
		// Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
		// first 元素就是node id，遍历
        if(KFit->first == Fit->first)
        {
			// second 是该node内存储的feature index
            const vector<unsigned int> vIndicesKF = KFit->second;
            const vector<unsigned int> vIndicesF = Fit->second;

			// Step 2：遍历KF中属于该node的特征点
            for(size_t iKF=0; iKF<vIndicesKF.size(); iKF++)
            {
				// 关键帧该节点中特征点的索引
                const unsigned int realIdxKF = vIndicesKF[iKF];

				// 取出KF中该特征对应的地图点
                MapPoint* pMP = vpMapPointsKF[realIdxKF];

                if(!pMP)
                    continue;

                if(pMP->isBad())
                    continue;                

                const cv::Mat &dKF= pKF->mDescriptors.row(realIdxKF);// 取出KF中该特征对应的描述子

                int bestDist1=256;// 最好的距离（最小距离）
                int bestIdxF =-1 ;
                int bestDist2=256;// 次好距离（倒数第二小距离）

				// Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
                for(size_t iF=0; iF<vIndicesF.size(); iF++)
                {
					// 和上面for循环重名了,这里的realIdxF是指普通帧该节点中特征点的索引
                    const unsigned int realIdxF = vIndicesF[iF];

					// 如果地图点存在，说明这个点已经被匹配过了，不再匹配，加快速度
                    if(vpMapPointMatches[realIdxF])
                        continue;

                    const cv::Mat &dF = F.mDescriptors.row(realIdxF);// 取出F中该特征对应的描述子
					 // 计算描述子的距离
                    const int dist =  DescriptorDistance(dKF,dF);

					// 遍历，记录最佳距离、最佳距离对应的索引、次佳距离等
					// 如果 dist < bestDist1 < bestDist2，更新bestDist1 bestDist2
                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdxF=realIdxF;
                    }
					// 如果bestDist1 < dist < bestDist2，更新bestDist2
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

				// Step 4：根据阈值 和 角度投票剔除误匹配
			   // Step 4.1：第一关筛选：匹配距离必须小于设定阈值
                if(bestDist1<=TH_LOW)
                {
					// Step 4.2：第二关筛选：最佳匹配比次佳匹配明显要好，那么最佳匹配才真正靠谱
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
						// Step 4.3：记录成功匹配特征点的对应的地图点(来自关键帧)
                        vpMapPointMatches[bestIdxF]=pMP;

						// 这里的realIdxKF是当前遍历到的关键帧的特征点id
                        const cv::KeyPoint &kp = pKF->mvKeysUn[realIdxKF];

						// Step 4.4：计算匹配点旋转角度差所在的直方图
                        if(mbCheckOrientation)
                        {
							// angle：每个特征点在提取描述子时的旋转主方向角度，如果图像旋转了，这个角度将发生改变
							// 所有的特征点的角度变化应该是一致的，通过直方图统计得到最准确的角度变化值
                            float rot = kp.angle-F.mvKeys[bestIdxF].angle;// 该特征点的角度变化值
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);// 将rot分配到bin组, 四舍五入, 其实就是离散到对应的直方图组中
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(bestIdxF); // 直方图统计
                        }
                        nmatches++;
                    }
                }

            }

            KFit++;
            Fit++;
        }
        else if(KFit->first < Fit->first)
        {
			// 对齐
            KFit = vFeatVecKF.lower_bound(Fit->first);
        }
        else
        {
			// 对齐
            Fit = F.mFeatVec.lower_bound(KFit->first);
        }
    }


	// Step 5 根据方向剔除误匹配的点
    if(mbCheckOrientation)
    {
		// index
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

		// 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
			// 如果特征点的旋转角度变化量属于这三个组，则保留
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
			// 剔除掉不在前三的匹配对，因为他们不符合“主流旋转方向” 
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMapPointMatches[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

int ORBmatcher::SearchByProjection(KeyFrame* pKF, cv::Mat Scw, const vector<MapPoint*> &vpPoints, vector<MapPoint*> &vpMatched, int th)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    set<MapPoint*> spAlreadyFound(vpMatched.begin(), vpMatched.end());
    spAlreadyFound.erase(static_cast<MapPoint*>(NULL));

    int nmatches=0;

    // For each Candidate MapPoint Project and Match
    for(int iMP=0, iendMP=vpPoints.size(); iMP<iendMP; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0)
            continue;

        // Project into Image
        const float invz = 1/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale invariance region of the point
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist = cv::norm(PO);

        if(dist<minDistance || dist>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist)
            continue;

        int nPredictedLevel = pMP->PredictScale(dist,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = 256;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;
            if(vpMatched[idx])
                continue;

            const int &kpLevel= pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_LOW)
        {
            vpMatched[bestIdx]=pMP;
            nmatches++;
        }

    }

    return nmatches;
}

/**
 * @brief 单目初始化中用于参考帧和当前帧的特征点匹配
 * 步骤
 * Step 1 构建旋转直方图
 * Step 2 在半径窗口内搜索当前帧F2中所有的候选匹配特征点
 * Step 3 遍历搜索搜索窗口中的所有潜在的匹配候选点，找到最优的和次优的
 * Step 4 对最优次优结果进行检查，满足阈值、最优/次优比例，删除重复匹配
 * Step 5 计算匹配点旋转角度差所在的直方图
 * Step 6 筛除旋转直方图中“非主流”部分
 * Step 7 将最后通过筛选的匹配好的特征点保存
 * @param[in] F1                        初始化参考帧
 * @param[in] F2                        当前帧
 * @param[in & out] vbPrevMatched       本来存储的是参考帧的所有特征点坐标，该函数更新为匹配好的当前帧的特征点坐标
 * @param[in & out] vnMatches12         保存参考帧F1中特征点是否匹配上，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
 * @param[in] windowSize                搜索窗口
 * @return int                          返回成功匹配的特征点数目
 */
int ORBmatcher::SearchForInitialization(Frame &F1, Frame &F2, vector<cv::Point2f> &vbPrevMatched, vector<int> &vnMatches12, int windowSize)
{
    int nmatches=0;
	// F1中特征点和F2中匹配关系，注意是按照F1特征点数目分配空间
    vnMatches12 = vector<int>(F1.mvKeysUn.size(),-1);

	// Step 1 构建旋转直方图，HISTO_LENGTH = 30
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
	// 每个bin里预分配500个，因为使用的是vector不够的话可以自动扩展容量
        rotHist[i].reserve(500);
	//! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为右边代码    const float factor = HISTO_LENGTH/360.0f;
    const float factor = 1.0f/HISTO_LENGTH;

	// 匹配点对距离，注意是按照F2特征点数目分配空间
    vector<int> vMatchedDistance(F2.mvKeysUn.size(),INT_MAX);
	// 从帧2到帧1的反向匹配，注意是按照F2特征点数目分配空间
    vector<int> vnMatches21(F2.mvKeysUn.size(),-1);

	// 遍历帧1中的所有特征点
    for(size_t i1=0, iend1=F1.mvKeysUn.size(); i1<iend1; i1++)
    {
        cv::KeyPoint kp1 = F1.mvKeysUn[i1];
        int level1 = kp1.octave;
		// 只使用原始图像上提取的特征点
        if(level1>0)
            continue;

		// Step 2 在半径窗口内搜索当前帧F2中所有的候选匹配特征点 
		// vbPrevMatched 输入的是参考帧 F1的特征点
		// windowSize = 100，输入最大最小金字塔层级 均为0
        vector<size_t> vIndices2 = F2.GetFeaturesInArea(vbPrevMatched[i1].x,vbPrevMatched[i1].y, windowSize,level1,level1);

		// 没有候选特征点，跳过
        if(vIndices2.empty())
            continue;

		// 取出参考帧F1中当前遍历特征点对应的描述子
        cv::Mat d1 = F1.mDescriptors.row(i1);

        int bestDist = INT_MAX;           //最佳描述子匹配距离，越小越好
        int bestDist2 = INT_MAX;          //次佳描述子匹配距离
        int bestIdx2 = -1;                //最佳候选特征点在F2中的index

		// Step 3 遍历搜索搜索窗口中的所有潜在的匹配候选点，找到最优的和次优的
        for(vector<size_t>::iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
        {
            size_t i2 = *vit;
			// 取出候选特征点对应的描述子
            cv::Mat d2 = F2.mDescriptors.row(i2);
			// 计算两个特征点描述子距离
            int dist = DescriptorDistance(d1,d2);

            if(vMatchedDistance[i2]<=dist)
                continue;
			// 如果当前匹配距离更小，更新最佳次佳距离
            if(dist<bestDist)
            {
                bestDist2=bestDist;
                bestDist=dist;
                bestIdx2=i2;
            }
            else if(dist<bestDist2)
            {
                bestDist2=dist;
            }
        }

		// Step 4 对最优次优结果进行检查，满足阈值、最优/次优比例，删除重复匹配
		// 即使算出了最佳描述子匹配距离，也不一定保证配对成功。要小于设定阈值
        if(bestDist<=TH_LOW)
        {
			// 最佳距离比次佳距离要小于设定的比例，这样特征点辨识度更高
            if(bestDist<(float)bestDist2*mfNNratio)
            {
				// 如果找到的候选特征点对应F1中特征点已经匹配过了，说明发生了重复匹配，将原来的匹配也删掉
                if(vnMatches21[bestIdx2]>=0)
                {
                    vnMatches12[vnMatches21[bestIdx2]]=-1;
                    nmatches--;
                }
				// 次优的匹配关系，双向建立
				// vnMatches12保存参考帧F1和F2匹配关系，index保存是F1对应特征点索引，值保存的是匹配好的F2特征点索引
                vnMatches12[i1]=bestIdx2;
                vnMatches21[bestIdx2]=i1;
                vMatchedDistance[bestIdx2]=bestDist;
                nmatches++;

				// Step 5 计算匹配点旋转角度差所在的直方图
                if(mbCheckOrientation)
                {
					// 计算匹配特征点的角度差，这里单位是角度°，不是弧度
                    float rot = F1.mvKeysUn[i1].angle-F2.mvKeysUn[bestIdx2].angle;
                    if(rot<0.0)
                        rot+=360.0f;
					// 前面factor = HISTO_LENGTH/360.0f 
					// bin = rot / 360.of * HISTO_LENGTH 表示当前rot被分配在第几个直方图bin  
                    int bin = round(rot*factor);
					// 如果bin 满了又是一个轮回
                    if(bin==HISTO_LENGTH)
                        bin=0;
                    assert(bin>=0 && bin<HISTO_LENGTH);
                    rotHist[bin].push_back(i1);
                }
            }
        }

    }

	// Step 6 筛除旋转直方图中“非主流”部分
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

		// 筛选出在旋转角度差落在在直方图区间内数量最多的前三个bin的索引
        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);//计算最大三个，排序

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
			// 剔除掉不在前三的匹配对，因为他们不符合“主流旋转方向”   
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                int idx1 = rotHist[i][j];
                if(vnMatches12[idx1]>=0)
                {
                    vnMatches12[idx1]=-1;
                    nmatches--;
                }
            }
        }

    }

    //Update prev matched
	// Step 7 将最后通过筛选的匹配好的特征点保存到vbPrevMatched
    for(size_t i1=0, iend1=vnMatches12.size(); i1<iend1; i1++)
        if(vnMatches12[i1]>=0)
            vbPrevMatched[i1]=F2.mvKeysUn[vnMatches12[i1]].pt;

    return nmatches;
}


/*
 * @brief 通过词袋，对关键帧的特征点进行跟踪
 * 步骤
 * Step 1：分别取出属于同一node的ORB特征点(只有属于同一node，才有可能是匹配点)
 * Step 2：遍历KF中属于该node的特征点
 * Step 3：遍历F中属于该node的特征点，寻找最佳匹配点
 * Step 4：根据阈值 和 角度投票剔除误匹配
 * Step 5：根据方向剔除误匹配的点
 * @param  pKF               关键帧
 * @param  F                 当前普通帧
 * @param  vpMapPointMatches F中地图点对应的匹配，NULL表示未匹配
 * @return                   成功匹配的数量
 */
int ORBmatcher::SearchByBoW(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint *> &vpMatches12)
{
	// 获取该关键帧的地图点
    const vector<cv::KeyPoint> &vKeysUn1 = pKF1->mvKeysUn;

	// 和普通帧F特征点的索引一致
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;

	// 取出关键帧的词袋特征向量
    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const cv::Mat &Descriptors1 = pKF1->mDescriptors;

    const vector<cv::KeyPoint> &vKeysUn2 = pKF2->mvKeysUn;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;
    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const cv::Mat &Descriptors2 = pKF2->mDescriptors;

    vpMatches12 = vector<MapPoint*>(vpMapPoints1.size(),static_cast<MapPoint*>(NULL));
    vector<bool> vbMatched2(vpMapPoints2.size(),false);

    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

    const float factor = 1.0f/HISTO_LENGTH;

    int nmatches = 0;

    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

    while(f1it != f1end && f2it != f2end)
    {
        if(f1it->first == f2it->first)
        {
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
                const size_t idx1 = f1it->second[i1];

                MapPoint* pMP1 = vpMapPoints1[idx1];
                if(!pMP1)
                    continue;
                if(pMP1->isBad())
                    continue;

                const cv::Mat &d1 = Descriptors1.row(idx1);

                int bestDist1=256;
                int bestIdx2 =-1 ;
                int bestDist2=256;

                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
                    const size_t idx2 = f2it->second[i2];

                    MapPoint* pMP2 = vpMapPoints2[idx2];

                    if(vbMatched2[idx2] || !pMP2)
                        continue;

                    if(pMP2->isBad())
                        continue;

                    const cv::Mat &d2 = Descriptors2.row(idx2);

                    int dist = DescriptorDistance(d1,d2);

                    if(dist<bestDist1)
                    {
                        bestDist2=bestDist1;
                        bestDist1=dist;
                        bestIdx2=idx2;
                    }
                    else if(dist<bestDist2)
                    {
                        bestDist2=dist;
                    }
                }

                if(bestDist1<TH_LOW)
                {
                    if(static_cast<float>(bestDist1)<mfNNratio*static_cast<float>(bestDist2))
                    {
                        vpMatches12[idx1]=vpMapPoints2[bestIdx2];
                        vbMatched2[bestIdx2]=true;

                        if(mbCheckOrientation)
                        {
                            float rot = vKeysUn1[idx1].angle-vKeysUn2[bestIdx2].angle;
                            if(rot<0.0)
                                rot+=360.0f;
                            int bin = round(rot*factor);
                            if(bin==HISTO_LENGTH)
                                bin=0;
                            assert(bin>=0 && bin<HISTO_LENGTH);
                            rotHist[bin].push_back(idx1);
                        }
                        nmatches++;
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
                vpMatches12[rotHist[i][j]]=static_cast<MapPoint*>(NULL);
                nmatches--;
            }
        }
    }

    return nmatches;
}

/*
@brief 利用基础矩阵F12极线约束，用BoW加速匹配两个关键帧的未匹配的特征点，产生新的匹配点对
具体来说，pKF1图像的每个特征点与pKF2图像同一node节点的所有特征点依次匹配，判断是否满足对极几何约束，满足约束的就是匹配的特征点
@param pKF1          关键帧1
@param pKF2          关键帧2
@param F12           从2到1的基础矩阵
@param vMatchedPairs 存储匹配特征点对，特征点用其在关键帧中的索引表示
@param bOnlyStereo   在双目和rgbd的情况下，是否要求特征点在右图存在匹配
@return              成功匹配的数量
*/
int ORBmatcher::SearchForTriangulation(KeyFrame *pKF1, KeyFrame *pKF2, cv::Mat F12,
                                       vector<pair<size_t, size_t> > &vMatchedPairs, const bool bOnlyStereo)
{    
    const DBoW2::FeatureVector &vFeatVec1 = pKF1->mFeatVec;
    const DBoW2::FeatureVector &vFeatVec2 = pKF2->mFeatVec;

    //Compute epipole in second image
	//step1 计算KF1的相机中心在KF2图像平面的二维像素坐标

	//KF1相机光心在世界坐标系坐标Cw
    cv::Mat Cw = pKF1->GetCameraCenter();
	//KF2相机位姿R2w,t2w,是世界坐标系到相机坐标系的转换
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();
	//KF1的相机光心转化到KF2坐标系中的坐标
    cv::Mat C2 = R2w*Cw+t2w;
    const float invz = 1.0f/C2.at<float>(2);
	//得到KF1的相机光心在KF2中的坐标，也叫极点，这里是像素坐标
    const float ex =pKF2->fx*C2.at<float>(0)*invz+pKF2->cx;
    const float ey =pKF2->fy*C2.at<float>(1)*invz+pKF2->cy;

    // Find matches between not tracked keypoints
    // Matching speed-up by ORB Vocabulary
    // Compare only ORB that share the same node

    int nmatches=0;
	//记录匹配时候成功，避免重复匹配
    vector<bool> vbMatched2(pKF2->N,false);
    vector<int> vMatches12(pKF1->N,-1);

	//用于统计点对旋转差的直方图
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);

	//!原作者代码时const float factor = 1.0f/HISTO_LENGTH;，是错误的，改成下面的
    const float factor = HISTO_LENGTH/ 360.0f;

	//step2 利用BoW加速匹配：只对同一节点（待定层）的ORB特征进行匹配
	//FeatureVector其实是一个map类，那就可以直接获取它的迭代器进行遍历
	//FeatureVector的数据结构类似于:{(node1,feature_vector1)(node2,feature_vector2)...}
	//f1it->first对应node编号，f1it->second对应属于该node的所有特征点编号
    DBoW2::FeatureVector::const_iterator f1it = vFeatVec1.begin();
    DBoW2::FeatureVector::const_iterator f2it = vFeatVec2.begin();
    DBoW2::FeatureVector::const_iterator f1end = vFeatVec1.end();
    DBoW2::FeatureVector::const_iterator f2end = vFeatVec2.end();

	//step2.1 遍历pKF1和pKF2中的node节点
    while(f1it!=f1end && f2it!=f2end)
    {
		//如果f1it和f2it属于同一个node节点才会进行匹配
        if(f1it->first == f2it->first)
        {
			//step2.2 遍历属于同一个node节点（id：f1it->first）下的所有特征点
            for(size_t i1=0, iend1=f1it->second.size(); i1<iend1; i1++)
            {
				//获取pKF1中属于该node节点的所有特征点索引
                const size_t idx1 = f1it->second[i1];
                
				//step2.3 通过特征点索引idx1在pKF1中取出对应的MapPoint
                MapPoint* pMP1 = pKF1->GetMapPoint(idx1);
                
                // If there is already a MapPoint skip
				// 由于寻找的是未匹配的特征点，所以pMP1应该为null
                if(pMP1)
                    continue;

				//如果mvuRight中的值大于0，表示的是双目，且该特征点有深度值
                const bool bStereo1 = pKF1->mvuRight[idx1]>=0;

                if(bOnlyStereo)
                    if(!bStereo1)
                        continue;
                
				//step2.4 通过特征点索引idx1在pKF1中取出对应的特征点
                const cv::KeyPoint &kp1 = pKF1->mvKeysUn[idx1];
                
				//通过特征点索引idx1在pKF1中取出对应特征点的描述子
                const cv::Mat &d1 = pKF1->mDescriptors.row(idx1);
                
                int bestDist = TH_LOW;//特征点匹配的最小阈值
                int bestIdx2 = -1;
                
				//step2.5：遍历该node节点下（f2it->first）对应KF2中所有的特征点
                for(size_t i2=0, iend2=f2it->second.size(); i2<iend2; i2++)
                {
					//获取pKF2中属于该node节点的所有特征点索引
                    size_t idx2 = f2it->second[i2];
                    
					//通过特征点索引idx2在pKF2中取出对应的MapPoint
                    MapPoint* pMP2 = pKF2->GetMapPoint(idx2);
                    
                    // If we have already matched or there is a MapPoint skip
					// 如果pKF2当前特征点索引idx2已经被匹配过或者对应3d点非空，那么跳过这个索引idx2
                    if(vbMatched2[idx2] || pMP2)
                        continue;

                    const bool bStereo2 = pKF2->mvuRight[idx2]>=0;

                    if(bOnlyStereo)
                        if(!bStereo2)
                            continue;
                    
					//通过特征点索引idx2在pKF2中取出对应的特征点的描述子
                    const cv::Mat &d2 = pKF2->mDescriptors.row(idx2);
                    
					//step2.6 计算idx1与idx2在两个关键帧中对应特征点的描述子距离
                    const int dist = DescriptorDistance(d1,d2);
                    
                    if(dist>TH_LOW || dist>bestDist)
                        continue;

					//通过特征点索引idx2在pKF2中取出对应的特征点
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[idx2];

					//为什么双目不需要判断像素点到极点的距离？
					//因为双目模式下可以左右互相匹配恢复三维点
                    if(!bStereo1 && !bStereo2)
                    {
                        const float distex = ex-kp2.pt.x;
                        const float distey = ey-kp2.pt.y;
						//step2.7 极点e2到kp2的像素距离如果小于阈值th，认为kp2对应的MapPoint距离pKF1相机太近，跳过该匹配
						//作者根据kp2金字塔尺度因子，(scale^n,scale=1.2,n为层数),定义阈值th
						//金字塔层数从0到7，对应距离sqrt(100*pKF2->mvscaleFactors[kp2.octave])是10-20像素
						//?对这个阈值的有效性持怀疑态度
                        if(distex*distex+distey*distey<100*pKF2->mvScaleFactors[kp2.octave])
                            continue;
                    }

					//step2.8 计算特征点kp2到kp1对应极线的距离是否小于阈值
                    if(CheckDistEpipolarLine(kp1,kp2,F12,pKF2))
                    {
						//bestIdx2,bestDist是kp1对应kp2中的最佳匹配点index及匹配距离
                        bestIdx2 = idx2;
                        bestDist = dist;
                    }
                }
                
                if(bestIdx2>=0)
                {
                    const cv::KeyPoint &kp2 = pKF2->mvKeysUn[bestIdx2];
					//记录匹配结果
                    vMatches12[idx1]=bestIdx2;
					vbMatched2[bestIdx2] = true;//记录已匹配，避免重复匹配，作者漏了
                    nmatches++;

					//记录旋转差直方图信息
                    if(mbCheckOrientation)
                    {
						//angle角度，表示匹配点对的方向差
                        float rot = kp1.angle-kp2.angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(idx1);
                    }
                }
            }

            f1it++;
            f2it++;
        }
        else if(f1it->first < f2it->first)
        {
            f1it = vFeatVec1.lower_bound(f2it->first);
        }
        else
        {
            f2it = vFeatVec2.lower_bound(f1it->first);
        }
    }
	//step3 用旋转差直方图来筛选掉错误匹配对
    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i==ind1 || i==ind2 || i==ind3)
                continue;
            for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
            {
				vbMatched2[vMatches12[rotHist[i][j]]] = false;//！清除匹配关系，原作者漏掉
                vMatches12[rotHist[i][j]]=-1;
                nmatches--;
            }
        }

    }

	//step4 储存匹配关系，下标是关键帧1的特征点id，储存的是关键帧2的特征点id
    vMatchedPairs.clear();
    vMatchedPairs.reserve(nmatches);

    for(size_t i=0, iend=vMatches12.size(); i<iend; i++)
    {
        if(vMatches12[i]<0)
            continue;
        vMatchedPairs.push_back(make_pair(i,vMatches12[i]));
    }

    return nmatches;
}

/**
 * @brief 将地图点投影到关键帧中进行匹配和融合；融合策略如下
 * 1.如果地图点能匹配关键帧的特征点，并且该点有对应的地图点，那么选择观测数目多的替换两个地图点
 * 2.如果地图点能匹配关键帧的特征点，并且该点没有对应的地图点，那么为该点添加该投影地图点

 * @param[in] pKF           关键帧
 * @param[in] vpMapPoints   待投影的地图点
 * @param[in] th            搜索窗口的阈值，默认为3
 * @return int              更新地图点的数量
 */
int ORBmatcher::Fuse(KeyFrame *pKF, const vector<MapPoint *> &vpMapPoints, const float th)
{
	// 取出当前帧位姿、内参、光心在世界坐标系下坐标
	cv::Mat Rcw = pKF->GetRotation();
	cv::Mat tcw = pKF->GetTranslation();

	const float &fx = pKF->fx;
	const float &fy = pKF->fy;
	const float &cx = pKF->cx;
	const float &cy = pKF->cy;
	const float &bf = pKF->mbf;

	cv::Mat Ow = pKF->GetCameraCenter();

	int nFused = 0;

	const int nMPs = vpMapPoints.size();

	// 遍历所有的待投影地图点
	for (int i = 0; i < nMPs; i++)
	{

		MapPoint* pMP = vpMapPoints[i];
		// Step 1 判断地图点的有效性 
		if (!pMP)
			continue;
		// 地图点无效 或 已经是该帧的地图点（无需融合），跳过
		if (pMP->isBad() || pMP->IsInKeyFrame(pKF))
			continue;

		// 将地图点变换到关键帧的相机坐标系下
		cv::Mat p3Dw = pMP->GetWorldPos();
		cv::Mat p3Dc = Rcw * p3Dw + tcw;

		// Depth must be positive
		// 深度值为负，跳过
		if (p3Dc.at<float>(2) < 0.0f)
			continue;

		// Step 2 得到地图点投影到关键帧的图像坐标
		const float invz = 1 / p3Dc.at<float>(2);
		const float x = p3Dc.at<float>(0)*invz;
		const float y = p3Dc.at<float>(1)*invz;

		const float u = fx * x + cx;
		const float v = fy * y + cy;

		// Point must be inside the image
		// 投影点需要在有效范围内
		if (!pKF->IsInImage(u, v))
			continue;

		const float ur = u - bf * invz;

		const float maxDistance = pMP->GetMaxDistanceInvariance();
		const float minDistance = pMP->GetMinDistanceInvariance();
		cv::Mat PO = p3Dw - Ow;
		const float dist3D = cv::norm(PO);

		// Depth must be inside the scale pyramid of the image
		// Step 3 地图点到关键帧相机光心距离需满足在有效范围内
		if (dist3D<minDistance || dist3D>maxDistance)
			continue;

		// Viewing angle must be less than 60 deg
		// Step 4 地图点到光心的连线与该地图点的平均观测向量之间夹角要小于60°
		cv::Mat Pn = pMP->GetNormal();
		if (PO.dot(Pn) < 0.5*dist3D)
			continue;
		// 根据地图点到相机光心距离预测匹配点所在的金字塔尺度
		int nPredictedLevel = pMP->PredictScale(dist3D, pKF);

		// Search in a radius
		// 确定搜索范围
		const float radius = th * pKF->mvScaleFactors[nPredictedLevel];
		// Step 5 在投影点附近搜索窗口内找到候选匹配点的索引
		const vector<size_t> vIndices = pKF->GetFeaturesInArea(u, v, radius);

		if (vIndices.empty())
			continue;

		// Match to the most similar keypoint in the radius
		 // Step 6 遍历寻找最佳匹配点
		const cv::Mat dMP = pMP->GetDescriptor();

		int bestDist = 256;
		int bestIdx = -1;
		for (vector<size_t>::const_iterator vit = vIndices.begin(), vend = vIndices.end(); vit != vend; vit++)// 步骤3：遍历搜索范围内的features
		{
			const size_t idx = *vit;

			const cv::KeyPoint &kp = pKF->mvKeysUn[idx];

			const int &kpLevel = kp.octave;
			// 金字塔层级要接近（同一层或小一层），否则跳过
			if (kpLevel<nPredictedLevel - 1 || kpLevel>nPredictedLevel)
				continue;

			// 计算投影点与候选匹配特征点的距离，如果偏差很大，直接跳过
			if (pKF->mvuRight[idx] >= 0)
			{
				// Check reprojection error in stereo
				// 双目情况
				const float &kpx = kp.pt.x;
				const float &kpy = kp.pt.y;
				const float &kpr = pKF->mvuRight[idx];
				const float ex = u - kpx;
				const float ey = v - kpy;
				// 右目数据的偏差也要考虑进去
				const float er = ur - kpr;
				const float e2 = ex * ex + ey * ey + er * er;

				//自由度为3, 误差小于1个像素,这种事情95%发生的概率对应卡方检验阈值为7.82
				if (e2*pKF->mvInvLevelSigma2[kpLevel] > 7.8)
					continue;
			}
			else
			{
				// 计算投影点与候选匹配特征点的距离，如果偏差很大，直接跳过
				// 单目情况
				const float &kpx = kp.pt.x;
				const float &kpy = kp.pt.y;
				const float ex = u - kpx;
				const float ey = v - kpy;
				const float e2 = ex * ex + ey * ey;

				// 自由度为2的，卡方检验阈值5.99（假设测量有一个像素的偏差）
				if (e2*pKF->mvInvLevelSigma2[kpLevel] > 5.99)
					continue;
			}

			const cv::Mat &dKF = pKF->mDescriptors.row(idx);

			const int dist = DescriptorDistance(dMP, dKF);
			// 和投影点的描述子距离最小
			if (dist < bestDist)
			{
				bestDist = dist;
				bestIdx = idx;
			}
		}

		// If there is already a MapPoint replace otherwise add new measurement
		// Step 7 找到投影点对应的最佳匹配特征点，根据是否存在地图点来融合或新增
		// 最佳匹配距离要小于阈值
		if (bestDist <= TH_LOW)
		{
			MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
			if (pMPinKF)
			{
				// 如果最佳匹配点有对应有效地图点，选择被观测次数最多的那个替换
				if (!pMPinKF->isBad())
				{
					if (pMPinKF->Observations() > pMP->Observations())
						pMP->Replace(pMPinKF);
					else
						pMPinKF->Replace(pMP);
				}
			}
			else
			{
				// 如果最佳匹配点没有对应地图点，添加观测信息
				pMP->AddObservation(pKF, bestIdx);
				pKF->AddMapPoint(pMP, bestIdx);
			}
			nFused++;
		}
	}

	return nFused;
}


int ORBmatcher::Fuse(KeyFrame *pKF, cv::Mat Scw, const vector<MapPoint *> &vpPoints, float th, vector<MapPoint *> &vpReplacePoint)
{
    // Get Calibration Parameters for later projection
    const float &fx = pKF->fx;
    const float &fy = pKF->fy;
    const float &cx = pKF->cx;
    const float &cy = pKF->cy;

    // Decompose Scw
    cv::Mat sRcw = Scw.rowRange(0,3).colRange(0,3);
    const float scw = sqrt(sRcw.row(0).dot(sRcw.row(0)));
    cv::Mat Rcw = sRcw/scw;
    cv::Mat tcw = Scw.rowRange(0,3).col(3)/scw;
    cv::Mat Ow = -Rcw.t()*tcw;

    // Set of MapPoints already found in the KeyFrame
    const set<MapPoint*> spAlreadyFound = pKF->GetMapPoints();

    int nFused=0;

    const int nPoints = vpPoints.size();

    // For each candidate MapPoint project and match
    for(int iMP=0; iMP<nPoints; iMP++)
    {
        MapPoint* pMP = vpPoints[iMP];

        // Discard Bad MapPoints and already found
        if(pMP->isBad() || spAlreadyFound.count(pMP))
            continue;

        // Get 3D Coords.
        cv::Mat p3Dw = pMP->GetWorldPos();

        // Transform into Camera Coords.
        cv::Mat p3Dc = Rcw*p3Dw+tcw;

        // Depth must be positive
        if(p3Dc.at<float>(2)<0.0f)
            continue;

        // Project into Image
        const float invz = 1.0/p3Dc.at<float>(2);
        const float x = p3Dc.at<float>(0)*invz;
        const float y = p3Dc.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF->IsInImage(u,v))
            continue;

        // Depth must be inside the scale pyramid of the image
        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        cv::Mat PO = p3Dw-Ow;
        const float dist3D = cv::norm(PO);

        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Viewing angle must be less than 60 deg
        cv::Mat Pn = pMP->GetNormal();

        if(PO.dot(Pn)<0.5*dist3D)
            continue;

        // Compute predicted scale level
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF);

        // Search in a radius
        const float radius = th*pKF->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius

        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(); vit!=vIndices.end(); vit++)
        {
            const size_t idx = *vit;
            const int &kpLevel = pKF->mvKeysUn[idx].octave;

            if(kpLevel<nPredictedLevel-1 || kpLevel>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF->mDescriptors.row(idx);

            int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        // If there is already a MapPoint replace otherwise add new measurement
        if(bestDist<=TH_LOW)
        {
            MapPoint* pMPinKF = pKF->GetMapPoint(bestIdx);
            if(pMPinKF)
            {
                if(!pMPinKF->isBad())
                    vpReplacePoint[iMP] = pMPinKF;
            }
            else
            {
                pMP->AddObservation(pKF,bestIdx);
                pKF->AddMapPoint(pMP,bestIdx);
            }
            nFused++;
        }
    }

    return nFused;
}

int ORBmatcher::SearchBySim3(KeyFrame *pKF1, KeyFrame *pKF2, vector<MapPoint*> &vpMatches12,
                             const float &s12, const cv::Mat &R12, const cv::Mat &t12, const float th)
{
    const float &fx = pKF1->fx;
    const float &fy = pKF1->fy;
    const float &cx = pKF1->cx;
    const float &cy = pKF1->cy;

    // Camera 1 from world
    cv::Mat R1w = pKF1->GetRotation();
    cv::Mat t1w = pKF1->GetTranslation();

    //Camera 2 from world
    cv::Mat R2w = pKF2->GetRotation();
    cv::Mat t2w = pKF2->GetTranslation();

    //Transformation between cameras
    cv::Mat sR12 = s12*R12;
    cv::Mat sR21 = (1.0/s12)*R12.t();
    cv::Mat t21 = -sR21*t12;

    const vector<MapPoint*> vpMapPoints1 = pKF1->GetMapPointMatches();
    const int N1 = vpMapPoints1.size();

    const vector<MapPoint*> vpMapPoints2 = pKF2->GetMapPointMatches();
    const int N2 = vpMapPoints2.size();

    vector<bool> vbAlreadyMatched1(N1,false);
    vector<bool> vbAlreadyMatched2(N2,false);

    for(int i=0; i<N1; i++)
    {
        MapPoint* pMP = vpMatches12[i];
        if(pMP)
        {
            vbAlreadyMatched1[i]=true;
            int idx2 = pMP->GetIndexInKeyFrame(pKF2);
            if(idx2>=0 && idx2<N2)
                vbAlreadyMatched2[idx2]=true;
        }
    }

    vector<int> vnMatch1(N1,-1);
    vector<int> vnMatch2(N2,-1);

    // Transform from KF1 to KF2 and search
    for(int i1=0; i1<N1; i1++)
    {
        MapPoint* pMP = vpMapPoints1[i1];

        if(!pMP || vbAlreadyMatched1[i1])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc1 = R1w*p3Dw + t1w;
        cv::Mat p3Dc2 = sR21*p3Dc1 + t21;

        // Depth must be positive
        if(p3Dc2.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc2.at<float>(2);
        const float x = p3Dc2.at<float>(0)*invz;
        const float y = p3Dc2.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF2->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc2);

        // Depth must be inside the scale invariance region
        if(dist3D<minDistance || dist3D>maxDistance )
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF2);

        // Search in a radius
        const float radius = th*pKF2->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF2->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF2->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF2->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch1[i1]=bestIdx;
        }
    }

    // Transform from KF2 to KF2 and search
    for(int i2=0; i2<N2; i2++)
    {
        MapPoint* pMP = vpMapPoints2[i2];

        if(!pMP || vbAlreadyMatched2[i2])
            continue;

        if(pMP->isBad())
            continue;

        cv::Mat p3Dw = pMP->GetWorldPos();
        cv::Mat p3Dc2 = R2w*p3Dw + t2w;
        cv::Mat p3Dc1 = sR12*p3Dc2 + t12;

        // Depth must be positive
        if(p3Dc1.at<float>(2)<0.0)
            continue;

        const float invz = 1.0/p3Dc1.at<float>(2);
        const float x = p3Dc1.at<float>(0)*invz;
        const float y = p3Dc1.at<float>(1)*invz;

        const float u = fx*x+cx;
        const float v = fy*y+cy;

        // Point must be inside the image
        if(!pKF1->IsInImage(u,v))
            continue;

        const float maxDistance = pMP->GetMaxDistanceInvariance();
        const float minDistance = pMP->GetMinDistanceInvariance();
        const float dist3D = cv::norm(p3Dc1);

        // Depth must be inside the scale pyramid of the image
        if(dist3D<minDistance || dist3D>maxDistance)
            continue;

        // Compute predicted octave
        const int nPredictedLevel = pMP->PredictScale(dist3D,pKF1);

        // Search in a radius of 2.5*sigma(ScaleLevel)
        const float radius = th*pKF1->mvScaleFactors[nPredictedLevel];

        const vector<size_t> vIndices = pKF1->GetFeaturesInArea(u,v,radius);

        if(vIndices.empty())
            continue;

        // Match to the most similar keypoint in the radius
        const cv::Mat dMP = pMP->GetDescriptor();

        int bestDist = INT_MAX;
        int bestIdx = -1;
        for(vector<size_t>::const_iterator vit=vIndices.begin(), vend=vIndices.end(); vit!=vend; vit++)
        {
            const size_t idx = *vit;

            const cv::KeyPoint &kp = pKF1->mvKeysUn[idx];

            if(kp.octave<nPredictedLevel-1 || kp.octave>nPredictedLevel)
                continue;

            const cv::Mat &dKF = pKF1->mDescriptors.row(idx);

            const int dist = DescriptorDistance(dMP,dKF);

            if(dist<bestDist)
            {
                bestDist = dist;
                bestIdx = idx;
            }
        }

        if(bestDist<=TH_HIGH)
        {
            vnMatch2[i2]=bestIdx;
        }
    }

    // Check agreement
    int nFound = 0;

    for(int i1=0; i1<N1; i1++)
    {
        int idx2 = vnMatch1[i1];

        if(idx2>=0)
        {
            int idx1 = vnMatch2[idx2];
            if(idx1==i1)
            {
                vpMatches12[i1] = vpMapPoints2[idx2];
                nFound++;
            }
        }
    }

    return nFound;
}


/**
 * @brief 将上一帧跟踪的地图点投影到当前帧，并且搜索匹配点。用于跟踪前一帧
 * 步骤
 * Step 1 建立旋转直方图，用于检测旋转一致性
 * Step 2 计算当前帧和前一帧的平移向量
 * Step 3 对于前一帧的每一个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
 * Step 4 根据相机的前后前进方向来判断搜索尺度范围
 * Step 5 遍历候选匹配点，寻找距离最小的最佳匹配点
 * Step 6 计算匹配点旋转角度差所在的直方图
 * Step 7 进行旋转一致检测，剔除不一致的匹配
 * @param[in] CurrentFrame          当前帧
 * @param[in] LastFrame             上一帧
 * @param[in] th                    搜索范围阈值，默认单目为7，双目15
 * @param[in] bMono                 是否为单目
 * @return int                      成功匹配的数量
 */
int ORBmatcher::SearchByProjection(Frame &CurrentFrame, const Frame &LastFrame, const float th, const bool bMono)
{
	int nmatches = 0;

	// Rotation Histogram (to check rotation consistency)
	// Step 1 建立旋转直方图，用于检测旋转一致性
	vector<int> rotHist[HISTO_LENGTH];
	for (int i = 0; i < HISTO_LENGTH; i++)
		rotHist[i].reserve(500);

	//! 原作者代码是 const float factor = 1.0f/HISTO_LENGTH; 是错误的，更改为下面代码
	const float factor = HISTO_LENGTH / 360.0f;

	// Step 2 计算当前帧和前一帧的平移向量
	//当前帧的相机位姿
	const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0, 3).colRange(0, 3);
	const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0, 3).col(3);

	//当前相机坐标系到世界坐标系的平移向量
	const cv::Mat twc = -Rcw.t()*tcw;

	//上一帧的相机位姿
	const cv::Mat Rlw = LastFrame.mTcw.rowRange(0, 3).colRange(0, 3);
	const cv::Mat tlw = LastFrame.mTcw.rowRange(0, 3).col(3); // tlw(l)

	// vector from LastFrame to CurrentFrame expressed in LastFrame
	// 当前帧相对于上一帧相机的平移向量
	const cv::Mat tlc = Rlw * twc + tlw;

	// 判断前进还是后退
	const bool bForward = tlc.at<float>(2) > CurrentFrame.mb && !bMono;     // 非单目情况，如果Z大于基线，则表示相机明显前进
	const bool bBackward = -tlc.at<float>(2) > CurrentFrame.mb && !bMono;   // 非单目情况，如果-Z小于基线，则表示相机明显后退

	//  Step 3 对于前一帧的每一个地图点，通过相机投影模型，得到投影到当前帧的像素坐标
	for (int i = 0; i < LastFrame.N; i++)
	{
		MapPoint* pMP = LastFrame.mvpMapPoints[i];

		if (pMP)
		{
			if (!LastFrame.mvbOutlier[i])
			{
				// 对上一帧有效的MapPoints投影到当前帧坐标系
				cv::Mat x3Dw = pMP->GetWorldPos();
				cv::Mat x3Dc = Rcw * x3Dw + tcw;

				const float xc = x3Dc.at<float>(0);
				const float yc = x3Dc.at<float>(1);
				const float invzc = 1.0 / x3Dc.at<float>(2);

				if (invzc < 0)
					continue;

				// 投影到当前帧中
				float u = CurrentFrame.fx*xc*invzc + CurrentFrame.cx;
				float v = CurrentFrame.fy*yc*invzc + CurrentFrame.cy;

				if (u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
					continue;
				if (v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
					continue;

				// 上一帧中地图点对应二维特征点所在的金字塔层级
				int nLastOctave = LastFrame.mvKeys[i].octave;

				// Search in a window. Size depends on scale
				// 搜索范围阈值,单目：th = 7，双目：th = 15
				float radius = th * CurrentFrame.mvScaleFactors[nLastOctave]; // 尺度越大，搜索范围越大

				// 记录候选匹配点的id
				vector<size_t> vIndices2;

				// Step 4 根据相机的前后前进方向来判断搜索尺度范围。
				// 以下可以这么理解，例如一个有一定面积的圆点，在某个尺度n下它是一个特征点
				// 当相机前进时，圆点的面积增大，在某个尺度m下它是一个特征点，由于面积增大，则需要在更高的尺度下才能检测出来
				// 当相机后退时，圆点的面积减小，在某个尺度m下它是一个特征点，由于面积减小，则需要在更低的尺度下才能检测出来
				if (bForward) // 前进,则上一帧兴趣点在所在的尺度nLastOctave<=nCurOctave
					vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave);
				else if (bBackward) // 后退,则上一帧兴趣点在所在的尺度0<=nCurOctave<=nLastOctave
					vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, 0, nLastOctave);
				else // 在[nLastOctave-1, nLastOctave+1]中搜索
					vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nLastOctave - 1, nLastOctave + 1);

				if (vIndices2.empty())
					continue;

				const cv::Mat dMP = pMP->GetDescriptor();

				int bestDist = 256;
				int bestIdx2 = -1;

				// Step 5 遍历候选匹配点，寻找距离最小的最佳匹配点 
				for (vector<size_t>::const_iterator vit = vIndices2.begin(), vend = vIndices2.end(); vit != vend; vit++)
				{
					const size_t i2 = *vit;

					// 如果该特征点已经有对应的MapPoint了,则退出该次循环
					if (CurrentFrame.mvpMapPoints[i2])
						if (CurrentFrame.mvpMapPoints[i2]->Observations() > 0)
							continue;

					if (CurrentFrame.mvuRight[i2] > 0)
					{
						// 双目和rgbd的情况，需要保证右图的点也在搜索半径以内
						const float ur = u - CurrentFrame.mbf*invzc;
						const float er = fabs(ur - CurrentFrame.mvuRight[i2]);
						if (er > radius)
							continue;
					}

					const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

					const int dist = DescriptorDistance(dMP, d);

					if (dist < bestDist)
					{
						bestDist = dist;
						bestIdx2 = i2;
					}
				}

				// 最佳匹配距离要小于设定阈值
				if (bestDist <= TH_HIGH)
				{
					CurrentFrame.mvpMapPoints[bestIdx2] = pMP;
					nmatches++;

					// Step 6 计算匹配点旋转角度差所在的直方图
					if (mbCheckOrientation)
					{
						float rot = LastFrame.mvKeysUn[i].angle - CurrentFrame.mvKeysUn[bestIdx2].angle;
						if (rot < 0.0)
							rot += 360.0f;
						int bin = round(rot*factor);
						if (bin == HISTO_LENGTH)
							bin = 0;
						assert(bin >= 0 && bin < HISTO_LENGTH);
						rotHist[bin].push_back(bestIdx2);
					}
				}
			}
		}
	}

	//Apply rotation consistency
	//  Step 7 进行旋转一致检测，剔除不一致的匹配
	if (mbCheckOrientation)
	{
		int ind1 = -1;
		int ind2 = -1;
		int ind3 = -1;

		ComputeThreeMaxima(rotHist, HISTO_LENGTH, ind1, ind2, ind3);

		for (int i = 0; i < HISTO_LENGTH; i++)
		{
			// 对于数量不是前3个的点对，剔除
			if (i != ind1 && i != ind2 && i != ind3)
			{
				for (size_t j = 0, jend = rotHist[i].size(); j < jend; j++)
				{
					CurrentFrame.mvpMapPoints[rotHist[i][j]] = static_cast<MapPoint*>(NULL);
					nmatches--;
				}
			}
		}
	}

	return nmatches;
}


int ORBmatcher::SearchByProjection(Frame &CurrentFrame, KeyFrame *pKF, const set<MapPoint*> &sAlreadyFound, const float th , const int ORBdist)
{
    int nmatches = 0;

    const cv::Mat Rcw = CurrentFrame.mTcw.rowRange(0,3).colRange(0,3);
    const cv::Mat tcw = CurrentFrame.mTcw.rowRange(0,3).col(3);
    const cv::Mat Ow = -Rcw.t()*tcw;

    // Rotation Histogram (to check rotation consistency)
    vector<int> rotHist[HISTO_LENGTH];
    for(int i=0;i<HISTO_LENGTH;i++)
        rotHist[i].reserve(500);
    const float factor = 1.0f/HISTO_LENGTH;

    const vector<MapPoint*> vpMPs = pKF->GetMapPointMatches();

    for(size_t i=0, iend=vpMPs.size(); i<iend; i++)
    {
        MapPoint* pMP = vpMPs[i];

        if(pMP)
        {
            if(!pMP->isBad() && !sAlreadyFound.count(pMP))
            {
                //Project
                cv::Mat x3Dw = pMP->GetWorldPos();
                cv::Mat x3Dc = Rcw*x3Dw+tcw;

                const float xc = x3Dc.at<float>(0);
                const float yc = x3Dc.at<float>(1);
                const float invzc = 1.0/x3Dc.at<float>(2);

                const float u = CurrentFrame.fx*xc*invzc+CurrentFrame.cx;
                const float v = CurrentFrame.fy*yc*invzc+CurrentFrame.cy;

                if(u<CurrentFrame.mnMinX || u>CurrentFrame.mnMaxX)
                    continue;
                if(v<CurrentFrame.mnMinY || v>CurrentFrame.mnMaxY)
                    continue;

                // Compute predicted scale level
                cv::Mat PO = x3Dw-Ow;
                float dist3D = cv::norm(PO);

                const float maxDistance = pMP->GetMaxDistanceInvariance();
                const float minDistance = pMP->GetMinDistanceInvariance();

                // Depth must be inside the scale pyramid of the image
                if(dist3D<minDistance || dist3D>maxDistance)
                    continue;

                int nPredictedLevel = pMP->PredictScale(dist3D,&CurrentFrame);

                // Search in a window
                const float radius = th*CurrentFrame.mvScaleFactors[nPredictedLevel];

                const vector<size_t> vIndices2 = CurrentFrame.GetFeaturesInArea(u, v, radius, nPredictedLevel-1, nPredictedLevel+1);

                if(vIndices2.empty())
                    continue;

                const cv::Mat dMP = pMP->GetDescriptor();

                int bestDist = 256;
                int bestIdx2 = -1;

                for(vector<size_t>::const_iterator vit=vIndices2.begin(); vit!=vIndices2.end(); vit++)
                {
                    const size_t i2 = *vit;
                    if(CurrentFrame.mvpMapPoints[i2])
                        continue;

                    const cv::Mat &d = CurrentFrame.mDescriptors.row(i2);

                    const int dist = DescriptorDistance(dMP,d);

                    if(dist<bestDist)
                    {
                        bestDist=dist;
                        bestIdx2=i2;
                    }
                }

                if(bestDist<=ORBdist)
                {
                    CurrentFrame.mvpMapPoints[bestIdx2]=pMP;
                    nmatches++;

                    if(mbCheckOrientation)
                    {
                        float rot = pKF->mvKeysUn[i].angle-CurrentFrame.mvKeysUn[bestIdx2].angle;
                        if(rot<0.0)
                            rot+=360.0f;
                        int bin = round(rot*factor);
                        if(bin==HISTO_LENGTH)
                            bin=0;
                        assert(bin>=0 && bin<HISTO_LENGTH);
                        rotHist[bin].push_back(bestIdx2);
                    }
                }

            }
        }
    }

    if(mbCheckOrientation)
    {
        int ind1=-1;
        int ind2=-1;
        int ind3=-1;

        ComputeThreeMaxima(rotHist,HISTO_LENGTH,ind1,ind2,ind3);

        for(int i=0; i<HISTO_LENGTH; i++)
        {
            if(i!=ind1 && i!=ind2 && i!=ind3)
            {
                for(size_t j=0, jend=rotHist[i].size(); j<jend; j++)
                {
                    CurrentFrame.mvpMapPoints[rotHist[i][j]]=NULL;
                    nmatches--;
                }
            }
        }
    }

    return nmatches;
}

void ORBmatcher::ComputeThreeMaxima(vector<int>* histo, const int L, int &ind1, int &ind2, int &ind3)
{
    int max1=0;
    int max2=0;
    int max3=0;

    for(int i=0; i<L; i++)
    {
        const int s = histo[i].size();
        if(s>max1)
        {
            max3=max2;
            max2=max1;
            max1=s;
            ind3=ind2;
            ind2=ind1;
            ind1=i;
        }
        else if(s>max2)
        {
            max3=max2;
            max2=s;
            ind3=ind2;
            ind2=i;
        }
        else if(s>max3)
        {
            max3=s;
            ind3=i;
        }
    }

    if(max2<0.1f*(float)max1)
    {
        ind2=-1;
        ind3=-1;
    }
    else if(max3<0.1f*(float)max1)
    {
        ind3=-1;
    }
}


// Bit set count operation from
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
int ORBmatcher::DescriptorDistance(const cv::Mat &a, const cv::Mat &b)
{
    const int *pa = a.ptr<int32_t>();
    const int *pb = b.ptr<int32_t>();

    int dist=0;

    for(int i=0; i<8; i++, pa++, pb++)
    {
        unsigned  int v = *pa ^ *pb;
        v = v - ((v >> 1) & 0x55555555);
        v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}

} //namespace ORB_SLAM
