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

#include "LoopClosing.h"

#include "Sim3Solver.h"

#include "Converter.h"

#include "Optimizer.h"

#include "ORBmatcher.h"

#include<mutex>
#include<thread>


namespace ORB_SLAM2
{

LoopClosing::LoopClosing(Map *pMap, KeyFrameDatabase *pDB, ORBVocabulary *pVoc, const bool bFixScale):
    mbResetRequested(false), mbFinishRequested(false), mbFinished(true), mpMap(pMap),
    mpKeyFrameDB(pDB), mpORBVocabulary(pVoc), mpMatchedKF(NULL), mLastLoopKFid(0), mbRunningGBA(false), mbFinishedGBA(true),
    mbStopGBA(false), mpThreadGBA(NULL), mbFixScale(bFixScale), mnFullBAIdx(0)
{
    mnCovisibilityConsistencyTh = 3;
}

void LoopClosing::SetTracker(Tracking *pTracker)
{
    mpTracker=pTracker;
}

void LoopClosing::SetLocalMapper(LocalMapping *pLocalMapper)
{
    mpLocalMapper=pLocalMapper;
}


void LoopClosing::Run()
{
    mbFinished =false;

    while(1)
    {
        // Check if there are keyframes in the queue
        if(CheckNewKeyFrames())
        {
            // Detect loop candidates and check covisibility consistency
            if(DetectLoop())
            {
               // Compute similarity transformation [sR|t]
               // In the stereo/RGBD case s=1
               if(ComputeSim3())
               {
                   // Perform loop fusion and pose graph optimization
                   CorrectLoop();
               }
            }
        }       

        ResetIfRequested();

        if(CheckFinish())
            break;

        usleep(5000);
    }

    SetFinish();
}

void LoopClosing::InsertKeyFrame(KeyFrame *pKF)
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    if(pKF->mnId!=0)
        mlpLoopKeyFrameQueue.push_back(pKF);
}

bool LoopClosing::CheckNewKeyFrames()
{
    unique_lock<mutex> lock(mMutexLoopQueue);
    return(!mlpLoopKeyFrameQueue.empty());
}

bool LoopClosing::DetectLoop()
{
    {
        unique_lock<mutex> lock(mMutexLoopQueue);
        mpCurrentKF = mlpLoopKeyFrameQueue.front();
        mlpLoopKeyFrameQueue.pop_front();
        // Avoid that a keyframe can be erased while it is being process by this thread
        mpCurrentKF->SetNotErase();
    }

    //If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
    if(mpCurrentKF->mnId<mLastLoopKFid+10)
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mpCurrentKF->SetErase();
        return false;
    }

    // Compute reference BoW similarity score
    // This is the lowest score to a connected keyframe in the covisibility graph
    // We will impose loop candidates to have a higher similarity than this
    const vector<KeyFrame*> vpConnectedKeyFrames = mpCurrentKF->GetVectorCovisibleKeyFrames();
    const DBoW2::BowVector &CurrentBowVec = mpCurrentKF->mBowVec;
    float minScore = 1;
    for(size_t i=0; i<vpConnectedKeyFrames.size(); i++)
    {
        KeyFrame* pKF = vpConnectedKeyFrames[i];
        if(pKF->isBad())
            continue;
        const DBoW2::BowVector &BowVec = pKF->mBowVec;

        float score = mpORBVocabulary->score(CurrentBowVec, BowVec);

        if(score<minScore)
            minScore = score;
    }

    // Query the database imposing the minimum score
    vector<KeyFrame*> vpCandidateKFs = mpKeyFrameDB->DetectLoopCandidates(mpCurrentKF, minScore);

    // If there are no loop candidates, just add new keyframe and return false
    if(vpCandidateKFs.empty())
    {
        mpKeyFrameDB->add(mpCurrentKF);
        mvConsistentGroups.clear();
        mpCurrentKF->SetErase();
        return false;
    }

    // For each loop candidate check consistency with previous loop candidates
    // Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph)
    // A group is consistent with a previous group if they share at least a keyframe
    // We must detect a consistent loop in several consecutive keyframes to accept it
    mvpEnoughConsistentCandidates.clear();

    vector<ConsistentGroup> vCurrentConsistentGroups;
    vector<bool> vbConsistentGroup(mvConsistentGroups.size(),false);
    for(size_t i=0, iend=vpCandidateKFs.size(); i<iend; i++)
    {
        KeyFrame* pCandidateKF = vpCandidateKFs[i];

        set<KeyFrame*> spCandidateGroup = pCandidateKF->GetConnectedKeyFrames();
        spCandidateGroup.insert(pCandidateKF);

        bool bEnoughConsistent = false;
        bool bConsistentForSomeGroup = false;
        for(size_t iG=0, iendG=mvConsistentGroups.size(); iG<iendG; iG++)
        {
            set<KeyFrame*> sPreviousGroup = mvConsistentGroups[iG].first;

            bool bConsistent = false;
            for(set<KeyFrame*>::iterator sit=spCandidateGroup.begin(), send=spCandidateGroup.end(); sit!=send;sit++)
            {
                if(sPreviousGroup.count(*sit))
                {
                    bConsistent=true;
                    bConsistentForSomeGroup=true;
                    break;
                }
            }

            if(bConsistent)
            {
                int nPreviousConsistency = mvConsistentGroups[iG].second;
                int nCurrentConsistency = nPreviousConsistency + 1;
                if(!vbConsistentGroup[iG])
                {
                    ConsistentGroup cg = make_pair(spCandidateGroup,nCurrentConsistency);
                    vCurrentConsistentGroups.push_back(cg);
                    vbConsistentGroup[iG]=true; //this avoid to include the same group more than once
                }
                if(nCurrentConsistency>=mnCovisibilityConsistencyTh && !bEnoughConsistent)
                {
                    mvpEnoughConsistentCandidates.push_back(pCandidateKF);
                    bEnoughConsistent=true; //this avoid to insert the same candidate more than once
                }
            }
        }

        // If the group is not consistent with any previous group insert with consistency counter set to zero
        if(!bConsistentForSomeGroup)
        {
            ConsistentGroup cg = make_pair(spCandidateGroup,0);
            vCurrentConsistentGroups.push_back(cg);
        }
    }

    // Update Covisibility Consistent Groups
    mvConsistentGroups = vCurrentConsistentGroups;


    // Add Current Keyframe to database
    mpKeyFrameDB->add(mpCurrentKF);

    if(mvpEnoughConsistentCandidates.empty())
    {
        mpCurrentKF->SetErase();
        return false;
    }
    else
    {
        return true;
    }

    mpCurrentKF->SetErase();
    return false;
}

/**
 * @brief 计算当前关键帧和上一步闭环候选帧的Sim3变换
 * 1. 遍历闭环候选帧集，筛选出与当前帧的匹配特征点数大于20的候选帧集合，并为每一个候选帧构造一个Sim3Solver
 * 2. 对每一个候选帧进行 Sim3Solver 迭代匹配，直到有一个候选帧匹配成功，或者全部失败
 * 3. 取出闭环匹配上关键帧的相连关键帧，得到它们的地图点放入 mvpLoopMapPoints
 * 4. 将闭环匹配上关键帧以及相连关键帧的地图点投影到当前关键帧进行投影匹配
 * 5. 判断当前帧与检测出的所有闭环关键帧是否有足够多的地图点匹配
 * 6. 清空mvpEnoughConsistentCandidates
 * @return true         只要有一个候选关键帧通过Sim3的求解与优化，就返回true
 * @return false        所有候选关键帧与当前关键帧都没有有效Sim3变换
 */
bool LoopClosing::ComputeSim3()
{
	// Sim3 计算流程说明：
	// 1. 通过Bow加速描述子的匹配，利用RANSAC粗略地计算出当前帧与闭环帧的Sim3（当前帧---闭环帧）          
	// 2. 根据估计的Sim3，对3D点进行投影找到更多匹配，通过优化的方法计算更精确的Sim3（当前帧---闭环帧）   
	// 3. 将闭环帧以及闭环帧相连的关键帧的地图点与当前帧的点进行匹配（当前帧---闭环帧+相连关键帧）     
	// 注意以上匹配的结果均都存在成员变量mvpCurrentMatchedPoints中，实际的更新步骤见CorrectLoop()步骤3
	// 对于双目或者是RGBD输入的情况,计算得到的尺度=1

    //  准备工作
	// For each consistent loop candidate we try to compute a Sim3
	// 对每个（上一步得到的具有足够连续关系的）闭环候选帧都准备算一个Sim3
    const int nInitialCandidates = mvpEnoughConsistentCandidates.size();

    // We compute first ORB matches for each candidate
    // If enough matches are found, we setup a Sim3Solver
    ORBmatcher matcher(0.75,true);

	// 存储每一个候选帧的Sim3Solver求解器
    vector<Sim3Solver*> vpSim3Solvers;
    vpSim3Solvers.resize(nInitialCandidates);

	// 存储每个候选帧的匹配地图点信息
    vector<vector<MapPoint*> > vvpMapPointMatches;
    vvpMapPointMatches.resize(nInitialCandidates);

	// 存储每个候选帧应该被放弃(True）或者 保留(False)
    vector<bool> vbDiscarded;
    vbDiscarded.resize(nInitialCandidates);

	// 完成 Step 1 的匹配后，被保留的候选帧数量
    int nCandidates=0; //candidates with enough matches

	// Step 1. 遍历闭环候选帧集，初步筛选出与当前关键帧的匹配特征点数大于20的候选帧集合，并为每一个候选帧构造一个Sim3Solver
    for(int i=0; i<nInitialCandidates; i++)
    {
		// Step 1.1 从筛选的闭环候选帧中取出一帧有效关键帧pKF
        KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

        // avoid that local mapping erase it while it is being processed in this thread
		// 避免在LocalMapping中KeyFrameCulling函数将此关键帧作为冗余帧剔除
        pKF->SetNotErase();

		// 如果候选帧质量不高，直接PASS
        if(pKF->isBad())
        {
            vbDiscarded[i] = true;
            continue;
        }

		// Step 1.2 将当前帧 mpCurrentKF 与闭环候选关键帧pKF匹配
		// 通过bow加速得到 mpCurrentKF 与 pKF 之间的匹配特征点
		// vvpMapPointMatches 是匹配特征点对应的地图点,本质上来自于候选闭环帧
        int nmatches = matcher.SearchByBoW(mpCurrentKF,pKF,vvpMapPointMatches[i]);

		// 粗筛：匹配的特征点数太少，该候选帧剔除
        if(nmatches<20)
        {
            vbDiscarded[i] = true;
            continue;
        }
        else
        {
			// Step 1.3 为保留的候选帧构造Sim3求解器
			// 如果 mbFixScale（是否固定尺度） 为 true，则是6 自由度优化（双目 RGBD）
			// 如果是false，则是7 自由度优化（单目）
            Sim3Solver* pSolver = new Sim3Solver(mpCurrentKF,pKF,vvpMapPointMatches[i],mbFixScale);
            pSolver->SetRansacParameters(0.99,20,300);//至少20个内点才停止迭代，最多迭代300次，置信度0.99
            vpSim3Solvers[i] = pSolver;
        }

		// 保留的候选帧数量
        nCandidates++;
    }
	// 用于标记是否有一个候选帧通过Sim3Solver的求解与优化
    bool bMatch = false;

    // Perform alternatively RANSAC iterations for each candidate
    // until one is succesful or all fail
	// Step 2 对每一个候选帧用Sim3Solver 迭代匹配，直到有一个候选帧匹配成功，或者全部失败
    while(nCandidates>0 && !bMatch)
    {
        for(int i=0; i<nInitialCandidates; i++)
        {
			// 遍历每一个候选帧
            if(vbDiscarded[i])
                continue;

            KeyFrame* pKF = mvpEnoughConsistentCandidates[i];

			// 内点（Inliers）标志
			// 即标记经过RANSAC sim3 求解后,vvpMapPointMatches中的哪些作为内点
			// Perform 5 Ransac Iterations
            vector<bool> vbInliers;
			// 内点（Inliers）数量
            int nInliers;
			// 是否到达了最优解
            bool bNoMore;

			// Step 2.1 取出从 Step 1.3 中为当前候选帧构建的 Sim3Solver 并开始迭代
            Sim3Solver* pSolver = vpSim3Solvers[i];
			// 最多迭代5次，返回的Scm是候选帧pKF到当前帧mpCurrentKF的Sim3变换（T12）
            cv::Mat Scm  = pSolver->iterate(5,bNoMore,vbInliers,nInliers);

            // If Ransac reachs max. iterations discard keyframe
			// 总迭代次数达到最大限制还没有求出合格的Sim3变换，该候选帧剔除
            if(bNoMore)
            {
                vbDiscarded[i]=true;
                nCandidates--;
            }

            // If RANSAC returns a Sim3, perform a guided matching and optimize with all correspondences
			// 如果计算出了Sim3变换，继续匹配出更多点并优化。因为之前 SearchByBoW 匹配可能会有遗漏
            if(!Scm.empty())
            {
				// 取出经过Sim3Solver 后匹配点中的内点集合
                vector<MapPoint*> vpMapPointMatches(vvpMapPointMatches[i].size(), static_cast<MapPoint*>(NULL));
                for(size_t j=0, jend=vbInliers.size(); j<jend; j++)
                {
					// 保存内点
                    if(vbInliers[j])
                       vpMapPointMatches[j]=vvpMapPointMatches[i][j];
                }

				// Step 2.2 通过上面求取的Sim3变换引导关键帧匹配，弥补Step 1中的漏匹配
				// 候选帧pKF到当前帧mpCurrentKF的R（R12），t（t12），变换尺度s（s12）
                cv::Mat R = pSolver->GetEstimatedRotation();
                cv::Mat t = pSolver->GetEstimatedTranslation();
                const float s = pSolver->GetEstimatedScale();

				// 查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数，之前使用SearchByBoW进行特征点匹配时会有漏匹配）
                matcher.SearchBySim3(mpCurrentKF,pKF,vpMapPointMatches,s,R,t,7.5);//通过Sim3变换，投影搜索pKF1的特征点在pKF2中的匹配，同理，投影搜索pKF2的特征点在pKF1中的匹配。只有互相都成功匹配的才认为是可靠的匹配。

				// Step 2.3 用新的匹配来优化 Sim3，只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
				// OpenCV的Mat矩阵转成Eigen的Matrix类型
				// gScm：候选关键帧到当前帧的Sim3变换
                g2o::Sim3 gScm(Converter::toMatrix3d(R),Converter::toVector3d(t),s);
				
				// 如果mbFixScale为true，则是6 自由度优化（双目 RGBD），如果是false，则是7 自由度优化（单目）
				// 优化mpCurrentKF与pKF对应的MapPoints间的Sim3，得到优化后的量gScm
                const int nInliers = Optimizer::OptimizeSim3(mpCurrentKF, pKF, vpMapPointMatches, gScm, 10, mbFixScale);

				// 如果优化成功，则停止while循环遍历闭环候选
                // If optimization is succesful stop ransacs and continue
                if(nInliers>=20)
                {
					// 为True时将不再进入 while循环
                    bMatch = true;
					// mpMatchedKF就是最终闭环检测出来与当前帧形成闭环的关键帧
                    mpMatchedKF = pKF;

					// gSmw：从世界坐标系 w 到该候选帧 m 的Sim3变换，都在一个坐标系下，所以尺度 Scale=1
                    g2o::Sim3 gSmw(Converter::toMatrix3d(pKF->GetRotation()),Converter::toVector3d(pKF->GetTranslation()),1.0);
					// 得到g2o优化后从世界坐标系到当前帧的Sim3变换
                    mg2oScw = gScm*gSmw;
                    mScw = Converter::toCvMat(mg2oScw);

                    mvpCurrentMatchedPoints = vpMapPointMatches;
					// 只要有一个候选帧通过Sim3的求解与优化，就跳出停止对其它候选帧的判断
                    break;
                }
            }
        }
    }
	// 退出上面while循环的原因有两种,一种是求解到了bMatch置位后出的,另外一种是nCandidates耗尽为0
    if(!bMatch)
    {
		// 如果没有一个闭环匹配候选帧通过Sim3的求解与优化
		// 清空mvpEnoughConsistentCandidates，这些候选关键帧以后都不会在再参加回环检测过程了
        for(int i=0; i<nInitialCandidates; i++)
             mvpEnoughConsistentCandidates[i]->SetErase();
		// 当前关键帧也将不会再参加回环检测了
        mpCurrentKF->SetErase();
		// Sim3 计算失败，退出了
        return false;
    }

	// Step 3：取出与当前帧闭环匹配上的关键帧及其共视关键帧，以及这些共视关键帧的地图点
	// 注意是闭环检测出来与当前帧形成闭环的关键帧 mpMatchedKF
	// 将mpMatchedKF共视的关键帧全部取出来放入 vpLoopConnectedKFs
	// 将vpLoopConnectedKFs的地图点取出来放入mvpLoopMapPoints
    // Retrieve MapPoints seen in Loop Keyframe and neighbors
    vector<KeyFrame*> vpLoopConnectedKFs = mpMatchedKF->GetVectorCovisibleKeyFrames();

	// 包含闭环匹配关键帧本身,形成一个“闭环关键帧小组”
    vpLoopConnectedKFs.push_back(mpMatchedKF);
    mvpLoopMapPoints.clear();

	// 遍历这个组中的每一个关键帧
    for(vector<KeyFrame*>::iterator vit=vpLoopConnectedKFs.begin(); vit!=vpLoopConnectedKFs.end(); vit++)
    {
        KeyFrame* pKF = *vit;
        vector<MapPoint*> vpMapPoints = pKF->GetMapPointMatches();

		// 遍历其中一个关键帧的所有有效地图点
        for(size_t i=0, iend=vpMapPoints.size(); i<iend; i++)
        {
            MapPoint* pMP = vpMapPoints[i];
            if(pMP)
            {
				// mnLoopPointForKF 用于标记，避免重复添加
                if(!pMP->isBad() && pMP->mnLoopPointForKF!=mpCurrentKF->mnId)
                {
                    mvpLoopMapPoints.push_back(pMP);
					//标记一下
                    pMP->mnLoopPointForKF=mpCurrentKF->mnId;
                }
            }
        }
    }

    // Find more matches projecting with the computed Sim3
	// Step 4：将闭环关键帧及其连接关键帧的所有地图点投影到当前关键帧进行投影匹配
	// 根据投影查找更多的匹配（成功的闭环匹配需要满足足够多的匹配特征点数）
	// 根据Sim3变换，将每个mvpLoopMapPoints投影到mpCurrentKF上，搜索新的匹配对
	// mvpCurrentMatchedPoints是前面经过SearchBySim3得到的已经匹配的点对，这里就忽略不再匹配了
	// 搜索范围系数为10
    matcher.SearchByProjection(mpCurrentKF, mScw, mvpLoopMapPoints, mvpCurrentMatchedPoints,10);//统计当前帧与闭环关键帧的匹配地图点数目，超过40个说明成功闭环，否则失败。如果当前回环可靠,保留当前待闭环关键帧，其他闭环候选全部删掉以后不用了。也就是说，所有的闭环候选关键帧中，到最后，只留下了一个。

    // If enough matches accept Loop
	// Step 5: 统计当前帧与闭环关键帧的匹配地图点数目，超过40个说明成功闭环，否则失败
    int nTotalMatches = 0;
    for(size_t i=0; i<mvpCurrentMatchedPoints.size(); i++)
    {
        if(mvpCurrentMatchedPoints[i])
            nTotalMatches++;
    }

    if(nTotalMatches>=40)
    {
        for(int i=0; i<nInitialCandidates; i++)
            if(mvpEnoughConsistentCandidates[i]!=mpMatchedKF)
                mvpEnoughConsistentCandidates[i]->SetErase();
        return true;
    }
    else
    {
		// 闭环不可靠，闭环候选及当前待闭环帧全部删除
        for(int i=0; i<nInitialCandidates; i++)
            mvpEnoughConsistentCandidates[i]->SetErase();
        mpCurrentKF->SetErase();
        return false;
    }

}

/**
 * @brief 闭环矫正
 * 1. 通过求解的Sim3以及相对姿态关系，调整与当前帧相连的关键帧位姿以及这些关键帧观测到的地图点位置（相连关键帧---当前帧）
 * 2. 将闭环帧以及闭环帧相连的关键帧的地图点和与当前帧相连的关键帧的点进行匹配（当前帧+相连关键帧---闭环帧+相连关键帧）
 * 3. 通过MapPoints的匹配关系更新这些帧之间的连接关系，即更新covisibility graph
 * 4. 对Essential Graph（Pose Graph）进行优化，MapPoints的位置则根据优化后的位姿做相对应的调整
 * 5. 创建线程进行全局Bundle Adjustment
 */
void LoopClosing::CorrectLoop()
{

	cout << "Loop detected!" << endl;
	// Step 0：结束局部地图线程、全局BA，为闭环矫正做准备
	// Step 1：根据共视关系更新当前帧与其它关键帧之间的连接
	// Step 2：通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的MapPoints
	// Step 3：检查当前帧的MapPoints与闭环匹配帧的MapPoints是否存在冲突，对冲突的MapPoints进行替换或填补
	// Step 4：通过将闭环时相连关键帧的mvpLoopMapPoints投影到这些关键帧中，进行MapPoints检查与替换
	// Step 5：更新当前关键帧之间的共视相连关系，得到因闭环时MapPoints融合而新得到的连接关系
	// Step 6：进行EssentialGraph优化，LoopConnections是形成闭环后新生成的连接关系，不包括步骤7中当前帧与闭环匹配帧之间的连接关系
	// Step 7：添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）
	// Step 8：新建一个线程用于全局BA优化

	// g2oSic： 当前关键帧 mpCurrentKF 到其共视关键帧 pKFi 的Sim3 相对变换
	// mg2oScw: 世界坐标系到当前关键帧的 Sim3 变换
	// g2oCorrectedSiw：世界坐标系到当前关键帧共视关键帧的Sim3 变换

	// Send a stop signal to Local Mapping
	// Avoid new keyframes are inserted while correcting the loop
	// Step 0：结束局部地图线程、全局BA，为闭环矫正做准备
	// 请求局部地图停止，防止在回环矫正时局部地图线程中InsertKeyFrame函数插入新的关键帧
	mpLocalMapper->RequestStop();

	if (isRunningGBA())
	{
		// 如果有全局BA在运行，终止掉，迎接新的全局BA
		unique_lock<mutex> lock(mMutexGBA);
		mbStopGBA = true;
		// 记录全局BA次数
		mnFullBAIdx++;
		if (mpThreadGBA)
		{
			// 停止全局BA线程
			mpThreadGBA->detach();
			delete mpThreadGBA;
		}
	}

	// Wait until Local Mapping has effectively stopped
	// 一直等到局部地图线程结束再继续
	while (!mpLocalMapper->isStopped())
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	// Ensure current keyframe is updated
	// Step 1：根据共视关系更新当前关键帧与其它关键帧之间的连接关系
	// 因为之前闭环检测、计算Sim3中改变了该关键帧的地图点，所以需要更新
	mpCurrentKF->UpdateConnections();

	// Retrive keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
	// Step 2：通过位姿传播，得到Sim3优化后，与当前帧相连的关键帧的位姿，以及它们的地图点
	// 当前帧与世界坐标系之间的Sim变换在ComputeSim3函数中已经确定并优化，
	// 通过相对位姿关系，可以确定这些相连的关键帧与世界坐标系之间的Sim3变换

	// 取出当前关键帧及其共视关键帧，称为“当前关键帧组”
	mvpCurrentConnectedKFs = mpCurrentKF->GetVectorCovisibleKeyFrames();
	mvpCurrentConnectedKFs.push_back(mpCurrentKF);

	// CorrectedSim3：存放闭环g2o优化后当前关键帧的共视关键帧的世界坐标系下Sim3 变换
	// NonCorrectedSim3：存放没有矫正的当前关键帧的共视关键帧的世界坐标系下Sim3 变换
	KeyFrameAndPose CorrectedSim3, NonCorrectedSim3;
	// 先将mpCurrentKF的Sim3变换存入，认为是准的，所以固定不动
	CorrectedSim3[mpCurrentKF] = mg2oScw;
	// 当前关键帧到世界坐标系下的变换矩阵
	cv::Mat Twc = mpCurrentKF->GetPoseInverse();

	// 对地图点操作
	{
		// Get Map Mutex
		// 锁定地图点
		unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

		// Step 2.1：通过mg2oScw（认为是准的）来进行位姿传播，得到当前关键帧的共视关键帧的世界坐标系下Sim3 位姿
		// 遍历"当前关键帧组""
		for (vector<KeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
		{
			KeyFrame* pKFi = *vit;
			cv::Mat Tiw = pKFi->GetPose();
			if (pKFi != mpCurrentKF)      //跳过当前关键帧，因为当前关键帧的位姿已经在前面优化过了，在这里是参考基准
			{
				// 得到当前关键帧 mpCurrentKF 到其共视关键帧 pKFi 的相对变换
				cv::Mat Tic = Tiw * Twc;
				cv::Mat Ric = Tic.rowRange(0, 3).colRange(0, 3);
				cv::Mat tic = Tic.rowRange(0, 3).col(3);

				// g2oSic：当前关键帧 mpCurrentKF 到其共视关键帧 pKFi 的Sim3 相对变换
				// 这里是non-correct, 所以scale=1.0
				g2o::Sim3 g2oSic(Converter::toMatrix3d(Ric), Converter::toVector3d(tic), 1.0);
				// 当前帧的位姿固定不动，其它的关键帧根据相对关系得到Sim3调整的位姿
				g2o::Sim3 g2oCorrectedSiw = g2oSic * mg2oScw;
				// Pose corrected with the Sim3 of the loop closure
				// 存放闭环g2o优化后当前关键帧的共视关键帧的Sim3 位姿
				CorrectedSim3[pKFi] = g2oCorrectedSiw;
			}

			cv::Mat Riw = Tiw.rowRange(0, 3).colRange(0, 3);
			cv::Mat tiw = Tiw.rowRange(0, 3).col(3);
			g2o::Sim3 g2oSiw(Converter::toMatrix3d(Riw), Converter::toVector3d(tiw), 1.0);
			// Pose without correction
			// 存放没有矫正的当前关键帧的共视关键帧的Sim3变换
			NonCorrectedSim3[pKFi] = g2oSiw;
		}

		// Correct all MapPoints obsrved by current keyframe and neighbors, so that they align with the other side of the loop
		// Step 2.2：得到矫正的当前关键帧的共视关键帧位姿后，修正这些共视关键帧的地图点
		// 遍历待矫正的共视关键帧（不包括当前关键帧）
		for (KeyFrameAndPose::iterator mit = CorrectedSim3.begin(), mend = CorrectedSim3.end(); mit != mend; mit++)
		{
			// 取出当前关键帧连接关键帧
			KeyFrame* pKFi = mit->first;
			// 取出经过位姿传播后的Sim3变换
			g2o::Sim3 g2oCorrectedSiw = mit->second;
			g2o::Sim3 g2oCorrectedSwi = g2oCorrectedSiw.inverse();
			// 取出未经过位姿传播的Sim3变换
			g2o::Sim3 g2oSiw = NonCorrectedSim3[pKFi];

			vector<MapPoint*> vpMPsi = pKFi->GetMapPointMatches();
			// 遍历待矫正共视关键帧中的每一个地图点
			for (size_t iMP = 0, endMPi = vpMPsi.size(); iMP < endMPi; iMP++)
			{
				MapPoint* pMPi = vpMPsi[iMP];
				// 跳过无效的地图点
				if (!pMPi)
					continue;
				if (pMPi->isBad())
					continue;
				// 标记，防止重复矫正
				if (pMPi->mnCorrectedByKF == mpCurrentKF->mnId)
					continue;

				// 矫正过程本质上也是基于当前关键帧的优化后的位姿展开的
				// Project with non-corrected pose and project back with corrected pose
				// 将该未校正的eigP3Dw先从世界坐标系映射到未校正的pKFi相机坐标系，然后再反映射到校正后的世界坐标系下
				cv::Mat P3Dw = pMPi->GetWorldPos();
				// 地图点世界坐标系下坐标
				Eigen::Matrix<double, 3, 1> eigP3Dw = Converter::toVector3d(P3Dw);
				// map(P) 内部做了相似变换 s*R*P +t  
				// 下面变换是：eigP3Dw： world →g2oSiw→ i →g2oCorrectedSwi→ world
				Eigen::Matrix<double, 3, 1> eigCorrectedP3Dw = g2oCorrectedSwi.map(g2oSiw.map(eigP3Dw));

				cv::Mat cvCorrectedP3Dw = Converter::toCvMat(eigCorrectedP3Dw);
				pMPi->SetWorldPos(cvCorrectedP3Dw);
				// 记录矫正该地图点的关键帧id，防止重复
				pMPi->mnCorrectedByKF = mpCurrentKF->mnId;
				// 记录该地图点所在的关键帧id
				pMPi->mnCorrectedReference = pKFi->mnId;
				// 因为地图点更新了，需要更新其平均观测方向以及观测距离范围
				pMPi->UpdateNormalAndDepth();
			}

			// Update keyframe pose with corrected Sim3. First transform Sim3 to SE3 (scale translation)
			// Step 2.3：将共视关键帧的Sim3转换为SE3，根据更新的Sim3，更新关键帧的位姿
			// 其实是现在已经有了更新后的关键帧组中关键帧的位姿,但是在上面的操作时只是暂时存储到了 KeyFrameAndPose 类型的变量中,还没有写回到关键帧对象中
			// 调用toRotationMatrix 可以自动归一化旋转矩阵
			Eigen::Matrix3d eigR = g2oCorrectedSiw.rotation().toRotationMatrix();
			Eigen::Vector3d eigt = g2oCorrectedSiw.translation();
			double s = g2oCorrectedSiw.scale();
			// 平移向量中包含有尺度信息，还需要用尺度归一化
			eigt *= (1. / s);

			/*
			这段代码使用了一个名为Converter的类的静态成员函数toCvSE3()，用于将旋转矩阵eigR和平移向量eigt转换为一个SE(3)类型的变换矩阵。

函数toCvSE3()可能是在Converter类中定义的一个静态成员函数。它接受两个参数：旋转矩阵eigR和平移向量eigt，并返回一个cv::Mat类型的变量correctedTiw。

根据函数名toCvSE3()，这个函数的目的是将旋转矩阵和平移向量转换为一个SE(3)类型的变换矩阵correctedTiw，即从相机坐标系到世界坐标系的变换矩阵。

通过这段代码，我们可以得到一个SE(3)类型的变换矩阵，用于将点从相机坐标系转换到世界坐标系。
			*/
			cv::Mat correctedTiw = Converter::toCvSE3(eigR, eigt);
			// 设置矫正后的新的pose
			pKFi->SetPose(correctedTiw);

			// Make sure connections are updated
			// Step 2.4：根据共视关系更新当前帧与其它关键帧之间的连接
			// 地图点的位置改变了,可能会引起共视关系\权值的改变 
			pKFi->UpdateConnections();
		}

		// Start Loop Fusion
		// Update matched map points and replace if duplicated
		// Step 3：检查当前帧的地图点与经过闭环匹配后该帧的地图点是否存在冲突，对冲突的进行替换或填补
		// mvpCurrentMatchedPoints 是当前关键帧和闭环关键帧组的所有地图点进行投影得到的匹配点
		for (size_t i = 0; i < mvpCurrentMatchedPoints.size(); i++)
		{
			if (mvpCurrentMatchedPoints[i])
			{
				//取出同一个索引对应的两种地图点，决定是否要替换
				// 匹配投影得到的地图点
				MapPoint* pLoopMP = mvpCurrentMatchedPoints[i];
				// 原来的地图点
				MapPoint* pCurMP = mpCurrentKF->GetMapPoint(i);
				if (pCurMP)
					// 如果有重复的MapPoint，则用匹配的地图点代替现有的
					// 因为匹配的地图点是经过一系列操作后比较精确的，现有的地图点很可能有累计误差
					pCurMP->Replace(pLoopMP);
				else
				{
					// 如果当前帧没有该MapPoint，则直接添加
					mpCurrentKF->AddMapPoint(pLoopMP, i);
					pLoopMP->AddObservation(mpCurrentKF, i);
					pLoopMP->ComputeDistinctiveDescriptors();
				}
			}
		}

	}

	// Project MapPoints observed in the neighborhood of the loop keyframe
	// into the current keyframe and neighbors using corrected poses.
	// Fuse duplications.
	// Step 4：将闭环相连关键帧组mvpLoopMapPoints 投影到当前关键帧组中，进行匹配，融合，新增或替换当前关键帧组中KF的地图点
	// 因为 闭环相连关键帧组mvpLoopMapPoints 在地图中时间比较久经历了多次优化，认为是准确的
	// 而当前关键帧组中的关键帧的地图点是最近新计算的，可能有累积误差
	// CorrectedSim3：存放矫正后当前关键帧的共视关键帧，及其世界坐标系下Sim3 变换
	SearchAndFuse(CorrectedSim3);


	// After the MapPoint fusion, new links in the covisibility graph will appear attaching both sides of the loop
	// Step 5：更新当前关键帧组之间的两级共视相连关系，得到因闭环时地图点融合而新得到的连接关系
	// LoopConnections：存储因为闭环地图点调整而新生成的连接关系
	map<KeyFrame*, set<KeyFrame*> > LoopConnections;

	// Step 5.1：遍历当前帧相连关键帧组（一级相连）
	for (vector<KeyFrame*>::iterator vit = mvpCurrentConnectedKFs.begin(), vend = mvpCurrentConnectedKFs.end(); vit != vend; vit++)
	{
		KeyFrame* pKFi = *vit;
		// Step 5.2：得到与当前帧相连关键帧的相连关键帧（二级相连）
		vector<KeyFrame*> vpPreviousNeighbors = pKFi->GetVectorCovisibleKeyFrames();

		// Update connections. Detect new links.
		// Step 5.3：更新一级相连关键帧的连接关系(会把当前关键帧添加进去,因为地图点已经更新和替换了)
		pKFi->UpdateConnections();
		// Step 5.4：取出该帧更新后的连接关系
		LoopConnections[pKFi] = pKFi->GetConnectedKeyFrames();
		// Step 5.5：从连接关系中去除闭环之前的二级连接关系，剩下的连接就是由闭环得到的连接关系
		for (vector<KeyFrame*>::iterator vit_prev = vpPreviousNeighbors.begin(), vend_prev = vpPreviousNeighbors.end(); vit_prev != vend_prev; vit_prev++)
		{
			LoopConnections[pKFi].erase(*vit_prev);
		}
		// Step 5.6：从连接关系中去除闭环之前的一级连接关系，剩下的连接就是由闭环得到的连接关系
		for (vector<KeyFrame*>::iterator vit2 = mvpCurrentConnectedKFs.begin(), vend2 = mvpCurrentConnectedKFs.end(); vit2 != vend2; vit2++)
		{
			LoopConnections[pKFi].erase(*vit2);
		}
	}

	// Optimize graph
	// Step 6：进行本质图优化，优化本质图中所有关键帧的位姿和地图点
	// LoopConnections是形成闭环后新生成的连接关系，不包括步骤7中当前帧与闭环匹配帧之间的连接关系
	Optimizer::OptimizeEssentialGraph(mpMap, mpMatchedKF, mpCurrentKF, NonCorrectedSim3, CorrectedSim3, LoopConnections, mbFixScale);

	// Add loop edge
	// Step 7：添加当前帧与闭环匹配帧之间的边（这个连接关系不优化）
	// !这两句话应该放在OptimizeEssentialGraph之前，因为OptimizeEssentialGraph的步骤4.2中有优化
	mpMatchedKF->AddLoopEdge(mpCurrentKF);
	mpCurrentKF->AddLoopEdge(mpMatchedKF);

	// Launch a new thread to perform Global Bundle Adjustment
	// Step 8：新建一个线程用于全局BA优化
	// OptimizeEssentialGraph只是优化了一些主要关键帧的位姿，这里进行全局BA可以全局优化所有位姿和MapPoints
	mbRunningGBA = true;
	mbFinishedGBA = false;
	mbStopGBA = false;
	mpThreadGBA = new thread(&LoopClosing::RunGlobalBundleAdjustment, this, mpCurrentKF->mnId);

	// Loop closed. Release Local Mapping.
	mpLocalMapper->Release();

	cout << "Loop Closed!" << endl;

	mLastLoopKFid = mpCurrentKF->mnId;
}

void LoopClosing::SearchAndFuse(const KeyFrameAndPose &CorrectedPosesMap)
{
    ORBmatcher matcher(0.8);

    for(KeyFrameAndPose::const_iterator mit=CorrectedPosesMap.begin(), mend=CorrectedPosesMap.end(); mit!=mend;mit++)
    {
        KeyFrame* pKF = mit->first;

        g2o::Sim3 g2oScw = mit->second;
        cv::Mat cvScw = Converter::toCvMat(g2oScw);

        vector<MapPoint*> vpReplacePoints(mvpLoopMapPoints.size(),static_cast<MapPoint*>(NULL));
        matcher.Fuse(pKF,cvScw,mvpLoopMapPoints,4,vpReplacePoints);

        // Get Map Mutex
        unique_lock<mutex> lock(mpMap->mMutexMapUpdate);
        const int nLP = mvpLoopMapPoints.size();
        for(int i=0; i<nLP;i++)
        {
            MapPoint* pRep = vpReplacePoints[i];
            if(pRep)
            {
                pRep->Replace(mvpLoopMapPoints[i]);
            }
        }
    }
}


void LoopClosing::RequestReset()
{
    {
        unique_lock<mutex> lock(mMutexReset);
        mbResetRequested = true;
    }

    while(1)
    {
        {
        unique_lock<mutex> lock2(mMutexReset);
        if(!mbResetRequested)
            break;
        }
        usleep(5000);
    }
}

void LoopClosing::ResetIfRequested()
{
    unique_lock<mutex> lock(mMutexReset);
    if(mbResetRequested)
    {
        mlpLoopKeyFrameQueue.clear();
        mLastLoopKFid=0;
        mbResetRequested=false;
    }
}

/**
 * @brief 全局BA线程,这个是这个线程的主函数
 *
 * @param[in] nLoopKF 看上去是闭环关键帧id,但是在调用的时候给的其实是当前关键帧的id
 */
void LoopClosing::RunGlobalBundleAdjustment(unsigned long nLoopKF)
{
	cout << "Starting Global Bundle Adjustment" << endl;

	// 记录GBA已经迭代次数,用来检查全局BA过程是否是因为意外结束的
	int idx = mnFullBAIdx;
	// mbStopGBA直接传引用过去了,这样当有外部请求的时候这个优化函数能够及时响应并且结束掉
	// 提问:进行完这个过程后我们能够获得哪些信息?
	// 回答：能够得到全部关键帧优化后的位姿,以及优化后的地图点

	// Step 1 执行全局BA，优化所有的关键帧位姿和地图中地图点
	Optimizer::GlobalBundleAdjustemnt(mpMap,        // 地图点对象
		10,           // 迭代次数
		&mbStopGBA,   // 外界控制 GBA 停止的标志
		nLoopKF,      // 形成了闭环的当前关键帧的id
		false);       // 不使用鲁棒核函数

// Update all MapPoints and KeyFrames
// Local Mapping was active during BA, that means that there might be new keyframes
// not included in the Global BA and they are not consistent with the updated map.
// We need to propagate the correction through the spanning tree
// 更新所有的地图点和关键帧
// 在global BA过程中local mapping线程仍然在工作，这意味着在global BA时可能有新的关键帧产生，但是并未包括在GBA里，
// 所以和更新后的地图并不连续。需要通过spanning tree来传播
	{
		unique_lock<mutex> lock(mMutexGBA);//上锁，保持独占
		// 如果全局BA过程是因为意外结束的,那么直接退出GBA
		if (idx != mnFullBAIdx)
			return;

		// 如果当前GBA没有中断请求，更新位姿和地图点
		// 这里和上面那句话的功能还有些不同,因为如果一次全局优化被中断,往往意味又要重新开启一个新的全局BA;为了中断当前正在执行的优化过程mbStopGBA将会被置位,同时会有一定的时间
		// 使得该线程进行响应;而在开启一个新的全局优化进程之前 mbStopGBA 将会被置为False
		// 因此,如果被强行中断的线程退出时已经有新的线程启动了,mbStopGBA=false,为了避免进行后面的程序,所以有了上面的程序;
		// 而如果被强行中断的线程退出时新的线程还没有启动,那么上面的条件就不起作用了(虽然概率很小,前面的程序中mbStopGBA置位后很快mnFullBAIdx就++了,保险起见),所以这里要再判断一次
		if (!mbStopGBA)
		{
			cout << "Global Bundle Adjustment finished" << endl;
			cout << "Updating map ..." << endl;
			mpLocalMapper->RequestStop();

			// Wait until Local Mapping has effectively stopped
			// 等待直到local mapping结束才会继续后续操作
			while (!mpLocalMapper->isStopped() && !mpLocalMapper->isFinished())
			{
				//usleep(1000);
				std::this_thread::sleep_for(std::chrono::milliseconds(1));//休眠1毫秒
			}

			// Get Map Mutex
			// 后续要更新地图所以要上锁
			unique_lock<mutex> lock(mpMap->mMutexMapUpdate);

			// Correct keyframes starting at map first keyframe
			// 从第0个关键帧开始矫正关键帧。刚开始只保存了初始化第0个关键帧
			list<KeyFrame*> lpKFtoCheck(mpMap->mvpKeyFrameOrigins.begin(), mpMap->mvpKeyFrameOrigins.end());

			// 问：GBA里锁住第一个关键帧位姿没有优化，其对应的pKF->mTcwGBA是不变的吧？那后面调整位姿的意义何在？
			// 回答：注意在前面essential graph BA里只锁住了回环帧，没有锁定第0个初始化关键帧位姿。所以第0个初始化关键帧位姿已经更新了
			// 在GBA里锁住第一个关键帧位姿没有优化，其对应的pKF->mTcwGBA应该是essential BA结果，在这里统一更新了
			// Step 2 遍历并更新全局地图中的所有spanning tree中的关键帧
			// 对一棵关键帧的最小生成树执行全局大规模优化（GBA），将优化后的位姿应用到所有子关键帧，以使整个最小生成树中的关键帧都受到优化的影响。
			while (!lpKFtoCheck.empty())
			{
				KeyFrame* pKF = lpKFtoCheck.front();
				const set<KeyFrame*> sChilds = pKF->GetChilds();
				cv::Mat Twc = pKF->GetPoseInverse();//即当前关键帧到世界坐标系的变换矩阵
				// 遍历当前关键帧的子关键帧
				for (set<KeyFrame*>::const_iterator sit = sChilds.begin(); sit != sChilds.end(); sit++)
				{
					KeyFrame* pChild = *sit;
					// 记录避免重复
					if (pChild->mnBAGlobalForKF != nLoopKF)//如果子关键帧的 mnBAGlobalForKF 不等于 nLoopKF，表示该子关键帧还没有被处理过。执行以下操作：
					{
						// 从父关键帧到当前子关键帧的位姿变换 T_child_farther
						cv::Mat Tchildc = pChild->GetPose()*Twc;//计算从父关键帧到当前子关键帧的位姿变换矩阵 Tchildc，这是通过将子关键帧的位姿矩阵与父关键帧到世界坐标系的变换矩阵相乘得到的结果。
						// 再利用优化后的父关键帧的位姿，转换到世界坐标系下，相当于更新了子关键帧的位姿
						// 这种最小生成树中除了根节点，其他的节点都会作为其他关键帧的子节点，这样做可以使得最终所有的关键帧都得到了优化
						pChild->mTcwGBA = Tchildc * pKF->mTcwGBA;//将优化后的父关键帧位姿矩阵与 Tchildc 相乘，得到子关键帧在世界坐标系下的位姿矩阵 pChild->mTcwGBA。这样做可以更新子关键帧的位姿，使其受到父关键帧位姿的优化影响。
						// 做个标记，避免重复
						pChild->mnBAGlobalForKF = nLoopKF;//将 nLoopKF 赋值给子关键帧的 mnBAGlobalForKF，标记该子关键帧已被处理过

					}
					lpKFtoCheck.push_back(pChild); //将子关键帧 pChild 添加到队列 lpKFtoCheck 的末尾，以便在下一次循环中继续处理。
				}
				// 记录未矫正的关键帧的位姿
				pKF->mTcwBefGBA = pKF->GetPose();
				// 记录已经矫正的关键帧的位姿
				pKF->SetPose(pKF->mTcwGBA);
				// 从列表中移除
				lpKFtoCheck.pop_front();
			}

			// Correct MapPoints
			const vector<MapPoint*> vpMPs = mpMap->GetAllMapPoints();

			// Step 3 遍历每一个地图点并用更新的关键帧位姿来更新地图点位置
			for (size_t i = 0; i < vpMPs.size(); i++)
			{
				MapPoint* pMP = vpMPs[i];

				if (pMP->isBad())
					continue;

				// 如果这个地图点直接参与到了全局BA优化的过程,那么就直接重新设置新位姿即可
				if (pMP->mnBAGlobalForKF == nLoopKF)
				{
					// If optimized by Global BA, just update，地图点的世界坐标位置将被更新为优化后的位置。
					pMP->SetWorldPos(pMP->mPosGBA);
				}
				else
				{
					// 如这个地图点并没有直接参与到全局BA优化的过程中,那么就使用其参考关键帧的新位姿来优化自己的坐标
					// Update according to the correction of its reference keyframe
					KeyFrame* pRefKF = pMP->GetReferenceKeyFrame();

					// 如果参考关键帧并没有经过此次全局BA优化，就跳过 
					if (pRefKF->mnBAGlobalForKF != nLoopKF)
						continue;

					// Map to non-corrected camera
					cv::Mat Rcw = pRefKF->mTcwBefGBA.rowRange(0, 3).colRange(0, 3);
					cv::Mat tcw = pRefKF->mTcwBefGBA.rowRange(0, 3).col(3);
					// 转换到其参考关键帧相机坐标系下的坐标
					cv::Mat Xc = Rcw * pMP->GetWorldPos() + tcw;

					// Backproject using corrected camera
					// 然后使用已经纠正过的参考关键帧的位姿,再将该地图点变换到世界坐标系下
					cv::Mat Twc = pRefKF->GetPoseInverse();
					cv::Mat Rwc = Twc.rowRange(0, 3).colRange(0, 3);
					cv::Mat twc = Twc.rowRange(0, 3).col(3);

					pMP->SetWorldPos(Rwc*Xc + twc);
				}
			}

			// 释放
			mpLocalMapper->Release();//mpLocalMapper 是指向某个局部地图生成器（Local Mapper）对象的指针。该代码调用 Release() 方法来释放局部地图生成器的资源。

			//  Release() 方法一般用于释放局部地图生成器的内部数据结构、缓存或其他资源。这样可以帮助优化内存使用并提高系统的效率。具体的释放操作可能因局部地图生成器的具体实现而有所不同，但通常会包括清空地图、释放特征点、重置状态等。

			//	释放局部地图生成器可能会导致当前地图的丢失，因此需要根据具体的应用场景和需求进行决策。如果需要重新开始建图，可能需要重新初始化局部地图生成器并开始新的建图过程。

			cout << "Map updated!" << endl;
		}

		mbFinishedGBA = true;
		mbRunningGBA = false;
	}
}


void LoopClosing::RequestFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinishRequested = true;
}

bool LoopClosing::CheckFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinishRequested;
}

void LoopClosing::SetFinish()
{
    unique_lock<mutex> lock(mMutexFinish);
    mbFinished = true;
}

bool LoopClosing::isFinished()
{
    unique_lock<mutex> lock(mMutexFinish);
    return mbFinished;
}


} //namespace ORB_SLAM
