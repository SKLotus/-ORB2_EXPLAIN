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

#ifndef CONVERTER_H
#define CONVERTER_H

#include<opencv2/core/core.hpp>

#include<Eigen/Dense>
#include"Thirdparty/g2o/g2o/types/types_six_dof_expmap.h"
#include"Thirdparty/g2o/g2o/types/types_seven_dof_expmap.h"

namespace ORB_SLAM2
{

class Converter
{
public:
    static std::vector<cv::Mat> toDescriptorVector(const cv::Mat &Descriptors);//第一个方法toDescriptorVector获取一个表示一组特征描述符的cv:：Mat对象，并返回一个cv:：Mat对象的向量，每个对象表示一个描述符。

    static g2o::SE3Quat toSE3Quat(const cv::Mat &cvT);
    static g2o::SE3Quat toSE3Quat(const g2o::Sim3 &gSim3);//toSE3Quat，在3D姿势的不同表示之间进行转换。第一种方法采用表示4x4变换矩阵的cv:：Mat对象，并返回表示相同姿势的g2o:：SE3Quat对象。第二种方法采用表示相似性变换的g2o:：Sim3对象，并返回表示相同姿势的g2o：：SE3Quat对象。

    static cv::Mat toCvMat(const g2o::SE3Quat &SE3);
    static cv::Mat toCvMat(const g2o::Sim3 &Sim3);
    static cv::Mat toCvMat(const Eigen::Matrix<double,4,4> &m);//toCvMat在特征矩阵和cv:：Mat对象之间进行转换。第一个方法获取一个g2o:：SE3Quat对象，并返回一个表示相同变换矩阵的cv:：Mat对象。第二个方法获取一个g2o:：Sim3对象，并返回一个表示相同变换矩阵的cv:：Mat对象。第三种方法采用4x4特征矩阵，并返回表示相同矩阵的cv:：Mat对象。
    static cv::Mat toCvMat(const Eigen::Matrix3d &m);
    static cv::Mat toCvMat(const Eigen::Matrix<double,3,1> &m);
    static cv::Mat toCvSE3(const Eigen::Matrix<double,3,3> &R, const Eigen::Matrix<double,3,1> &t);//toCvMat在不同的特征矩阵类型和cv:：Mat对象之间进行转换。第一种方法采用3x3的特征矩阵，并返回表示相同矩阵的cv:：Mat对象。第二种方法采用3x1的特征矩阵，并返回表示相同向量的cv:：Mat对象。第三种方法采用3x3旋转矩阵和3x1平移向量，并返回表示相同4x4变换矩阵的cv:：Mat对象。

    static Eigen::Matrix<double,3,1> toVector3d(const cv::Mat &cvVector);
    static Eigen::Matrix<double,3,1> toVector3d(const cv::Point3f &cvPoint);
    static Eigen::Matrix<double,3,3> toMatrix3d(const cv::Mat &cvMat3);//toVector3d和toMatrix3d，在cv:：Mat对象和特征矩阵类型之间转换。第一种方法采用表示3x1向量的cv:：Mat对象，并返回表示相同向量的3x1特征矩阵。第二种方法采用表示3x3矩阵的cv:：Mat对象，并返回表示相同矩阵的3x3本征矩阵。

    static std::vector<float> toQuaternion(const cv::Mat &M);
};

}// namespace ORB_SLAM

#endif // CONVERTER_H
