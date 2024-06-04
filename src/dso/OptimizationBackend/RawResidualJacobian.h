/**
* This file is part of DSO.
*
* Copyright 2016 Technical University of Munich and Intel.
* Developed by Jakob Engel <engelj at in dot tum dot de>,
* for more information see <http://vision.in.tum.de/dso>.
* If you use this code, please cite the respective publications as
* listed on the above website.
*
* DSO is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* DSO is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with DSO. If not, see <http://www.gnu.org/licenses/>.
*/


#pragma once

 
#include "util/NumType.h"

namespace dso
{
struct RawResidualJacobian
{
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
	// ================== new structure: save independently =============.
	// 8x1 对应residual_21，1个点的8个pattern residual
	VecNRf resF;

	// the two rows of d[x,y]/d[xi].
	// target帧中像素坐标(u, v)_2x1关于host到target间相对位姿(\xi_21)_6x1的梯度_2x6
	Vec6f Jpdxi[2];			// 2x6

	// the two rows of d[x,y]/d[C].
	// 像素坐标(u, v)_2x1关于相机内参(fx, fy, cx, cy)_4x1的梯度_2x4
	VecCf Jpdc[2];			// 2x4

	// the two rows of d[x,y]/d[idepth].
	// 像素坐标(u, v)_2x1关于host帧逆深度_1x1的梯度_2x1
	Vec2f Jpdd;				// 2x1

	// the two columns of d[r]/d[x,y].
	// 残差关于像素坐标(u, v)_2x1的梯度
	VecNRf JIdx[2];			// 9x2 -->实际为 8x2

	// = the two columns of d[r] / d[ab]
	// 残差关于帧间光度参数(a, b)_2x1的梯度
	VecNRf JabF[2];			// 9x2 -->实际为 8x2


	// = JIdx^T * JIdx (inner product). Only as a shorthand.
	Mat22f JIdx2;				// 2x2
	// = Jab^T * JIdx (inner product). Only as a shorthand.
	Mat22f JabJIdx;			// 2x2
	// = Jab^T * Jab (inner product). Only as a shorthand.
	Mat22f Jab2;			// 2x2

};
}

