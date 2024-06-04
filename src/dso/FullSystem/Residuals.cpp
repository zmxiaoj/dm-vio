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


/*
 * KFBuffer.cpp
 *
 *  Created on: Jan 7, 2014
 *      Author: engelj
 */

#include "FullSystem/FullSystem.h"

#include "stdio.h"
#include "util/globalFuncs.h"
#include <Eigen/LU>
#include <algorithm>
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include <Eigen/SVD>
#include <Eigen/Eigenvalues>

#include "FullSystem/ResidualProjections.h"
#include "OptimizationBackend/EnergyFunctional.h"
#include "OptimizationBackend/EnergyFunctionalStructs.h"

#include "FullSystem/HessianBlocks.h"

namespace dso
{
int PointFrameResidual::instanceCounter = 0;


long runningResID=0;


PointFrameResidual::PointFrameResidual(){assert(false); instanceCounter++;}

PointFrameResidual::~PointFrameResidual(){assert(efResidual==0); instanceCounter--; delete J;}

PointFrameResidual::PointFrameResidual(PointHessian* point_, FrameHessian* host_, FrameHessian* target_) :
	point(point_),
	host(host_),
	target(target_)
{
	efResidual=0;
	instanceCounter++;
	resetOOB();
	J = new RawResidualJacobian();
	assert(((long)J)%16==0);

	isNew=true;
}



/**
 * @brief 求解能量值及残差关于各个参数的梯度
 * 
 * @param HCalib [in] 标定参数
 * @return double 
 */
double PointFrameResidual::linearize(CalibHessian* HCalib)
{
	state_NewEnergyWithOutlier=-1;

	if(state_state == ResState::OOB)
		{ state_NewState = ResState::OOB; return state_energy; }

	// 取出host(主导帧)到target(目标帧)间的相对位姿变换
	FrameFramePrecalc* precalc = &(host->targetPrecalc[target->idx]);
	float energyLeft=0;
	const Eigen::Vector3f* dIl = target->dI;
	//const float* const Il = target->I;
	const Mat33f &PRE_KRKiTll = precalc->PRE_KRKiTll;
	const Vec3f &PRE_KtTll = precalc->PRE_KtTll;
	const Mat33f &PRE_RTll_0 = precalc->PRE_RTll_0;
	const Vec3f &PRE_tTll_0 = precalc->PRE_tTll_0;
	const float * const color = point->color;
	const float * const weights = point->weights;

	Vec2f affLL = precalc->PRE_aff_mode;
	float b0 = precalc->PRE_b0_mode;

	// 位姿关于像素的导数(6维)
	Vec6f d_xi_x, d_xi_y;
	// 相机内参关于像素的导数(4维)
	Vec4f d_C_x, d_C_y;
	float d_d_x, d_d_y;
	{
		// target帧逆深度与host帧逆深度比值
		float drescale, u, v, new_idepth;
		float Ku, Kv;
		Vec3f KliP;

		// 检验host帧上的点能否投影到target帧上
		// (u, v)投影点的 (Ku, Kv)投影点的像素坐标
		if(!projectPoint(point->u, point->v, point->idepth_zero_scaled, 0, 0,HCalib,
				PRE_RTll_0,PRE_tTll_0, drescale, u, v, Ku, Kv, KliP, new_idepth))
			{ state_NewState = ResState::OOB; return state_energy; }

		// 投影后坐标及逆深度
		centerProjectedTo = Vec3f(Ku, Kv, new_idepth);


		// diff d_idepth
		// 归一化坐标(u, v)关于host帧逆深度的梯度
		d_d_x = drescale * (PRE_tTll_0[0]-PRE_tTll_0[2]*u)*SCALE_IDEPTH*HCalib->fxl();
		d_d_y = drescale * (PRE_tTll_0[1]-PRE_tTll_0[2]*v)*SCALE_IDEPTH*HCalib->fyl();




		// diff calib
		// 归一化坐标(u, v)关于相机内参的梯度
		d_C_x[2] = drescale*(PRE_RTll_0(2,0)*u-PRE_RTll_0(0,0));
		d_C_x[3] = HCalib->fxl() * drescale*(PRE_RTll_0(2,1)*u-PRE_RTll_0(0,1)) * HCalib->fyli();
		d_C_x[0] = KliP[0]*d_C_x[2];
		d_C_x[1] = KliP[1]*d_C_x[3];

		d_C_y[2] = HCalib->fyl() * drescale*(PRE_RTll_0(2,0)*v-PRE_RTll_0(1,0)) * HCalib->fxli();
		d_C_y[3] = drescale*(PRE_RTll_0(2,1)*v-PRE_RTll_0(1,1));
		d_C_y[0] = KliP[0]*d_C_y[2];
		d_C_y[1] = KliP[1]*d_C_y[3];

		d_C_x[0] = (d_C_x[0]+u)*SCALE_F;
		d_C_x[1] *= SCALE_F;
		d_C_x[2] = (d_C_x[2]+1)*SCALE_C;
		d_C_x[3] *= SCALE_C;

		d_C_y[0] *= SCALE_F;
		d_C_y[1] = (d_C_y[1]+v)*SCALE_F;
		d_C_y[2] *= SCALE_C;
		d_C_y[3] = (d_C_y[3]+1)*SCALE_C;


		// 归一化坐标关于位姿(xyzrpy)的梯度
		d_xi_x[0] = new_idepth*HCalib->fxl();
		d_xi_x[1] = 0;
		d_xi_x[2] = -new_idepth*u*HCalib->fxl();
		d_xi_x[3] = -u*v*HCalib->fxl();
		d_xi_x[4] = (1+u*u)*HCalib->fxl();
		d_xi_x[5] = -v*HCalib->fxl();

		d_xi_y[0] = 0;
		d_xi_y[1] = new_idepth*HCalib->fyl();
		d_xi_y[2] = -new_idepth*v*HCalib->fyl();
		d_xi_y[3] = -(1+v*v)*HCalib->fyl();
		d_xi_y[4] = u*v*HCalib->fyl();
		d_xi_y[5] = u*HCalib->fyl();
	}


	{
		// 像素坐标(u, v)_2x1关于host到target间相对位姿(\xi_21)_6x1的梯度_2x6
		J->Jpdxi[0] = d_xi_x;
		J->Jpdxi[1] = d_xi_y;

		// 像素坐标(u, v)_2x1关于相机内参(fx, fy, cx, cy)_4x1的梯度_2x4
		J->Jpdc[0] = d_C_x;
		J->Jpdc[1] = d_C_y;

		// 像素坐标(u, v)_2x1关于host帧逆深度_1x1的梯度_2x1
		J->Jpdd[0] = d_d_x;
		J->Jpdd[1] = d_d_y;

	}






	float JIdxJIdx_00=0, JIdxJIdx_11=0, JIdxJIdx_10=0;
	float JabJIdx_00=0, JabJIdx_01=0, JabJIdx_10=0, JabJIdx_11=0;
	float JabJab_00=0, JabJab_01=0, JabJab_11=0;

	float wJI2_sum = 0;

	// 遍历host帧上点的pattern
	for(int idx=0;idx<patternNum;idx++)
	{
		float Ku, Kv;
		if(!projectPoint(point->u+patternP[idx][0], point->v+patternP[idx][1], point->idepth_scaled, PRE_KRKiTll, PRE_KtTll, Ku, Kv))
			{ state_NewState = ResState::OOB; return state_energy; }

		// 投影后的像素坐标
		projectedTo[idx][0] = Ku;
		projectedTo[idx][1] = Kv;


		// 双线性插值得到投影点的像素值, 水平梯度, 垂直梯度
        Vec3f hitColor = (getInterpolatedElement33(dIl, Ku, Kv, wG[0]));
		// 残差
        float residual = hitColor[0] - (float)(affLL[0] * color[idx] + affLL[1]);


		// 残差对于光度仿射变换A的梯度
		float drdA = (color[idx]-b0);
		// 像素值不是有限值，跳过
		if(!std::isfinite((float)hitColor[0]))
		{ state_NewState = ResState::OOB; return state_energy; }


		// 基于图像梯度的权重
		float w = sqrtf(setting_outlierTHSumComponent / (setting_outlierTHSumComponent + hitColor.tail<2>().squaredNorm()));
        // 和patch位置相关的权重
		w = 0.5f*(w + weights[idx]);


		// huber函数
		float hw = fabsf(residual) < setting_huberTH ? 1 : setting_huberTH / fabsf(residual);
		// 计算考虑梯度和pattern权重的能量(残差平方和)
		energyLeft += w*w*hw *residual*residual*(2-hw);

		{
			if(hw < 1) hw = sqrtf(hw);
			hw = hw*w;

			hitColor[1]*=hw;
			hitColor[2]*=hw;

			// 对应idx pattern的残差
			J->resF[idx] = residual*hw;

			// 残差关于idx处像素坐标(u, v)_2x1的梯度
			J->JIdx[0][idx] = hitColor[1];
			J->JIdx[1][idx] = hitColor[2];
			// 残差关于idx处帧间光度参数(a, b)_2x1的梯度
			J->JabF[0][idx] = drdA*hw;
			J->JabF[1][idx] = hw;

			// 计算JIdx^T * JIdx (2x2)
			JIdxJIdx_00+=hitColor[1]*hitColor[1];
			JIdxJIdx_11+=hitColor[2]*hitColor[2];
			JIdxJIdx_10+=hitColor[1]*hitColor[2];

			// 计算Jab^T * JIdx (2x2)
			JabJIdx_00+= drdA*hw * hitColor[1];
			JabJIdx_01+= drdA*hw * hitColor[2];
			JabJIdx_10+= hw * hitColor[1];
			JabJIdx_11+= hw * hitColor[2];

			// 计算Jab^T * Jab (2x2)
			JabJab_00+= drdA*drdA*hw*hw;
			JabJab_01+= drdA*hw*hw;
			JabJab_11+= hw*hw;


			// 累加梯度平方和
			wJI2_sum += hw*hw*(hitColor[1]*hitColor[1]+hitColor[2]*hitColor[2]);

			if(setting_affineOptModeA < 0) J->JabF[0][idx]=0;
			if(setting_affineOptModeB < 0) J->JabF[1][idx]=0;

		}
	}

	J->JIdx2(0,0) = JIdxJIdx_00;
	J->JIdx2(0,1) = JIdxJIdx_10;
	J->JIdx2(1,0) = JIdxJIdx_10;
	J->JIdx2(1,1) = JIdxJIdx_11;
	J->JabJIdx(0,0) = JabJIdx_00;
	J->JabJIdx(0,1) = JabJIdx_01;
	J->JabJIdx(1,0) = JabJIdx_10;
	J->JabJIdx(1,1) = JabJIdx_11;
	J->Jab2(0,0) = JabJab_00;
	J->Jab2(0,1) = JabJab_01;
	J->Jab2(1,0) = JabJab_01;
	J->Jab2(1,1) = JabJab_11;

	state_NewEnergyWithOutlier = energyLeft;

	// 能量大于阈值或者梯度平方和小于2，认为是outlier
	if(energyLeft > std::max<float>(host->frameEnergyTH, target->frameEnergyTH) || wJI2_sum < 2)
	{
		energyLeft = std::max<float>(host->frameEnergyTH, target->frameEnergyTH);
		state_NewState = ResState::OUTLIER;
	}
	else
	{
		state_NewState = ResState::IN;
	}

	state_NewEnergy = energyLeft;
	return energyLeft;
}



void PointFrameResidual::debugPlot()
{
	if(state_state==ResState::OOB) return;
	Vec3b cT = Vec3b(0,0,0);

	if(freeDebugParam5==0)
	{
		float rT = 20*sqrt(state_energy/9);
		if(rT<0) rT=0; if(rT>255)rT=255;
		cT = Vec3b(0,255-rT,rT);
	}
	else
	{
		if(state_state == ResState::IN) cT = Vec3b(255,0,0);
		else if(state_state == ResState::OOB) cT = Vec3b(255,255,0);
		else if(state_state == ResState::OUTLIER) cT = Vec3b(0,0,255);
		else cT = Vec3b(255,255,255);
	}

	for(int i=0;i<patternNum;i++)
	{
		if((projectedTo[i][0] > 2 && projectedTo[i][1] > 2 && projectedTo[i][0] < wG[0]-3 && projectedTo[i][1] < hG[0]-3 ))
			target->debugImage->setPixel1((float)projectedTo[i][0], (float)projectedTo[i][1],cT);
	}
}


/**
 * @brief 更新残差状态，将计算的残差和梯度赋给EFResidual
 * 
 * @param copyJacobians [in] bool
 */
void PointFrameResidual::applyRes(bool copyJacobians)
{
	if(copyJacobians)
	{
		if(state_state == ResState::OOB)
		{
			assert(!efResidual->isActiveAndIsGoodNEW);
			return;	// can never go back from OOB
		}
		// 残差为内点
		if(state_NewState == ResState::IN)// && )
		{
			// 更新残差状态
			efResidual->isActiveAndIsGoodNEW=true;
			// 计算JpJdF
			efResidual->takeDataF();
		}
		else
		{
			efResidual->isActiveAndIsGoodNEW=false;
		}
	}

	// 更新状态
	setState(state_NewState);
	state_energy = state_NewEnergy;
}
}
