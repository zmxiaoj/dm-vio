/**
* This file is part of DSO, written by Jakob Engel.
* It has been modified by Lukas von Stumberg for the inclusion in DM-VIO (http://vision.in.tum.de/dm-vio).
*
* Copyright 2022 Lukas von Stumberg <lukas dot stumberg at tum dot de>
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

#include "FullSystem/CoarseInitializer.h"
#include "FullSystem/FullSystem.h"
#include "FullSystem/HessianBlocks.h"
#include "FullSystem/Residuals.h"
#include "FullSystem/PixelSelector.h"
#include "FullSystem/PixelSelector2.h"
#include "util/nanoflann.h"

#include <opencv2/highgui/highgui.hpp>

#if !defined(__SSE3__) && !defined(__SSE2__) && !defined(__SSE1__)
#include "SSE2NEON.h"
#endif

namespace dso
{

CoarseInitializer::CoarseInitializer(int ww, int hh)
        : thisToNext_aff(0, 0), thisToNext(SE3())
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		points[lvl] = 0;
		numPoints[lvl] = 0;
	}

	JbBuffer = new Vec10f[ww*hh];
	JbBuffer_new = new Vec10f[ww*hh];


	frameID=-1;
	fixAffine=true;
	printDebug=false;

	wM.diagonal()[0] = wM.diagonal()[1] = wM.diagonal()[2] = SCALE_XI_ROT;
	wM.diagonal()[3] = wM.diagonal()[4] = wM.diagonal()[5] = SCALE_XI_TRANS;
	wM.diagonal()[6] = SCALE_A;
	wM.diagonal()[7] = SCALE_B;
}
CoarseInitializer::~CoarseInitializer()
{
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		if(points[lvl] != 0) delete[] points[lvl];
	}

	delete[] JbBuffer;
	delete[] JbBuffer_new;
}


/**
 * @brief 
 * 
 * @param newFrameHessian [in] FrameHessian对象
 * @param wraps [in&out] 输出对象
 * @return bool 是否跟踪成功
 */
bool CoarseInitializer::trackFrame(FrameHessian *newFrameHessian, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
	newFrame = newFrameHessian;

	// 显示当前帧图像
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushLiveFrame(newFrameHessian);

	int maxIterations[] = {5,5,10,30,50};

	// 初始化参数
	alphaK = 2.5*2.5;//*freeDebugParam1*freeDebugParam1;
	alphaW = 150*150;//*freeDebugParam2*freeDebugParam2;
	regWeight = 0.8;//*freeDebugParam4;
	couplingWeight = 1;//*freeDebugParam5;

	// 初始化未完成
	if(!snapped)
	{
		// 初始化位姿
		thisToNext.translation().setZero();
		// 遍历图像金字塔
		for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
		{
			// 初始化全部特征像素
			int npts = numPoints[lvl];
			Pnt* ptsl = points[lvl];
			for(int i=0;i<npts;i++)
			{
				ptsl[i].iR = 1;
				ptsl[i].idepth_new = 1;
				ptsl[i].lastHessian = 0;
			}
		}
	}


	SE3 refToNew_current = thisToNext;

	AffLight refToNew_aff_current = thisToNext_aff;

	// 如果第一帧和当前帧的曝光时间都大于0，估计仿射变换初值
	if(firstFrame->ab_exposure>0 && newFrame->ab_exposure>0)
		// 仿射参数初始化结果，a为曝光时间比值的对数，b为0
		refToNew_aff_current = AffLight(logf(newFrame->ab_exposure /  firstFrame->ab_exposure),0); // coarse approximation.


	Vec3f latestRes = Vec3f::Zero();
	// 自顶向下遍历图像金字塔
	for(int lvl=pyrLevelsUsed-1; lvl>=0; lvl--)
	{
		// 如果不是最高层，从更高层向当前层传播
		if(lvl<pyrLevelsUsed-1)
			propagateDown(lvl+1);

		Mat88f H,Hsc; Vec8f b,bsc;
		// 更新lvl层的特征像素，如果是最高层的坏点特征像素进行处理
		resetPoints(lvl);
		// 计算残差、Hessian矩阵和Hessian块
		Vec3f resOld = calcResAndGS(lvl, H, b, Hsc, bsc, refToNew_current, refToNew_aff_current, false);
		applyStep(lvl);

		float lambda = 0.1;
		float eps = 1e-4;
		int fails=0;

		if(printDebug)
		{
			printf("lvl %d, it %d (l=%f) %s: %.3f+%.5f -> %.3f+%.5f (%.3f->%.3f) (|inc| = %f)! \t",
					lvl, 0, lambda,
					"INITIA",
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					sqrtf((float)(resOld[0] / resOld[2])),
					sqrtf((float)(resOld[1] / resOld[2])),
					(resOld[0]+resOld[1]) / resOld[2],
					(resOld[0]+resOld[1]) / resOld[2],
					0.0f);
			std::cout << refToNew_current.log().transpose() << " AFF " << refToNew_aff_current.vec().transpose() <<"\n";
		}

		int iteration=0;
		while(true)
		{
			Mat88f Hl = H;
			for(int i=0;i<8;i++) Hl(i,i) *= (1+lambda);
			Hl -= Hsc*(1/(1+lambda));
			Vec8f bl = b - bsc*(1/(1+lambda));

			Hl = wM * Hl * wM * (0.01f/(w[lvl]*h[lvl]));
			bl = wM * bl * (0.01f/(w[lvl]*h[lvl]));


            Vec8f inc;
            SE3 refToNew_new;
            if (fixAffine)
            {
                // Note as we set the weights of rotation and translation to 1 the wM is just the identity in this case.
                inc.head<6>() = -(wM.toDenseMatrix().topLeftCorner<6, 6>() *
                                  (Hl.topLeftCorner<6, 6>().ldlt().solve(bl.head<6>())));
                inc.tail<2>().setZero();
            } else
                inc = -(wM * (Hl.ldlt().solve(bl)));    //=-H^-1 * b.

            double incNorm = inc.norm();

            refToNew_new = SE3::exp(inc.head<6>().cast<double>()) * refToNew_current;

			AffLight refToNew_aff_new = refToNew_aff_current;
			refToNew_aff_new.a += inc[6];
			refToNew_aff_new.b += inc[7];
			doStep(lvl, lambda, inc);


			Mat88f H_new, Hsc_new; Vec8f b_new, bsc_new;
			Vec3f resNew = calcResAndGS(lvl, H_new, b_new, Hsc_new, bsc_new, refToNew_new, refToNew_aff_new, false);
			Vec3f regEnergy = calcEC(lvl);

			float eTotalNew = (resNew[0]+resNew[1]+regEnergy[1]);
			float eTotalOld = (resOld[0]+resOld[1]+regEnergy[0]);


			bool accept = eTotalOld > eTotalNew;

			if(printDebug)
			{
				printf("lvl %d, it %d (l=%f) %s: %.5f + %.5f + %.5f -> %.5f + %.5f + %.5f (%.2f->%.2f) (|inc| = %f)! \t",
						lvl, iteration, lambda,
						(accept ? "ACCEPT" : "REJECT"),
						sqrtf((float)(resOld[0] / resOld[2])),
						sqrtf((float)(regEnergy[0] / regEnergy[2])),
						sqrtf((float)(resOld[1] / resOld[2])),
						sqrtf((float)(resNew[0] / resNew[2])),
						sqrtf((float)(regEnergy[1] / regEnergy[2])),
						sqrtf((float)(resNew[1] / resNew[2])),
						eTotalOld / resNew[2],
						eTotalNew / resNew[2],
						incNorm);
				std::cout << refToNew_new.log().transpose() << " AFF " << refToNew_aff_new.vec().transpose() <<"\n";
			}

			if(accept)
			{
				if(resNew[1] == alphaK*numPoints[lvl])
					snapped = true;
				H = H_new;
				b = b_new;
				Hsc = Hsc_new;
				bsc = bsc_new;
				resOld = resNew;
				refToNew_aff_current = refToNew_aff_new;
				refToNew_current = refToNew_new;
				applyStep(lvl);
				optReg(lvl);
				lambda *= 0.5;
				fails=0;
				if(lambda < 0.0001) lambda = 0.0001;
			}
			else
			{
				fails++;
				lambda *= 4;
				if(lambda > 10000) lambda = 10000;
			}

			bool quitOpt = false;

			if(!(incNorm > eps) || iteration >= maxIterations[lvl] || fails >= 2)
			{
				Mat88f H,Hsc; Vec8f b,bsc;

				quitOpt = true;
			}


			if(quitOpt) break;
			iteration++;
		}
		latestRes = resOld;

	}



	thisToNext = refToNew_current;
	thisToNext_aff = refToNew_aff_current;

	for(int i=0;i<pyrLevelsUsed-1;i++)
		propagateUp(i);




	frameID++;
	if(!snapped) snappedAt=0;

	if(snapped && snappedAt==0)
		snappedAt = frameID;



    debugPlot(0,wraps);



	return snapped && frameID > snappedAt+5;
}

void CoarseInitializer::debugPlot(int lvl, std::vector<IOWrap::Output3DWrapper*> &wraps)
{
    bool needCall = false;
    for(IOWrap::Output3DWrapper* ow : wraps)
        needCall = needCall || ow->needPushDepthImage();
    if(!needCall) return;


	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];

	MinimalImageB3 iRImg(wl,hl);

	for(int i=0;i<wl*hl;i++)
		iRImg.at(i) = Vec3b(colorRef[i][0],colorRef[i][0],colorRef[i][0]);


	int npts = numPoints[lvl];

	float nid = 0, sid=0;
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(point->isGood)
		{
			nid++;
			sid += point->iR;
		}
	}
	float fac = nid / sid;


	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;

		if(!point->isGood)
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,Vec3b(0,0,0));

		else
			iRImg.setPixel9(point->u+0.5f,point->v+0.5f,makeRainbow3B(point->iR*fac));
	}


	//IOWrap::displayImage("idepth-R", &iRImg, false);
    for(IOWrap::Output3DWrapper* ow : wraps)
        ow->pushDepthImage(&iRImg);
}

// calculates residual, Hessian and Hessian-block neede for re-substituting depth.
/**
 * @brief 计算残差、Hessian矩阵和Hessian块
 * 
 * @param lvl [in]
 * @param H_out [out]
 * @param b_out [out]
 * @param H_out_sc [out]
 * @param b_out_sc [out]
 * @param refToNew [in]
 * @param refToNew_aff [in] 
 * @param plot [in]
 * @return Vec3f 
 */
Vec3f CoarseInitializer::calcResAndGS(
		int lvl, Mat88f &H_out, Vec8f &b_out,
		Mat88f &H_out_sc, Vec8f &b_out_sc,
		const SE3 &refToNew, AffLight refToNew_aff,
		bool plot)
{
	int wl = w[lvl], hl = h[lvl];
	Eigen::Vector3f* colorRef = firstFrame->dIp[lvl];
	Eigen::Vector3f* colorNew = newFrame->dIp[lvl];

	Mat33f RKi = (refToNew.rotationMatrix() * Ki[lvl]).cast<float>();
	Vec3f t = refToNew.translation().cast<float>();
	Eigen::Vector2f r2new_aff = Eigen::Vector2f(exp(refToNew_aff.a), refToNew_aff.b);

	float fxl = fx[lvl];
	float fyl = fy[lvl];
	float cxl = cx[lvl];
	float cyl = cy[lvl];



    for(auto&& acc9 : acc9s)
    {
        acc9.initialize();
    }
    for(auto&& E : accE)
    {
        E.initialize();
    }

	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];

    // This part takes most of the time for this method --> parallelize this only.
    auto processPointsForReduce = [&](int min=0, int max=1, double* stats=0, int tid=0)
    {
        auto& acc9 = acc9s[tid];
        auto& E = accE[tid];
        for(int i = min; i < max; i++)
        {
            Pnt* point = ptsl + i;

            point->maxstep = 1e10;
            if(!point->isGood)
            {
                E.updateSingle((float) (point->energy[0]));
                point->energy_new = point->energy;
                point->isGood_new = false;
                continue;
            }

            VecNRf dp0;
            VecNRf dp1;
            VecNRf dp2;
            VecNRf dp3;
            VecNRf dp4;
            VecNRf dp5;
            VecNRf dp6;
            VecNRf dp7;
            VecNRf dd;
            VecNRf r;
            JbBuffer_new[i].setZero();

            // sum over all residuals.
            bool isGood = true;
            float energy = 0;
            for(int idx = 0; idx < patternNum; idx++)
            {
                int dx = patternP[idx][0];
                int dy = patternP[idx][1];


                Vec3f pt = RKi * Vec3f(point->u + dx, point->v + dy, 1) + t * point->idepth_new;
                float u = pt[0] / pt[2];
                float v = pt[1] / pt[2];
                float Ku = fxl * u + cxl;
                float Kv = fyl * v + cyl;
                float new_idepth = point->idepth_new / pt[2];

                if(!(Ku > 1 && Kv > 1 && Ku < wl - 2 && Kv < hl - 2 && new_idepth > 0))
                {
                    isGood = false;
                    break;
                }

                Vec3f hitColor = getInterpolatedElement33(colorNew, Ku, Kv, wl);
                //Vec3f hitColor = getInterpolatedElement33BiCub(colorNew, Ku, Kv, wl);

                //float rlR = colorRef[point->u+dx + (point->v+dy) * wl][0];
                float rlR = getInterpolatedElement31(colorRef, point->u + dx, point->v + dy, wl);

                if(!std::isfinite(rlR) || !std::isfinite((float) hitColor[0]))
                {
                    isGood = false;
                    break;
                }


                float residual = hitColor[0] - r2new_aff[0] * rlR - r2new_aff[1];
                float hw = fabs(residual) < setting_huberTH ? 1 : setting_huberTH / fabs(residual);
                energy += hw * residual * residual * (2 - hw);


                float dxdd = (t[0] - t[2] * u) / pt[2];
                float dydd = (t[1] - t[2] * v) / pt[2];

                if(hw < 1) hw = sqrtf(hw);
                float dxInterp = hw * hitColor[1] * fxl;
                float dyInterp = hw * hitColor[2] * fyl;
                dp0[idx] = new_idepth * dxInterp;
                dp1[idx] = new_idepth * dyInterp;
                dp2[idx] = -new_idepth * (u * dxInterp + v * dyInterp);
                dp3[idx] = -u * v * dxInterp - (1 + v * v) * dyInterp;
                dp4[idx] = (1 + u * u) * dxInterp + u * v * dyInterp;
                dp5[idx] = -v * dxInterp + u * dyInterp;
                dp6[idx] = -hw * r2new_aff[0] * rlR;
                dp7[idx] = -hw * 1;
                dd[idx] = dxInterp * dxdd + dyInterp * dydd;
                r[idx] = hw * residual;

                float maxstep = 1.0f / Vec2f(dxdd * fxl, dydd * fyl).norm();
                if(maxstep < point->maxstep) point->maxstep = maxstep;

                // immediately compute dp*dd' and dd*dd' in JbBuffer1.
                JbBuffer_new[i][0] += dp0[idx] * dd[idx];
                JbBuffer_new[i][1] += dp1[idx] * dd[idx];
                JbBuffer_new[i][2] += dp2[idx] * dd[idx];
                JbBuffer_new[i][3] += dp3[idx] * dd[idx];
                JbBuffer_new[i][4] += dp4[idx] * dd[idx];
                JbBuffer_new[i][5] += dp5[idx] * dd[idx];
                JbBuffer_new[i][6] += dp6[idx] * dd[idx];
                JbBuffer_new[i][7] += dp7[idx] * dd[idx];
                JbBuffer_new[i][8] += r[idx] * dd[idx];
                JbBuffer_new[i][9] += dd[idx] * dd[idx];
            }

            if(!isGood || energy > point->outlierTH * 20)
            {
                E.updateSingle((float) (point->energy[0]));
                point->isGood_new = false;
                point->energy_new = point->energy;
                continue;
            }


            // add into energy.
            E.updateSingle(energy);
            point->isGood_new = true;
            point->energy_new[0] = energy;

            // update Hessian matrix.
            for(int i = 0; i + 3 < patternNum; i += 4)
                acc9.updateSSE(
                        _mm_load_ps(((float*) (&dp0)) + i),
                        _mm_load_ps(((float*) (&dp1)) + i),
                        _mm_load_ps(((float*) (&dp2)) + i),
                        _mm_load_ps(((float*) (&dp3)) + i),
                        _mm_load_ps(((float*) (&dp4)) + i),
                        _mm_load_ps(((float*) (&dp5)) + i),
                        _mm_load_ps(((float*) (&dp6)) + i),
                        _mm_load_ps(((float*) (&dp7)) + i),
                        _mm_load_ps(((float*) (&r)) + i));


            for(int i = ((patternNum >> 2) << 2); i < patternNum; i++)
                acc9.updateSingle(
                        (float) dp0[i], (float) dp1[i], (float) dp2[i], (float) dp3[i],
                        (float) dp4[i], (float) dp5[i], (float) dp6[i], (float) dp7[i],
                        (float) r[i]);


        }
    };

    reduce.reduce(processPointsForReduce, 0, npts, 50);

    for(auto&& acc9 : acc9s)
    {
        acc9.finish();
    }
    for(auto&& E : accE)
    {
        E.finish();
    }


	// calculate alpha energy, and decide if we cap it.
	Accumulator11 EAlpha;
	EAlpha.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
		{
            // This should actually be EAlpha, but it seems like fixing this might change the optimal values of some
            // parameters, so it's kept like it is (see https://github.com/JakobEngel/dso/issues/52)
            // At the moment, this code will not change the value of E.A (because E.finish() is not called again after
            // this. It will however change E.num.
			accE[0].updateSingle((float)(point->energy[1]));
		}
		else
		{
			point->energy_new[1] = (point->idepth_new-1)*(point->idepth_new-1);
			accE[0].updateSingle((float)(point->energy_new[1]));
		}
	}
	EAlpha.finish();
	float alphaEnergy = alphaW*(EAlpha.A + refToNew.translation().squaredNorm() * npts);

	//printf("AE = %f * %f + %f\n", alphaW, EAlpha.A, refToNew.translation().squaredNorm() * npts);


	// compute alpha opt.
	float alphaOpt;
	if(alphaEnergy > alphaK*npts)
	{
		alphaOpt = 0;
		alphaEnergy = alphaK*npts;
	}
	else
	{
		alphaOpt = alphaW;
	}


	acc9SC.initialize();
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		if(!point->isGood_new)
			continue;

		point->lastHessian_new = JbBuffer_new[i][9];

		JbBuffer_new[i][8] += alphaOpt*(point->idepth_new - 1);
		JbBuffer_new[i][9] += alphaOpt;

		if(alphaOpt==0)
		{
			JbBuffer_new[i][8] += couplingWeight*(point->idepth_new - point->iR);
			JbBuffer_new[i][9] += couplingWeight;
		}

		JbBuffer_new[i][9] = 1/(1+JbBuffer_new[i][9]);
		acc9SC.updateSingleWeighted(
				(float)JbBuffer_new[i][0],(float)JbBuffer_new[i][1],(float)JbBuffer_new[i][2],(float)JbBuffer_new[i][3],
				(float)JbBuffer_new[i][4],(float)JbBuffer_new[i][5],(float)JbBuffer_new[i][6],(float)JbBuffer_new[i][7],
				(float)JbBuffer_new[i][8],(float)JbBuffer_new[i][9]);
	}
	acc9SC.finish();


    H_out.setZero();
    b_out.setZero();
    // This needs to sum up the acc9s from all the workers!
    for(auto&& acc9 : acc9s)
    {
        H_out += acc9.H.topLeftCorner<8,8>();// / acc9.num;
        b_out += acc9.H.topRightCorner<8,1>();// / acc9.num;
    }
	H_out_sc = acc9SC.H.topLeftCorner<8,8>();// / acc9.num;
	b_out_sc = acc9SC.H.topRightCorner<8,1>();// / acc9.num;



	H_out(0,0) += alphaOpt*npts;
	H_out(1,1) += alphaOpt*npts;
	H_out(2,2) += alphaOpt*npts;

	Vec3f tlog = refToNew.log().head<3>().cast<float>();
	b_out[0] += tlog[0]*alphaOpt*npts;
	b_out[1] += tlog[1]*alphaOpt*npts;
	b_out[2] += tlog[2]*alphaOpt*npts;


	// Add zero prior to translation.
    // setting_weightZeroPriorDSOInitY is the squared weight of the prior residual.
    H_out(1, 1) += setting_weightZeroPriorDSOInitY;
    b_out(1) += setting_weightZeroPriorDSOInitY * refToNew.translation().y();

    H_out(0, 0) += setting_weightZeroPriorDSOInitX;
    b_out(0) += setting_weightZeroPriorDSOInitX * refToNew.translation().x();

    double A = 0;
    int num = 0;
    for(auto&& E : accE)
    {
        A += E.A;
        num += E.num;
    }

	return Vec3f(A, alphaEnergy, num);
}

float CoarseInitializer::rescale()
{
	float factor = 20*thisToNext.translation().norm();
//	float factori = 1.0f/factor;
//	float factori2 = factori*factori;
//
//	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
//	{
//		int npts = numPoints[lvl];
//		Pnt* ptsl = points[lvl];
//		for(int i=0;i<npts;i++)
//		{
//			ptsl[i].iR *= factor;
//			ptsl[i].idepth_new *= factor;
//			ptsl[i].lastHessian *= factori2;
//		}
//	}
//	thisToNext.translation() *= factori;

	return factor;
}


Vec3f CoarseInitializer::calcEC(int lvl)
{
	if(!snapped) return Vec3f(0,0,numPoints[lvl]);
	AccumulatorX<2> E;
	E.initialize();
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		Pnt* point = points[lvl]+i;
		if(!point->isGood_new) continue;
		float rOld = (point->idepth-point->iR);
		float rNew = (point->idepth_new-point->iR);
		E.updateNoWeight(Vec2f(rOld*rOld,rNew*rNew));

		//printf("%f %f %f!\n", point->idepth, point->idepth_new, point->iR);
	}
	E.finish();

	//printf("ER: %f %f %f!\n", couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], (float)E.num.numIn1m);
	return Vec3f(couplingWeight*E.A1m[0], couplingWeight*E.A1m[1], E.num);
}
/**
 * @brief 使用特征像素的最近邻信息对特征像素的逆深度(仅iR)进行平滑处理
 * 
 * @param lvl 
 */
void CoarseInitializer::optReg(int lvl)
{
	int npts = numPoints[lvl];
	Pnt* ptsl = points[lvl];
	// 未完成初始化，直接返回
	if(!snapped)
	{
		return;
	}

	// 遍历特征像素
	for(int i=0;i<npts;i++)
	{
		Pnt* point = ptsl+i;
		// 特征像素是坏点，直接跳过
		if(!point->isGood) continue;

		float idnn[10];
		int nnn=0;
		// 遍历特征像素的10个最近邻
		for(int j=0;j<10;j++)
		{
			if(point->neighbours[j] == -1) continue;
			// 获取特征像素的最近邻对象
			Pnt* other = ptsl+point->neighbours[j];
			if(!other->isGood) continue;
			// 记录特征像素最近邻的逆深度信息
			idnn[nnn] = other->iR;
			// 统计有效最近邻的数量
			nnn++;
		}

		// 有效最近邻的数量大于2
		if(nnn > 2)
		{
			// 部分排列idnn数组，划分为两部分，前nnn/2个元素不大于idnn[nnn/2]，后nnn/2个元素不小于idnn[nnn/2]
			std::nth_element(idnn,idnn+nnn/2,idnn+nnn);
			// 选择最近邻的中位数，对特征像素的逆深度(iR)进行平滑处理
			point->iR = (1-regWeight)*point->idepth + regWeight*idnn[nnn/2];
		}
	}

}



void CoarseInitializer::propagateUp(int srcLvl)
{
	assert(srcLvl+1<pyrLevelsUsed);
	// set idepth of target

	int nptss= numPoints[srcLvl];
	int nptst= numPoints[srcLvl+1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl+1];

	// set to zero.
	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		parent->iR=0;
		parent->iRSumNum=0;
	}

	for(int i=0;i<nptss;i++)
	{
		Pnt* point = ptss+i;
		if(!point->isGood) continue;

		Pnt* parent = ptst + point->parent;
		parent->iR += point->iR * point->lastHessian;
		parent->iRSumNum += point->lastHessian;
	}

	for(int i=0;i<nptst;i++)
	{
		Pnt* parent = ptst+i;
		if(parent->iRSumNum > 0)
		{
			parent->idepth = parent->iR = (parent->iR / parent->iRSumNum);
			parent->isGood = true;
		}
	}

	optReg(srcLvl+1);
}

/**
 * @brief 将逆深度信息(更新iR, idepth, idepth_new)从srcLvl层传播到srcLvl-1层
 * 
 * @param srcLvl 当金字塔上一层
 */
void CoarseInitializer::propagateDown(int srcLvl)
{
	assert(srcLvl>0);
	// set idepth of target

	int nptst= numPoints[srcLvl-1];
	Pnt* ptss = points[srcLvl];
	Pnt* ptst = points[srcLvl-1];

	// 遍历srcLvl-1层的所有特征像素
	for(int i=0;i<nptst;i++)
	{
		// 获取srcLvl-1层特征像素Pnt对象
		Pnt* point = ptst+i;
		// 获取特征像素在srcLvl的父节点特征像素
		Pnt* parent = ptss+point->parent;

		// 检查父节点特征像素是否有效
		if(!parent->isGood || parent->lastHessian < 0.1) continue;
		// 检查srcLvl-1层特征像素是否有效
		// 不是好点
		if(!point->isGood)
		{
			// 将srcLvl层父节点特征像素的深度信息传播到srcLvl-1层特征像素
			point->iR = point->idepth = point->idepth_new = parent->iR;
			// 更新srcLvl-1层特征像素的状态
			point->isGood=true;
			point->lastHessian=0;
		}
		// 是好点
		else
		{
			// 通过hessian给point->iR和parent->iR加权求得新的iR，hessian为信息矩阵(方差的倒数)
			// \mu_new = (\Inform_c * \mu_c + \Inform_p * \mu_p) / (\Inform_c + \Inform_p)
			float newiR = (point->iR*point->lastHessian*2 + parent->iR*parent->lastHessian) / (point->lastHessian*2+parent->lastHessian);
			point->iR = point->idepth = point->idepth_new = newiR;
		}
	}
	// 对srcLvl-1层特征像素的逆深度(仅iR)进行平滑处理
	optReg(srcLvl-1);
}


void CoarseInitializer::makeGradients(Eigen::Vector3f** data)
{
	for(int lvl=1; lvl<pyrLevelsUsed; lvl++)
	{
		int lvlm1 = lvl-1;
		int wl = w[lvl], hl = h[lvl], wlm1 = w[lvlm1];

		Eigen::Vector3f* dINew_l = data[lvl];
		Eigen::Vector3f* dINew_lm = data[lvlm1];

		for(int y=0;y<hl;y++)
			for(int x=0;x<wl;x++)
				dINew_l[x + y*wl][0] = 0.25f * (dINew_lm[2*x   + 2*y*wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1][0] +
													dINew_lm[2*x   + 2*y*wlm1+wlm1][0] +
													dINew_lm[2*x+1 + 2*y*wlm1+wlm1][0]);

		for(int idx=wl;idx < wl*(hl-1);idx++)
		{
			dINew_l[idx][1] = 0.5f*(dINew_l[idx+1][0] - dINew_l[idx-1][0]);
			dINew_l[idx][2] = 0.5f*(dINew_l[idx+wl][0] - dINew_l[idx-wl][0]);
		}
	}
}

/**
 * @brief 初始化第一帧
 * 
 * @param HCalib [in]，相机标定参数
 * @param newFrameHessian [in]，图像帧处理后的对象(包含图像金字塔各层灰度和梯度)
 */
void CoarseInitializer::setFirst(	CalibHessian* HCalib, FrameHessian* newFrameHessian)
{
	// 计算图像金字塔各层的内参矩阵、逆矩阵
	makeK(HCalib);
	firstFrame = newFrameHessian;

	// 创建PixelSelector对象
	PixelSelector sel(w[0],h[0]);

	// 创建statusMap和statusMapB保存图像像素状态
	// statusMap保存第0层的像素状态 [0, 1, 2, 4] 0-不是特征像素，1-第0层特征像素，2-第1层特征像素，4-第2层特征像素
	float* statusMap = new float[w[0]*h[0]];
	// statusMapB保存其他层的像素状态 [false, true]
	bool* statusMapB = new bool[w[0]*h[0]];

	// 不同层取得的像素点密度 [0, 1, 2, 3, 4]
	float densities[] = {0.03,0.05,0.15,0.5,1};
	// 遍历图像金字塔各层
	for(int lvl=0; lvl<pyrLevelsUsed; lvl++)
	{
		sel.currentPotential = 3;
		// 选取的像素数目
		int npts;
		if(lvl == 0)
			// 第0层提取特征像素
			npts = sel.makeMaps(firstFrame, statusMap,densities[lvl]*w[0]*h[0],1,false,2);
		else
			// 其他层提取特征像素
			npts = makePixelStatus(firstFrame->dIp[lvl], statusMapB, w[lvl], h[lvl], densities[lvl]*w[0]*h[0]);


		// 删除原有points[lvl]指向的内存空间，重新分配内存
		if(points[lvl] != 0) 
			delete[] points[lvl];
		points[lvl] = new Pnt[npts];

		// set idepth map to initially 1 everywhere.
		int wl = w[lvl], hl = h[lvl];
		// 获取Pnt对象指针
		Pnt* pl = points[lvl];
		// 对应特征像素的索引
		int nl = 0;
		// 遍历图像金字塔各层的像素点，留出patternPadding边界，borderSize=2
		for(int y=patternPadding+1;y<hl-patternPadding-2;y++)
		{
			for(int x=patternPadding+1;x<wl-patternPadding-2;x++)
			{
				//if(x==2) printf("y=%d!\n",y);
				// 如果是特征像素
				if((lvl!=0 && statusMapB[x+y*wl]) || (lvl==0 && statusMap[x+y*wl] != 0))
				{
					//assert(patternNum==9);
					// 初始化Pnt对象
					pl[nl].u = x+0.1;
					pl[nl].v = y+0.1;
					// 初始化逆深度为1
					pl[nl].idepth = 1;
					pl[nl].iR = 1;
					pl[nl].isGood=true;
					pl[nl].energy.setZero();
					pl[nl].lastHessian=0;
					pl[nl].lastHessian_new=0;
					// 初始化特征像素类型
					pl[nl].my_type= (lvl!=0) ? 1 : statusMap[x+y*wl];

					// 当前像素的像素、水平梯度、垂直梯度
					Eigen::Vector3f* cpt = firstFrame->dIp[lvl] + x + y*w[lvl];
					float sumGrad2=0;
					// 统计Pattern中的像素梯度平方和，patternNum=8
					for(int idx=0;idx<patternNum;idx++)
					{
						int dx = patternP[idx][0];
						int dy = patternP[idx][1];
						float absgrad = cpt[dx + dy*w[lvl]].tail<2>().squaredNorm();
						sumGrad2 += absgrad;
					}

	//				float gth = setting_outlierTH * (sqrtf(sumGrad2)+setting_outlierTHSumComponent);
	//				pl[nl].outlierTH = patternNum*gth*gth;
	//

					// 设置外点阈值
					pl[nl].outlierTH = patternNum*setting_outlierTH;


					// 特征像素索引增加
					nl++;
					assert(nl <= npts);
				}
			}
		}
		
		// 统计当前层特征像素数目
		numPoints[lvl]=nl;
	}
	delete[] statusMap;
	delete[] statusMapB;
	
	// 计算点的最近邻和父点
	makeNN();

	// 参数初始化
	thisToNext=SE3();
	// 标记未收敛，初始化未完成
	snapped = false;
	// 初始化帧ID
	frameID = snappedAt = 0;

	for(int i=0;i<pyrLevelsUsed;i++)
		dGrads[i].setZero();

}

/**
 * @brief 对lvl层的特征像素进行重置并对图像金字塔最高层的坏点特征像素进行处理
 * 
 * @param lvl [in] 
 */
void CoarseInitializer::resetPoints(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	// 遍历当前层的特征像素
	for(int i=0;i<npts;i++)
	{
		// 重置特征像素的能量
		pts[i].energy.setZero();
		// 更新特征像素的逆深度
		pts[i].idepth_new = pts[i].idepth;

		// 如果当前是图像金字塔最高层 且 特征像素是坏点
		if(lvl==pyrLevelsUsed-1 && !pts[i].isGood)
		{
			// 分别记录特征像素最近邻逆深度和，有效数目
			float snd=0, sn=0;
			// 遍历特征像素的10个最近邻
			for(int n = 0;n<10;n++)
			{
				// 如果最近邻不存在或者最近邻不是好点，直接跳过
				if(pts[i].neighbours[n] == -1 || !pts[pts[i].neighbours[n]].isGood) continue;
				snd += pts[pts[i].neighbours[n]].iR;
				sn += 1;
			}

			if(sn > 0)
			{
				// 标记特征像素为好点
				pts[i].isGood=true;
				// 更新特征像素的逆深度(iR, idepth, idepth_new)为最近邻逆深度的平均值
				pts[i].iR = pts[i].idepth = pts[i].idepth_new = snd/sn;
			}
		}
	}
}

void CoarseInitializer::doStep(int lvl, float lambda, Vec8f inc)
{

	const float maxPixelStep = 0.25;
	const float idMaxStep = 1e10;
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood) continue;


		float b = JbBuffer[i][8] + JbBuffer[i].head<8>().dot(inc);
		float step = - b * JbBuffer[i][9] / (1+lambda);


		float maxstep = maxPixelStep*pts[i].maxstep;
		if(maxstep > idMaxStep) maxstep=idMaxStep;

		if(step >  maxstep) step = maxstep;
		if(step < -maxstep) step = -maxstep;

		float newIdepth = pts[i].idepth + step;
		if(newIdepth < 1e-3 ) newIdepth = 1e-3;
		if(newIdepth > 50) newIdepth = 50;
		pts[i].idepth_new = newIdepth;
	}

}
void CoarseInitializer::applyStep(int lvl)
{
	Pnt* pts = points[lvl];
	int npts = numPoints[lvl];
	for(int i=0;i<npts;i++)
	{
		if(!pts[i].isGood)
		{
			pts[i].idepth = pts[i].idepth_new = pts[i].iR;
			continue;
		}
		pts[i].energy = pts[i].energy_new;
		pts[i].isGood = pts[i].isGood_new;
		pts[i].idepth = pts[i].idepth_new;
		pts[i].lastHessian = pts[i].lastHessian_new;
	}
	std::swap<Vec10f*>(JbBuffer, JbBuffer_new);
}

/**
 * @brief 计算图像金字塔各层的内参矩阵和逆矩阵
 * 
 * @param HCalib 
 */
void CoarseInitializer::makeK(CalibHessian* HCalib)
{
	// 图像金字塔第0层的宽高和相机内参
	w[0] = wG[0];
	h[0] = hG[0];

	fx[0] = HCalib->fxl();
	fy[0] = HCalib->fyl();
	cx[0] = HCalib->cxl();
	cy[0] = HCalib->cyl();

	// 从第1层开始遍历图像金字塔
	for (int level = 1; level < pyrLevelsUsed; ++ level)
	{
		// 各层图像金字塔的宽高为 第0层/(2^level)
		w[level] = w[0] >> level;
		h[level] = h[0] >> level;
		// fx fy 
		fx[level] = fx[level-1] * 0.5;
		fy[level] = fy[level-1] * 0.5;
		cx[level] = (cx[0] + 0.5) / ((int)1<<level) - 0.5;
		cy[level] = (cy[0] + 0.5) / ((int)1<<level) - 0.5;
	}

	// 计算各层图像金字塔相机内参的逆矩阵
	for (int level = 0; level < pyrLevelsUsed; ++ level)
	{
		K[level]  << fx[level], 0.0, cx[level], 0.0, fy[level], cy[level], 0.0, 0.0, 1.0;
		// K^-1 = [fx^-1 0 -cx*fx^-1; 0 fy^-1 -cy*fy^-1; 0 0 1]
		Ki[level] = K[level].inverse();
		fxi[level] = Ki[level](0,0);
		fyi[level] = Ki[level](1,1);
		cxi[level] = Ki[level](0,2);
		cyi[level] = Ki[level](1,2);
	}
}



/**
 * @brief 计算各层像素的最近邻和父点
 * 
 */
void CoarseInitializer::makeNN()
{
	// 最近邻距离衰减因子
	const float NNDistFactor=0.05;

	// KDTreeSingleIndexAdaptor模板类用于创建KDTree对象
	// 第一个参数为距离度量函数对象(对于FLANNPointcloud类型点云计算L2范数)
	// 第二个是点云类型, 第三个是维数
	typedef nanoflann::KDTreeSingleIndexAdaptor<
			nanoflann::L2_Simple_Adaptor<float, FLANNPointcloud> ,
			FLANNPointcloud,2> KDTree;

	// build indices
	// 为每一层的点云数据创建FLANNPointcloud对象和KDTree对象
	FLANNPointcloud pcs[PYR_LEVELS];
	KDTree* indexes[PYR_LEVELS];
	// 遍历图像金字塔各层
	for(int i=0;i<pyrLevelsUsed;i++)
	{
		// 创建FLANNPointcloud对象
		pcs[i] = FLANNPointcloud(numPoints[i], points[i]);
		// 创建KDTree对象
		// 参数1：维数；参数2：FLANNPointcloud对象；参数3：叶节点中的最大点数(参数小使增加树的深度，增加查询时间，减少内存使用)
		indexes[i] = new KDTree(2, pcs[i], nanoflann::KDTreeSingleIndexAdaptorParams(5) );
		// 构建KDTree的索引
		indexes[i]->buildIndex();
	}

	// 最近邻数目
	const int nn=10;

	// find NN & parents
	// 查找最近邻和父点
	// 遍历图像金字塔各层
	for(int lvl=0;lvl<pyrLevelsUsed;lvl++)
	{
		// 特征像素数组和当前层特征像素数目
		Pnt* pts = points[lvl];
		int npts = numPoints[lvl];

		// 搜索到的最近邻点索引和距离
		int ret_index[nn];
		float ret_dist[nn];
		// 搜索最近邻的nn个点
		nanoflann::KNNResultSet<float, int, int> resultSet(nn);
		// 搜索最近邻的1个点
		nanoflann::KNNResultSet<float, int, int> resultSet1(1);

		// 遍历当前层特征像素
		for(int i=0;i<npts;i++)
		{
			//resultSet.init(pts[i].neighbours, pts[i].neighboursDist );
			// 初始化resultSet
			resultSet.init(ret_index, ret_dist);
			// 当前特征像素的坐标
			Vec2f pt = Vec2f(pts[i].u,pts[i].v);
			// 查找最近邻
			indexes[lvl]->findNeighbors(resultSet, (float*)&pt, nanoflann::SearchParams());
			// 表示当前特征像素的邻居数目
			int myidx=0;
			// 表示当前特征像素的最近邻指数衰减距离的和
			float sumDF = 0;
			// 遍历nn个最近邻
			for(int k=0;k<nn;k++)
			{
				// 将第k最近邻索引存储到当前特征像素的邻居数组中
				pts[i].neighbours[myidx]=ret_index[k];
				// 计算最近邻的指数衰减距离
				float df = expf(-ret_dist[k]*NNDistFactor);
				// 累加指数衰减距离
				sumDF += df;
				// 将最近邻的指数衰减距离存储到当前特征像素的邻居距离数组中
				pts[i].neighboursDist[myidx]=df;
				assert(ret_index[k]>=0 && ret_index[k] < npts);
				myidx++;
			}
			// 将最近邻的指数衰减距离归一化到[0,10]
			for(int k=0;k<nn;k++)
				pts[i].neighboursDist[k] *= 10/sumDF;


			// 在更高层图像中找到父节点
			if(lvl < pyrLevelsUsed-1 )
			{
				// 初始化resultSet1
				resultSet1.init(ret_index, ret_dist);
				// 计算当前特征像素的坐标在更高层图像中的坐标
				// 高层坐标[x, y]与底层坐标[2x+1/2, 2y+1/2]的对应关系
				pt = pt*0.5f-Vec2f(0.25f,0.25f);
				// 找到1个最近邻
				indexes[lvl+1]->findNeighbors(resultSet1, (float*)&pt, nanoflann::SearchParams());

				// 将最近邻的索引存储到当前特征像素的父节点数组中
				pts[i].parent = ret_index[0];
				// 计算最近邻的指数衰减距离
				pts[i].parentDist = expf(-ret_dist[0]*NNDistFactor);

				assert(ret_index[0]>=0 && ret_index[0] < numPoints[lvl+1]);
			}
			// 如果是最高层图像，则父节点为-1
			else
			{
				pts[i].parent = -1;
				pts[i].parentDist = -1;
			}
		}
	}



	// done.
	// 清除内存
	for(int i=0;i<pyrLevelsUsed;i++)
		delete indexes[i];
}
}

