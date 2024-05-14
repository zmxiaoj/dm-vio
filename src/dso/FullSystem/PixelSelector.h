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


const float minUseGrad_pixsel = 10;


/**
 * @brief 
 * 
 * @tparam pot 
 * @param grads 
 * @param map_out 
 * @param w 
 * @param h 
 * @param THFac 
 * @return int 
 */
template<int pot>
inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, float THFac)
{

	// 将map_out数组初始化为0
	memset(map_out, 0, sizeof(bool)*w*h);

	int numGood = 0;
	// 从(1,1)开始，遍历图像金字塔的每个pot*pot的网格
	for(int y=1;y<h-pot;y+=pot)
	{
		for(int x=1;x<w-pot;x+=pot)
		{
			// 初始化水平梯度、垂直梯度、水平梯度和垂直梯度差值、水平梯度和垂直梯度和值最大值和索引
			int bestXXID = -1;
			int bestYYID = -1;
			int bestXYID = -1;
			int bestYXID = -1;

			float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

			// 获取当前网格的像素、水平梯度、垂直梯度索引
			Eigen::Vector3f* grads0 = grads+x+y*w;
			// 遍历网格中的每个pixel
			for(int dx=0;dx<pot;dx++)
				for(int dy=0;dy<pot;dy++)
				{
					// 获取pixel索引
					int idx = dx+dy*w;
					Eigen::Vector3f g=grads0[idx];
					// 取出水平梯度和垂直梯度，计算平方和
					float sqgd = g.tail<2>().squaredNorm();
					// 计算阈值(THFac=1;minUseGrad_pixsel=10)
					float TH = THFac*minUseGrad_pixsel * (0.75f);

					// 如果梯度平方和大于阈值的平方
					if(sqgd > TH*TH)
					{
						// 计算水平梯度的绝对值
						float agx = fabs((float)g[1]);
						// 更新水平梯度最大值和索引
						if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

						// 计算垂直梯度的绝对值
						float agy = fabs((float)g[2]);
						// 更新垂直梯度最大值和索引
						if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

						// 计算水平梯度和垂直梯度的差值
						float gxpy = fabs((float)(g[1]-g[2]));
						// 更新水平梯度和垂直梯度差值最大值和索引
						if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

						// 计算水平梯度和垂直梯度的和值
						float gxmy = fabs((float)(g[1]+g[2]));
						// 更新水平梯度和垂直梯度和值最大值和索引
						if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
					}
				}

			// 获取当前网格的map_out指针
			bool* map0 = map_out+x+y*w;

			// 更新map_out数组
			// 如果水平梯度最大值索引大于等于0
			if(bestXXID>=0)
			{
				// 如果map_out数组中的值为0
				if(!map0[bestXXID])
					// 特征像素数量加1
					numGood++;
				// 更新map_out数组
				map0[bestXXID] = true;

			}
			if(bestYYID>=0)
			{
				if(!map0[bestYYID])
					numGood++;
				map0[bestYYID] = true;

			}
			if(bestXYID>=0)
			{
				if(!map0[bestXYID])
					numGood++;
				map0[bestXYID] = true;

			}
			if(bestYXID>=0)
			{
				if(!map0[bestYXID])
					numGood++;
				map0[bestYXID] = true;

			}
		}
	}

	// 返回特征像素数量
	return numGood;
}


// 同上模版函数，将pot作为参数传入
inline int gridMaxSelection(Eigen::Vector3f* grads, bool* map_out, int w, int h, int pot, float THFac)
{

	memset(map_out, 0, sizeof(bool)*w*h);

	int numGood = 0;
	for(int y=1;y<h-pot;y+=pot)
	{
		for(int x=1;x<w-pot;x+=pot)
		{
			int bestXXID = -1;
			int bestYYID = -1;
			int bestXYID = -1;
			int bestYXID = -1;

			float bestXX=0, bestYY=0, bestXY=0, bestYX=0;

			Eigen::Vector3f* grads0 = grads+x+y*w;
			for(int dx=0;dx<pot;dx++)
				for(int dy=0;dy<pot;dy++)
				{
					int idx = dx+dy*w;
					Eigen::Vector3f g=grads0[idx];
					float sqgd = g.tail<2>().squaredNorm();
					float TH = THFac*minUseGrad_pixsel * (0.75f);

					if(sqgd > TH*TH)
					{
						float agx = fabs((float)g[1]);
						if(agx > bestXX) {bestXX=agx; bestXXID=idx;}

						float agy = fabs((float)g[2]);
						if(agy > bestYY) {bestYY=agy; bestYYID=idx;}

						float gxpy = fabs((float)(g[1]-g[2]));
						if(gxpy > bestXY) {bestXY=gxpy; bestXYID=idx;}

						float gxmy = fabs((float)(g[1]+g[2]));
						if(gxmy > bestYX) {bestYX=gxmy; bestYXID=idx;}
					}
				}

			bool* map0 = map_out+x+y*w;

			if(bestXXID>=0)
			{
				if(!map0[bestXXID])
					numGood++;
				map0[bestXXID] = true;

			}
			if(bestYYID>=0)
			{
				if(!map0[bestYYID])
					numGood++;
				map0[bestYYID] = true;

			}
			if(bestXYID>=0)
			{
				if(!map0[bestXYID])
					numGood++;
				map0[bestXYID] = true;

			}
			if(bestYXID>=0)
			{
				if(!map0[bestYXID])
					numGood++;
				map0[bestYXID] = true;

			}
		}
	}

	return numGood;
}


/**
 * @brief 选择图像金字塔第0层外的特征像素
 * 
 * @param grads [in] 当前层图像金字塔的像素、水平梯度(dx)、垂直梯度(dy)
 * @param map [in]
 * @param w [in]
 * @param h [in]
 * @param desiredDensity 
 * @param recsLeft 
 * @param THFac 
 * @return int [out] 返回特征像素的数量
 */
inline int makePixelStatus(Eigen::Vector3f* grads, bool* map, int w, int h, float desiredDensity, int recsLeft=5, float THFac = 1)
{
	// default=5
	if(sparsityFactor < 1) sparsityFactor = 1;

	int numGoodPoints;

	// 根据sparsityFactor选择特征点
	if(sparsityFactor==1) numGoodPoints = gridMaxSelection<1>(grads, map, w, h, THFac);
	else if(sparsityFactor==2) numGoodPoints = gridMaxSelection<2>(grads, map, w, h, THFac);
	else if(sparsityFactor==3) numGoodPoints = gridMaxSelection<3>(grads, map, w, h, THFac);
	else if(sparsityFactor==4) numGoodPoints = gridMaxSelection<4>(grads, map, w, h, THFac);
	else if(sparsityFactor==5) numGoodPoints = gridMaxSelection<5>(grads, map, w, h, THFac);
	else if(sparsityFactor==6) numGoodPoints = gridMaxSelection<6>(grads, map, w, h, THFac);
	else if(sparsityFactor==7) numGoodPoints = gridMaxSelection<7>(grads, map, w, h, THFac);
	else if(sparsityFactor==8) numGoodPoints = gridMaxSelection<8>(grads, map, w, h, THFac);
	else if(sparsityFactor==9) numGoodPoints = gridMaxSelection<9>(grads, map, w, h, THFac);
	else if(sparsityFactor==10) numGoodPoints = gridMaxSelection<10>(grads, map, w, h, THFac);
	else if(sparsityFactor==11) numGoodPoints = gridMaxSelection<11>(grads, map, w, h, THFac);
	else numGoodPoints = gridMaxSelection(grads, map, w, h, sparsityFactor, THFac);


	/*
	 * #points is approximately proportional to sparsityFactor^2.
	 */

	// 计算当前特征像素和需要的特征像素的比值
	float quotia = numGoodPoints / (float)(desiredDensity);

	// 更新网格大小
	int newSparsity = (sparsityFactor * sqrtf(quotia))+0.7f;

	// clamp ewSparsity
	if(newSparsity < 1) newSparsity=1;


	float oldTHFac = THFac;
	// 网格大小减少到最小数目仍不够，降低阈值
	if(newSparsity==1 && sparsityFactor==1) THFac = 0.5;


	// 如果新的网格大小和阈值和旧的网格大小相差不大且阈值不变，或者现有特征像素和需要的特征像素的比值大于0.8且特征像素和需要的特征像素的比值的倒数大于0.8，或者剩余递归次数为0
	if((abs(newSparsity-sparsityFactor) < 1 && THFac==oldTHFac) ||
			( quotia > 0.8 &&  1.0f / quotia > 0.8) ||
			recsLeft == 0)
	{

//		printf(" \n");
		//all good
		sparsityFactor = newSparsity;
		// 返回特征像素的数量
		return numGoodPoints;
	}
	else
	{
//		printf(" -> re-evaluate! \n");
		// re-evaluate.
		sparsityFactor = newSparsity;
		// 递归调用，重新选择特征像素
		return makePixelStatus(grads, map, w,h, desiredDensity, recsLeft-1, THFac);
	}
}

}

