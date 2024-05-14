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


#include "FullSystem/PixelSelector2.h"
 
// 



#include "util/NumType.h"
#include "IOWrapper/ImageDisplay.h"
#include "util/globalCalib.h"
#include "FullSystem/HessianBlocks.h"
#include "util/globalFuncs.h"

namespace dso
{

/**
 * @brief Construct a new Pixel Selector:: Pixel Selector object
 *        构造PixelSelector对象，初始化随机模式数组，设置随机种子，初始化梯度直方图和阈值数组
 * @param w 
 * @param h 
 */
PixelSelector::PixelSelector(int w, int h)
{
	randomPattern = new unsigned char[w*h];
	// 设置随机种子，希望系统是确定性的
	std::srand(3141592);	// want to be deterministic.
	for(int i=0;i<w*h;i++) randomPattern[i] = rand() & 0xFF;

	currentPotential=3;

	// We create 32 blocks in width dimension, and adjust the number of blocks for the height accordingly.
    // Always use block size of 16.
	// 使用16*16的块，计算宽、高方向的块数
    bW = 16;
    bH = 16;
    nbW = w / bW;
    nbH = h / bH;
    if(w != bW * nbW || h != bH * nbH)
    {
        std::cout << "ERROR: Height or width seem to be not divisible by 16!" << std::endl;
        assert(0);
    }

    std::cout << "PixelSelector: Using block sizes: " << bW << ", " << bH << '\n';

	// 初始化数组用于存储梯度直方图、阈值、平滑后的阈值(3x3block内阈值的均值平方)
	gradHist = new int[100*(1+nbW)*(1+nbH)];
	ths = new float[(nbW)*(nbH)+100];
	thsSmoothed = new float[(nbW)*(nbH)+100];

	// 设置标志控制是否允许快速像素选择
	allowFast=false;
	gradHistFrame=0;
}

PixelSelector::~PixelSelector()
{
	delete[] randomPattern;
	delete[] gradHist;
	delete[] ths;
	delete[] thsSmoothed;
}

/**
 * @brief 计算梯度直方图的分位数
 * 
 * @param hist [in] 直方图数组
 * @param below [in] default: 0.5f, 50%分位数的比例
 * @return int 
 */
int computeHistQuantil(int* hist, float below)
{
	// hist[0]保存了全部有效pixel的数量
	// 计算直方图的(50%)分位数，四舍五入取整
	int th = hist[0]*below+0.5f;
	// 遍历直方图，找到分位数的位置
	for(int i=0;i<90;i++)
	{
		th -= hist[i+1];
		if(th<0) return i;
	}
	return 90;
}

/**
 * @brief 生成梯度直方图
 * 
 * @param fh 
 */
void PixelSelector::makeHists(const FrameHessian* const fh)
{
	gradHistFrame = fh;
	// 取出第0层图像的梯度平方和(dx^2 + dy^2)
	float * mapmax0 = fh->absSquaredGrad[0];

	// 取出第0层图像的宽、高
	int w = wG[0];
	int h = hG[0];

	// 取出图像宽、高划分的块数
	// DSO中，使用块的大小为32x32(1024) pixels
	// DMVIO中，使用块的大小为16x16(256) pixels
	int w32 = nbW;
	int h32 = nbH;
	thsStep = w32;
	
	// 遍历每个图像块，y为块的行数，x为块的列数(x,y)
	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			// 取出当前块的梯度平方和
			float* map0 = mapmax0+bW*x+bH*y*w;
			int* hist0 = gradHist;// + 50*(x+y*w32);
			// 直方图内存清零并初始化大小为50
			// hist0[0]统计block内参与直方图统计的有效pixel数量，hist0[1]~hist0[50]统计梯度直方图
			memset(hist0,0,sizeof(int)*50);

			// 遍历当前block内的每个pixel
			for(int j=0;j<bH;j++) 
				for(int i=0;i<bW;i++)
				{
					// 当前pixel的索引(i,j)
					int it = i+bW*x;
					int jt = j+bH*y;
					// 当前pixel超出block范围，跳过
					if(it>w-2 || jt>h-2 || it<1 || jt<1) 
						continue;
					// 取出当前pixel的梯度平方和的平方根
					int g = sqrtf(map0[i+j*w]);
					// clamp梯度值
					if(g>48) 
						g=48;
					// 更新梯度直方图
					hist0[g+1]++;
					// 更新有效pixel数量
					hist0[0]++;
				}
			// 计算当前block的梯度直方图的分位数
			// 在50%分位数的基础上增加setting_minGradHistAdd(default=7)作为当前block的阈值
			ths[x+y*w32] = computeHistQuantil(hist0,setting_minGradHistCut) + setting_minGradHistAdd;
		}

	// 遍历每个图像块，y为块的行数，x为块的列数(x,y)
	// 用3x3范围的block均值滤波器对阈值进行平滑
	for(int y=0;y<h32;y++)
		for(int x=0;x<w32;x++)
		{
			// 统计block内的梯度阈值总数和有效block数量
			float sum=0,num=0;
			if(x>0)
			{
				if(y>0) 	{num++; 	sum+=ths[x-1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x-1+(y+1)*w32];}
				num++; sum+=ths[x-1+(y)*w32];
			}

			if(x<w32-1)
			{
				if(y>0) 	{num++; 	sum+=ths[x+1+(y-1)*w32];}
				if(y<h32-1) {num++; 	sum+=ths[x+1+(y+1)*w32];}
				num++; sum+=ths[x+1+(y)*w32];
			}

			if(y>0) 	{num++; 	sum+=ths[x+(y-1)*w32];}
			if(y<h32-1) {num++; 	sum+=ths[x+(y+1)*w32];}
			num++; sum+=ths[x+y*w32];

			// 计算阈值在3x3block内的均值平方
			thsSmoothed[x+y*w32] = (sum/num) * (sum/num);

		}
}

/**
 * @brief 对第0层提取特征像素
 * 
 * @param fh FrameHessian对象
 * @param map_out 输出的特征像素图
 * @param density 特征像素的密度
 * @param recursionsLeft 最大递归次数
 * @param plot 画图
 * @param thFactor 阈值因子 default=2
 * @return int 
 */
int PixelSelector::makeMaps(
		const FrameHessian* const fh,
		float* map_out, float density, int recursionsLeft, bool plot, float thFactor)
{
	float numHave=0;
	float numWant=density;
	float quotia;
	int idealPotential = currentPotential;


//	if(setting_pixelSelectionUseFast>0 && allowFast)
//	{
//		memset(map_out, 0, sizeof(float)*wG[0]*hG[0]);
//		std::vector<cv::KeyPoint> pts;
//		cv::Mat img8u(hG[0],wG[0],CV_8U);
//		for(int i=0;i<wG[0]*hG[0];i++)
//		{
//			float v = fh->dI[i][0]*0.8;
//			img8u.at<uchar>(i) = (!std::isfinite(v) || v>255) ? 255 : v;
//		}
//		cv::FAST(img8u, pts, setting_pixelSelectionUseFast, true);
//		for(unsigned int i=0;i<pts.size();i++)
//		{
//			int x = pts[i].pt.x+0.5;
//			int y = pts[i].pt.y+0.5;
//			map_out[x+y*wG[0]]=1;
//			numHave++;
//		}
//
//		printf("FAST selection: got %f / %f!\n", numHave, numWant);
//		quotia = numWant / numHave;
//	}
//	else
	{




		// the number of selected pixels behaves approximately as
		// K / (pot+1)^2, where K is a scene-dependent constant.
		// we will allow sub-selecting pixels by up to a quotia of 0.25, otherwise we will re-select.
		// 像素选择的数量大约为K/(pot+1)^2，其中K是一个与场景相关的常数

		// 没有计算直方图, 以及选点的阈值, 生成直方图和阈值
		if(fh != gradHistFrame) 
			makeHists(fh);

		// select!
		// 选择符合条件的特征像素
		Eigen::Vector3i n = this->select(fh, map_out, currentPotential, thFactor);

		// sub-select!
		numHave = n[0]+n[1]+n[2];
		quotia = numWant / numHave;

		// by default we want to over-sample by 40% just to be sure.
		// 默认进行40%的过采样，确保有足够的特征点
		float K = numHave * (currentPotential+1) * (currentPotential+1);
		idealPotential = sqrtf(K/numWant)-1;	// round down.
		if(idealPotential<1) idealPotential=1;

		if( recursionsLeft>0 && quotia > 1.25 && currentPotential>1)
		{
			//re-sample to get more points!
			// potential needs to be smaller
			if(idealPotential>=currentPotential)
				idealPotential = currentPotential-1;

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);
			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);
		}
		else if(recursionsLeft>0 && quotia < 0.25)
		{
			// re-sample to get less points!

			if(idealPotential<=currentPotential)
				idealPotential = currentPotential+1;

	//		printf("PixelSelector: have %.2f%%, need %.2f%%. RESAMPLE with pot %d -> %d.\n",
	//				100*numHave/(float)(wG[0]*hG[0]),
	//				100*numWant/(float)(wG[0]*hG[0]),
	//				currentPotential,
	//				idealPotential);
			currentPotential = idealPotential;
			return makeMaps(fh,map_out, density, recursionsLeft-1, plot,thFactor);

		}
	}

	int numHaveSub = numHave;
	if(quotia < 0.95)
	{
		int wh=wG[0]*hG[0];
		int rn=0;
		unsigned char charTH = 255*quotia;
		for(int i=0;i<wh;i++)
		{
			if(map_out[i] != 0)
			{
				if(randomPattern[rn] > charTH )
				{
					map_out[i]=0;
					numHaveSub--;
				}
				rn++;
			}
		}
	}

//	printf("PixelSelector: have %.2f%%, need %.2f%%. KEEPCURR with pot %d -> %d. Subsampled to %.2f%%\n",
//			100*numHave/(float)(wG[0]*hG[0]),
//			100*numWant/(float)(wG[0]*hG[0]),
//			currentPotential,
//			idealPotential,
//			100*numHaveSub/(float)(wG[0]*hG[0]));
	currentPotential = idealPotential;


	if(plot)
	{
		int w = wG[0];
		int h = hG[0];


		MinimalImageB3 img(w,h);

		for(int i=0;i<w*h;i++)
		{
			float c = fh->dI[i][0]*0.7;
			if(c>255) c=255;
			img.at(i) = Vec3b(c,c,c);
		}
		IOWrap::displayImage("Selector Image", &img);

		for(int y=0; y<h;y++)
			for(int x=0;x<w;x++)
			{
				int i=x+y*w;
				if(map_out[i] == 1)
					img.setPixelCirc(x,y,Vec3b(0,255,0));
				else if(map_out[i] == 2)
					img.setPixelCirc(x,y,Vec3b(255,0,0));
				else if(map_out[i] == 4)
					img.setPixelCirc(x,y,Vec3b(0,0,255));
			}
		IOWrap::displayImage("Selector Pixels", &img);
	}

	return numHaveSub;
}


/**
 * @brief 在当前帧上选择符合条件的像素
 * 
 * @param fh FrameHessian对象
 * @param map_out 筛选出的像素图
 * @param pot 选点的范围大小(pot内选一个点) default=3
 * @param thFactor 阈值因子 default=2
 * @return Eigen::Vector3i [out] 不同金字塔层级选择的特征像素数量
 */
Eigen::Vector3i PixelSelector::select(const FrameHessian* const fh,
		float* map_out, int pot, float thFactor)
{

	// const 在*左, 指针内容不可改, 在*右指针不可改
	// 等价const Eigen::Vector3f * const
	// 取出第0层图像像素、水平梯度dx、垂直梯度dy
	Eigen::Vector3f const * const map0 = fh->dI;

	// 取出第0、1、2层图像的梯度平方和(dx^2 + dy^2)
	float * mapmax0 = fh->absSquaredGrad[0];
	float * mapmax1 = fh->absSquaredGrad[1];
	float * mapmax2 = fh->absSquaredGrad[2];

	// 不同层的图像大小
	int w = wG[0];
	int w1 = wG[1];
	int w2 = wG[2];
	int h = hG[0];

	// 选取16个方向
	const Vec2f directions[16] = {
			// 90.0'
	         Vec2f(0,    1.0000),
			// 67.5'
	         Vec2f(0.3827,    0.9239),
			// 78.75'
	         Vec2f(0.1951,    0.9808),
			// 22.5'
	         Vec2f(0.9239,    0.3827),
			// 45.0'
	         Vec2f(0.7071,    0.7071),
			// -67.5'
	         Vec2f(0.3827,   -0.9239),
			// 33.75'
	         Vec2f(0.8315,    0.5556),
			// -33.75'
	         Vec2f(0.8315,   -0.5556),
			// -56.25'
	         Vec2f(0.5556,   -0.8315),
			// 11.25'
	         Vec2f(0.9808,    0.1951),
			// -22.5'
	         Vec2f(0.9239,   -0.3827),
			// -45.0'
	         Vec2f(0.7071,   -0.7071),
			// 56.25'
	         Vec2f(0.5556,    0.8315),
	        // -11.25'
			 Vec2f(0.9808,   -0.1951),
			// 0'
	         Vec2f(1.0000,    0.0000),
			// -78.75'
	         Vec2f(0.1951,   -0.9808)};

	// 分配map_out内存
	memset(map_out,0,w*h*sizeof(PixelSelectorStatus));

	// 金字塔层级的阈值下降权重
	// 第1层，default=0.75f
	float dw1 = setting_gradDownweightPerLevel;
	// 第2层
	float dw2 = dw1*dw1;

	// 统计不同金字塔层级下满足条件的特征像素数量
	int n3=0, n2=0, n4=0;
	// 遍历每个4potx4pot pixel块
	for(int y4=0;y4<h;y4+=(4*pot)) for(int x4=0;x4<w;x4+=(4*pot))
	{
		// 保证4potx4pot pixel块在图像范围内
		int my3 = std::min((4*pot), h-y4);
		int mx3 = std::min((4*pot), w-x4);
		// 记录最佳4potx4pot索引和值
		int bestIdx4=-1; float bestVal4=0;
		// 取低4位保证索引范围为[0,15]，随机选择directions中的一个方向
		Vec2f dir4 = directions[randomPattern[n2] & 0xF];
		// 遍历4potx4pot pixel块内的每个2potx2pot pixel块
		for(int y3=0;y3<my3;y3+=(2*pot)) for(int x3=0;x3<mx3;x3+=(2*pot))
		{
			// 计算2potx2pot pixel块的坐标
			int x34 = x3+x4;
			int y34 = y3+y4;
			// 保证2potx2pot pixel块在图像范围内
			int my2 = std::min((2*pot), h-y34);
			int mx2 = std::min((2*pot), w-x34);
			// 记录最佳2potx2pot索引和值
			int bestIdx3=-1; float bestVal3=0;
			// 随机选择directions中的一个方向
			Vec2f dir3 = directions[randomPattern[n2] & 0xF];
			// 遍历2potx2pot pixel块内的每个1potx1pot pixel块
			for(int y2=0;y2<my2;y2+=pot) for(int x2=0;x2<mx2;x2+=pot)
			{
				// 计算1potx1pot pixel块的坐标
				int x234 = x2+x34;
				int y234 = y2+y34;
				// 保证1potx1pot pixel块在图像范围内
				int my1 = std::min(pot, h-y234);
				int mx1 = std::min(pot, w-x234);
				// 记录最佳1potx1pot索引和值
				int bestIdx2=-1; float bestVal2=0;
				// 随机选择directions中的一个方向
				Vec2f dir2 = directions[randomPattern[n2] & 0xF];
				// 遍历1potx1pot pixel块内的每个pixel
				for(int y1=0;y1<my1;y1+=1) for(int x1=0;x1<mx1;x1+=1)
				{
					assert(x1+x234 < w);
					assert(y1+y234 < h);
					// 计算当前pixel的索引
					int idx = x1+x234 + w*(y1+y234);
					// 计算当前pixel的坐标
					int xf = x1+x234;
					int yf = y1+y234;

					// 保证当前pixel在图像范围内
					if(xf<4 || xf>=w-5 || yf<4 || yf>h-4) 
						continue;

					// 获取当前pixel的阈值
					// 第0层
                    float pixelTH0 = thsSmoothed[xf / bW + (yf / bH) * thsStep];
					// 计算按权重降低后的阈值
					// 第1层
					float pixelTH1 = pixelTH0*dw1;
					// 第2层
					float pixelTH2 = pixelTH1*dw2;

					// 获取当前pixel的梯度平方和
					float ag0 = mapmax0[idx];
					// 当前pixel的梯度平方和大于阈值
					if(ag0 > pixelTH0*thFactor)
					{
						// 获取当前pixel的水平、垂直梯度 dx, dy
						Vec2f ag0d = map0[idx].tail<2>();
						// 计算梯度与方向的点积
						// |[dx dy]*[dir_x dir_y]|
						float dirNorm = fabsf((float)(ag0d.dot(dir2)));
						// 判断是否选择方向分布进行判断
						if(!setting_selectDirectionDistribution) dirNorm = ag0;

						// 更新1potx1pot最佳值和索引
						// 并标记2potx2pot和4potx4pot的索引
						if(dirNorm > bestVal2)
						{ bestVal2 = dirNorm; bestIdx2 = idx; bestIdx3 = -2; bestIdx4 = -2;}
					}
					// 如果1potx1pot内已经找到最佳值，跳过
					if(bestIdx3==-2) continue;

					// 取出当前pixel对应第1层的梯度平方和
					float ag1 = mapmax1[(int)(xf*0.5f+0.25f) + (int)(yf*0.5f+0.25f)*w1];
					// 当前pixel的梯度平方和大于阈值
					if(ag1 > pixelTH1*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir3)));
						if(!setting_selectDirectionDistribution) dirNorm = ag1;

						if(dirNorm > bestVal3)
						{ bestVal3 = dirNorm; bestIdx3 = idx; bestIdx4 = -2;}
					}
					// 如果2potx2pot内已经找到最佳值，跳过
					if(bestIdx4==-2) continue;

					// 取出当前pixel对应第2层的梯度平方和
					float ag2 = mapmax2[(int)(xf*0.25f+0.125) + (int)(yf*0.25f+0.125)*w2];
					if(ag2 > pixelTH2*thFactor)
					{
						Vec2f ag0d = map0[idx].tail<2>();
						float dirNorm = fabsf((float)(ag0d.dot(dir4)));
						if(!setting_selectDirectionDistribution) dirNorm = ag2;

						if(dirNorm > bestVal4)
						{ bestVal4 = dirNorm; bestIdx4 = idx; }
					}
				}
				// 如果1potx1pot内找到最佳值，标记2potx2pot和4potx4pot的索引
				if(bestIdx2>0)
				{
					// 记录最佳值像素属于的金字塔层级
					map_out[bestIdx2] = 1;
					// 设置2potx2pot的最佳值
					bestVal3 = 1e10;
					// 更新1potx1pot特征像素数量
					n2++;
				}
			}
			// 如果2potx2pot内找到最佳值，标记4potx4pot的索引
			if(bestIdx3>0)
			{
				map_out[bestIdx3] = 2;
				bestVal4 = 1e10;
				n3++;
			}
		}

		if(bestIdx4>0)
		{
			map_out[bestIdx4] = 4;
			n4++;
		}
	}

	// 返回在1potx1pot、2potx2pot、4potx4pot(0、1、2层)内找到的特征像素数量
	return Eigen::Vector3i(n2,n3,n4);
}


}

