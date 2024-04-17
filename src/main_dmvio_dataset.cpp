/**
* This file is based on the file main_dso_pangolin.cpp of the project DSO written by Jakob Engel.
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

// Main file for running on datasets, based on the main file of DSO.

#include "util/MainSettings.h"
#include <thread>
#include <locale.h>
#include <signal.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>

#include "IOWrapper/Output3DWrapper.h"
#include "IOWrapper/ImageDisplay.h"


#include <boost/thread.hpp>
#include "dso/util/settings.h"
#include "dso/util/globalFuncs.h"
#include "dso/util/DatasetReader.h"
#include "dso/util/globalCalib.h"
#include "util/TimeMeasurement.h"

#include "dso/util/NumType.h"
#include "FullSystem/FullSystem.h"
#include "OptimizationBackend/MatrixAccumulators.h"
#include "FullSystem/PixelSelector2.h"

#include <util/SettingsUtil.h>

#include "IOWrapper/Pangolin/PangolinDSOViewer.h"
#include "IOWrapper/OutputWrapper/SampleOutputWrapper.h"

std::string gtFile = "";
std::string source = "";
std::string imuFile = "";

bool reverse = false;
int start = 0;
int end = 100000;
// 最大预加载图像数
int maxPreloadImages = 0; // If set we only preload if there are less images to be loade.
bool useSampleOutput = false;

using namespace dso;


dmvio::MainSettings mainSettings;
dmvio::IMUCalibration imuCalibration;
dmvio::IMUSettings imuSettings;

/**
 * @brief 捕获到ctrl+C信号时的处理函数
 * 
 * @param s 
 */
void my_exit_handler(int s)
{
    printf("Caught signal %d\n", s);
    // 结束当前进程，及其全部子线程
    exit(1);
}

/**
 * @brief 用于处理ctrl+C信号的函数
 * 
 */
void exitThread()
{
    struct sigaction sigIntHandler;
    // 设置信号处理函数
    sigIntHandler.sa_handler = my_exit_handler;
    // 初始化sa_mask为空，表示无信号被屏蔽(阻塞)
    sigemptyset(&sigIntHandler.sa_mask);
    // 设置sa_flags为0，表示默认行为
    sigIntHandler.sa_flags = 0;
    // 注册信号处理函数，当收到SIGINT信号时，调用my_exit_handler函数
    sigaction(SIGINT, &sigIntHandler, NULL);

    // 捕获信号，当收到SIGINT信号时，调用my_exit_handler函数
    while(true) pause();
}





/**
 * @brief 
 * 
 * @param reader [in] 数据加载对象，从数据集中读取预加载的图像 
 * @param viewer [in] 可视化对象
 */
void run(ImageFolderReader* reader, IOWrap::PangolinDSOViewer* viewer)
{
    // 设置光度校准但是未获取到gamma校准参数
    if(setting_photometricCalibration > 0 && reader->getPhotometricGamma() == 0)
    {
        // 输出错误并结束进程
        printf("ERROR: dont't have photometric calibation. Need to use commandline options mode=1 or mode=2 ");
        exit(1);
    }


    int lstart = start;
    int lend = end;
    // 用于控制图像数据处理的方向，1表示正向，-1表示反向
    int linc = 1;
    // 是否反向处理图像数据，[end, start]，不支持IMU
    if(reverse)
    {
        assert(!setting_useIMU); // Reverse is not supported with IMU data at the moment!
        printf("REVERSE!!!!");
        // 根据图像数据设置lstart, lend, linc
        lstart = end - 1;
        if(lstart >= reader->getNumImages())
            lstart = reader->getNumImages() - 1;
        lend = start;
        linc = -1;
    }

    // 是否尽可能快地处理数据集数据
    bool linearizeOperation = (mainSettings.playbackSpeed == 0);

    // 针对非实时模式，设置相邻关键帧间的最小帧数
    if(linearizeOperation && setting_minFramesBetweenKeyframes < 0)
    {
        setting_minFramesBetweenKeyframes = -setting_minFramesBetweenKeyframes;
        std::cout << "Using setting_minFramesBetweenKeyframes=" << setting_minFramesBetweenKeyframes
                  << " because of non-realtime mode." << std::endl;
    }

    // 创建处理图像和IMU数据对象
    FullSystem* fullSystem = new FullSystem(linearizeOperation, imuCalibration, imuSettings);
    // 设置gamma校准参数
    fullSystem->setGammaFunction(reader->getPhotometricGamma());

    // 不进行可视化显示
    if(viewer != 0)
    {
        // 将PangolinDSOViewer对象的原始指针添加到outputWrapper中
        fullSystem->outputWrapper.push_back(viewer);
    }

    // 创建SampleOutputWrapper对象的unique_ptr指针
    std::unique_ptr<IOWrap::SampleOutputWrapper> sampleOutPutWrapperPtr;
    // 如果选择使用SampleOutputWrapper对象
    if(useSampleOutput)
    {
        // 更新unique_ptr指向的对象，保证旧对象被正确删除
        sampleOutPutWrapperPtr.reset(new IOWrap::SampleOutputWrapper());
        // 将SampleOutputWrapper对象的原始指针添加到outputWrapper中
        // 即使unique_ptr对象被销毁，原始指针仍然存在
        fullSystem->outputWrapper.push_back(sampleOutPutWrapperPtr.get());
    }

    // 初始化容器，存储图像id和对应的时间戳
    std::vector<int> idsToPlay;
    std::vector<double> timesToPlayAt;
    // 遍历图像id和对应的时间戳，保存到容器中
    for(int i = lstart; i >= 0 && i < reader->getNumImages() && linc * i < linc * lend; i += linc)
    {
        // 将当前图像id加入idsToPlay容器
        idsToPlay.push_back(i);
        // 如果timesToPlayAt容器为空，从0开始计算时间戳
        if(timesToPlayAt.size() == 0)
        {
            timesToPlayAt.push_back((double) 0);
        }
        else
        {
            // 进入该分支，idsToPlay容器中至少有2个元素
            double tsThis = reader->getTimestamp(idsToPlay[idsToPlay.size() - 1]);
            double tsPrev = reader->getTimestamp(idsToPlay[idsToPlay.size() - 2]);
            // 计算当前帧相对于上一帧的时间戳差值，更新当前帧的时间戳
            // timesToPlayAt容器中存储的是每一帧的时间戳相对于第一帧的时间戳的差值
            timesToPlayAt.push_back(timesToPlayAt.back() + fabs(tsThis - tsPrev) / mainSettings.playbackSpeed);
        }
    }

    // 判断是否需要预加载图像
    if(mainSettings.preload && maxPreloadImages > 0)
    {
        // 图像总数大于最大预加载图像数，不进行预加载
        if(reader->getNumImages() > maxPreloadImages)
        {
            printf("maxPreloadImages EXCEEDED! NOT PRELOADING!\n");
            mainSettings.preload = false;
        }
    }

    // 初始化容器，存储ImageAndExposure对象指针，预加载的图像和曝光时间
    std::vector<ImageAndExposure*> preloadedImages;
    // 预加载图像
    if(mainSettings.preload)
    {
        printf("LOADING ALL IMAGES!\n");
        for(int ii = 0; ii < (int) idsToPlay.size(); ii++)
        {
            // 取出图像id
            int i = idsToPlay[ii];
            // 将ImageAndExposure对象指针加入preloadedImages
            // 在该部分对图像进行了去畸变
            preloadedImages.push_back(reader->getImage(i));
        }
    }

    struct timeval tv_start;
    // 获取当前UTC时间
    gettimeofday(&tv_start, NULL);
    // 获取开始的CPU时间
    clock_t started = clock();
    // 记录开始时间到初始化完成时间的offset
    double sInitializerOffset = 0;

    // 加载数据集的GT
    bool gtDataThere = reader->loadGTData(gtFile);

    bool imuDataSkipped = false;
    dmvio::IMUData skippedIMUData;
    // 遍历图像id
    for(int ii = 0; ii < (int) idsToPlay.size(); ii++)
    {
        // 如果系统未初始化，重置开始时间
        if(!fullSystem->initialized)    // if not initialized: reset start time.
        {
            gettimeofday(&tv_start, NULL);
            started = clock();
            sInitializerOffset = timesToPlayAt[ii];
        }

        int i = idsToPlay[ii];

        // 读取图像
        ImageAndExposure* img;
        // 根据是否预加载，从不同地方读取图像
        if(mainSettings.preload)
            img = preloadedImages[ii];
        else
            img = reader->getImage(i);


        bool skipFrame = false;
        // 是否设置图像播放速率
        if(mainSettings.playbackSpeed != 0)
        {
            // 获取当前时间
            struct timeval tv_now;
            gettimeofday(&tv_now, NULL);
            // 计算开始时间到当前时间的差值
            double sSinceStart = sInitializerOffset + ((tv_now.tv_sec - tv_start.tv_sec) +
                                                       (tv_now.tv_usec - tv_start.tv_usec) / (1000.0f * 1000.0f));
            // 当前时间小于预定播放时间，等待
            if(sSinceStart < timesToPlayAt[ii])
                usleep((int) ((timesToPlayAt[ii] - sSinceStart) * 1000 * 1000));
            // 当前时间大于预定播放时间+阈值，设定标记跳过当前帧
            else if(sSinceStart > timesToPlayAt[ii] + 0.5 + 0.1 * (ii % 2))
            {
                printf("SKIPFRAME %d (play at %f, now it is %f)!\n", ii, timesToPlayAt[ii], sSinceStart);
                skipFrame = true;
            }
        }

        dmvio::GTData data;
        bool found = false;
        // 读取当前帧对应的GT
        if(gtDataThere)
        {
            data = reader->getGTData(i, found);
        }

        std::unique_ptr<dmvio::IMUData> imuData;
        if(setting_useIMU)
        {
            // 使用当前帧对应的IMU数据创建IMUData对象，并获取其unique_ptr指针
            imuData = std::make_unique<dmvio::IMUData>(reader->getIMUData(i));
        }
        // 不跳过当前帧
        if(!skipFrame)
        {
            // 如果IMU数据被跳过，且存在IMU数据
            if(imuDataSkipped && imuData)
            {
                imuData->insert(imuData->begin(), skippedIMUData.begin(), skippedIMUData.end());
                skippedIMUData.clear();
                imuDataSkipped = false;
            }
            fullSystem->addActiveFrame(img, i, imuData.get(), (gtDataThere && found) ? &data : 0);
            // 更新可视化位姿
            if(gtDataThere && found && !disableAllDisplay)
            {
                viewer->addGTCamPose(data.pose);
            }
        }
        // 跳过当前帧且存在IMU数据
        else if(imuData)
        {
            // 标记IMU数据被跳过
            imuDataSkipped = true;
            // 将imuData全部数据加入skippedIMUData容器
            skippedIMUData.insert(skippedIMUData.end(), imuData->begin(), imuData->end());
        }

        // 删除img对象
        delete img;

        if(fullSystem->initFailed || setting_fullResetRequested)
        {
            if(ii < 250 || setting_fullResetRequested)
            {
                printf("RESETTING!\n");
                std::vector<IOWrap::Output3DWrapper*> wraps = fullSystem->outputWrapper;
                delete fullSystem;
                for(IOWrap::Output3DWrapper* ow : wraps) ow->reset();

                fullSystem = new FullSystem(linearizeOperation, imuCalibration, imuSettings);
                fullSystem->setGammaFunction(reader->getPhotometricGamma());
                fullSystem->outputWrapper = wraps;

                setting_fullResetRequested = false;
            }
        }

        // 如果viewer不为空且用户关闭了窗口，退出
        if(viewer != nullptr && viewer->shouldQuit())
        {
            std::cout << "User closed window -> Quit!" << std::endl;
            break;
        }

        // 系统跟踪丢失，退出
        if(fullSystem->isLost)
        {
            printf("LOST!!\n");
            break;
        }

    }
    // 等待mapping线程结束
    fullSystem->blockUntilMappingIsFinished();
    // 获取结束的UTC时间
    clock_t ended = clock();
    struct timeval tv_end;
    gettimeofday(&tv_end, NULL);

    // 输出结果
    fullSystem->printResult(imuSettings.resultsPrefix + "result.txt", false, false, true);
    fullSystem->printResult(imuSettings.resultsPrefix + "resultKFs.txt", true, false, false);
    fullSystem->printResult(imuSettings.resultsPrefix + "resultScaled.txt", false, true, true);

    dmvio::TimeMeasurement::saveResults(imuSettings.resultsPrefix + "timings.txt");


    int numFramesProcessed = abs(idsToPlay[0] - idsToPlay.back());
    double numSecondsProcessed = fabs(reader->getTimestamp(idsToPlay[0]) - reader->getTimestamp(idsToPlay.back()));
    double MilliSecondsTakenSingle = 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC);
    double MilliSecondsTakenMT = sInitializerOffset + ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f +
                                                       (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f);
    printf("\n======================"
           "\n%d Frames (%.1f fps)"
           "\n%.2fms per frame (single core); "
           "\n%.2fms per frame (multi core); "
           "\n%.3fx (single core); "
           "\n%.3fx (multi core); "
           "\n======================\n\n",
           numFramesProcessed, numFramesProcessed / numSecondsProcessed,
           MilliSecondsTakenSingle / numFramesProcessed,
           MilliSecondsTakenMT / (float) numFramesProcessed,
           1000 / (MilliSecondsTakenSingle / numSecondsProcessed),
           1000 / (MilliSecondsTakenMT / numSecondsProcessed));
    fullSystem->printFrameLifetimes();
    if(setting_logStuff)
    {
        std::ofstream tmlog;
        tmlog.open("logs/time.txt", std::ios::trunc | std::ios::out);
        tmlog << 1000.0f * (ended - started) / (float) (CLOCKS_PER_SEC * reader->getNumImages()) << " "
              << ((tv_end.tv_sec - tv_start.tv_sec) * 1000.0f + (tv_end.tv_usec - tv_start.tv_usec) / 1000.0f) /
                 (float) reader->getNumImages() << "\n";
        tmlog.flush();
        tmlog.close();
    }

    for(IOWrap::Output3DWrapper* ow : fullSystem->outputWrapper)
    {
        ow->join();
    }

    // 销毁fullSystem对象
    printf("DELETE FULLSYSTEM!\n");
    delete fullSystem;

    // 销毁reader对象
    printf("DELETE READER!\n");
    delete reader;

    printf("EXIT NOW!\n");
}

int main(int argc, char** argv)
{
    setlocale(LC_ALL, "C");

#ifdef DEBUG
    std::cout << "DEBUG MODE!" << std::endl;
#endif

    bool use16Bit = false;

    // 创建共享指针管理SettingsUtil对象
    auto settingsUtil = std::make_shared<dmvio::SettingsUtil>();

    // Create Settings files.
    imuSettings.registerArgs(*settingsUtil);
    imuCalibration.registerArgs(*settingsUtil);
    mainSettings.registerArgs(*settingsUtil);

    // Dataset specific arguments. For other commandline arguments check out MainSettings::parseArgument,
    // MainSettings::registerArgs, IMUSettings.h and IMUInitSettings.h
    // 注册参数名称
    settingsUtil->registerArg("files", source);
    settingsUtil->registerArg("start", start);
    settingsUtil->registerArg("end", end);
    settingsUtil->registerArg("imuFile", imuFile);
    settingsUtil->registerArg("gtFile", gtFile);
    settingsUtil->registerArg("sampleoutput", useSampleOutput);
    settingsUtil->registerArg("reverse", reverse);
    settingsUtil->registerArg("use16Bit", use16Bit);
    settingsUtil->registerArg("maxPreloadImages", maxPreloadImages);

    // This call will parse all commandline arguments and potentially also read a settings yaml file if passed.
    // 解析命令行参数
    mainSettings.parseArguments(argc, argv, *settingsUtil);

    // 从文件读取IMU配置信息
    if(mainSettings.imuCalibFile != "")
    {
        imuCalibration.loadFromFile(mainSettings.imuCalibFile);
    }

    // Print settings to commandline and file.
    // 打印设置信息到命令行和文件
    std::cout << "Settings:\n";
    settingsUtil->printAllSettings(std::cout);
    {
        std::ofstream settingsStream;
        settingsStream.open(imuSettings.resultsPrefix + "usedSettingsdso.txt");
        settingsUtil->printAllSettings(settingsStream);
    }

    // hook crtl+C.
    // 创建线程，用于处理ctrl+C信号
    boost::thread exThread = boost::thread(exitThread);

    // 创建ImageFolderReader对象reader
    // 输入参数数据集路径，相机内参，gamma校准参数，渐晕校准参数，是否使用16位图像
    ImageFolderReader* reader = new ImageFolderReader(source, mainSettings.calib, mainSettings.gammaCalib, mainSettings.vignette, use16Bit);
    // 加载IMU数据
    reader->loadIMUData(imuFile);
    // 设置全局相机标定参数
    reader->setGlobalCalibration();

    // 可视化显示
    // 在当前线程执行viewer可视化线程，创建新的线程执行run()
    if(!disableAllDisplay)
    {
        // 创建PangolinDSOViewer对象viewer
        IOWrap::PangolinDSOViewer* viewer = new IOWrap::PangolinDSOViewer(wG[0], hG[0], false, settingsUtil,
                                                                          nullptr);

        // 创建线程runThread运行run()，并将reader, viewer作为参数传入
        boost::thread runThread = boost::thread(boost::bind(run, reader, viewer));

        viewer->run();

        delete viewer;

        // Make sure that the destructor of FullSystem, etc. finishes, so all log files are properly flushed.
        // 由于在单独线程中执行run()，进行阻塞操作，等待runThread线程结束
        runThread.join();
    }
    // 关闭可视化显示
    // 在当前线程执行run()
    else
    {
        // 直接运行run()，并将reader, nullptr作为参数传入
        // 在当前线程中执行run()，会在执行完成返回前阻塞
        run(reader, 0);
    }


    return 0;
}
