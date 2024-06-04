#! /usr/bin/zsh

cd /home/zmxj/code/dm-vio/build

PATH_TO_DATASET="/home/zmxj/code/Datasets/Tumvi/dataset-corridor1_512_16"
PATH_TO_DMVIO="/home/zmxj/code/dm-vio"
PATH_TO_RESULTS="$PATH_TO_DMVIO/result"

echo "PATH_TO_DATASET = $PATH_TO_DATASET"
echo "PATH_TO_DMVIO = $PATH_TO_DMVIO"
echo "PATH_TO_RESULTS = $PATH_TO_RESULTS"

bin/dmvio_dataset
    files=$PATH_TO_DATASET/dso/cam0/images              
    vignette=$PATH_TO_DATASET/dso/cam0/vignette.png
    imuFile=$PATH_TO_DATASET/dso/imu.txt
    gtFile=$PATH_TO_DATASET/dso/gt_imu.csv
    calib=$PATH_TO_DMVIO/configs/tumvi_calib/camera02.txt
    gamma=$PATH_TO_DMVIO/configs/tumvi_calib/pcalib.txt
    imuCalib=$PATH_TO_DMVIO/configs/tumvi_calib/camchain.yaml
    resultsPrefix=$PATH_TO_RESULTS/
    settingsFile=$PATH_TO_DMVIO/configs/tumvi.yaml
    mode=0
    use16Bit=1
    preset=0                                                        # use 1 for realtime
    nogui=1                                                         # use 1 to enable GUI
    start=2                                                         

