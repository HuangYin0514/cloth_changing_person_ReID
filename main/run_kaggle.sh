###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a

############################################################
# ltcc
############################################################
# =================== 完整模型  ==============================
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug MODEL.KD=0.002 

# =================== 参数分析  ==============================
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.000
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.001
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.003
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.004
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.005
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.006
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.007
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.008
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.009
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky MODEL.KD=0.010

# =================== 可视化  ==============================
mkdir -p results/outputs/models
gdown -O results/outputs/models/model_89.pth 1TEsdxgbhxxuQ5Jheej6kvNbjCfVp_i6Y 
python vis_main.py --config_file "config/method.yml" TASK.NOTES=237-Vis_rank TEST.RESUME_EPOCH=89
tar -czf ../result_method.tar.gz results
rm -rf results/outputs/*

# =================== 基线模型  ==============================
# python main.py --config_file "config/method.yml" TASK.NOTES=234-B MODEL.MODULE=Baseline

############################################################
# prcc
############################################################
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky DATA.TRAIN_DATASET=prcc DATA.TRAIN_ROOT=/kaggle/input/prcc-dataset/prcc/rgb/ MODEL.BACKBONE_TYPE=resnet50 
# OPTIMIZER.TOTAL_TRAIN_EPOCH=60