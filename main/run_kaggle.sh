###########################################################
# Wandb
###########################################################
wandb login c74133df8c2cf575304acf8a99fe03ab74b6fe6a

############################################################
# ltcc
############################################################
# Lucky / ltcc  ===================================================================================================================
# python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky

############################################################
# prcc
############################################################
python main.py --config_file "config/method.yml" TASK.NOTES=Debug TASK.NAME=Lucky DATA.TRAIN_DATASET=prcc DATA.TRAIN_ROOT=/kaggle/input/prcc-dataset/prcc/rgb/
