####################################
# 拷贝
####################################

# rm -rf main
# cp -rf _历史代码/1_调试代码/121-L4_Cam_Correction_KD main
# echo "\n✅ 成功：代码已恢复！"

####################################
# git
####################################
VERSION_NAME="123-L4_Cam_Correction_KD"

# cp -rf main _历史代码/0_调试代码/${VERSION_NAME}
cp -rf main "_历史代码/1_调试代码/${VERSION_NAME}"
echo "\n✅ 成功：代码已备份！"


git status # 1. 查看状态
git add . # 2. 添加所有修改
git commit -m "feat: 123-L4_Cam_Correction_KD" # 3. 提交到本地（替换提交说明）
git pull origin main # 4. 拉取远程最新代码
git push origin main # 5. 推送到远程\
echo -e "\n✅ 成功：代码已提交并同步到远程分支！"
