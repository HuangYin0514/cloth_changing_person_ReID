

# cp -rf main _历史代码/0_调试代码/121-L4_Cam_Correction_KD

# cp -rf main _历史代码/1_调试代码/122-L4_Cam_Correction_KD


# rm -rf main
# cp -rf _历史代码/1_调试代码/121-L4_Cam_Correction_KD main


####################################
# git
####################################
# 1. 查看状态
git status
# 2. 添加所有修改
git add .
# 3. 提交到本地（替换提交说明）
git commit -m "feat: 123-L4_Cam_Correction_KD"
# 4. 拉取远程最新代码（替换分支名）
git pull origin main
# 5. 推送到远程（替换分支名）
git push origin main

echo -e "\n✅ 成功：代码已提交并同步到远程分支！"
