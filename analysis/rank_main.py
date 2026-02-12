import os

from PIL import Image, ImageDraw, ImageFont


def merge_images_vertically(folder1, folder2, output_folder):
    """
    将两个文件夹中对应名称的图片上下拼接，并添加文件名标题，保存到指定文件夹

    Args:
        folder1: 第一个图片文件夹路径
        folder2: 第二个图片文件夹路径
        output_folder: 输出文件夹路径
    """
    # 创建输出文件夹（不存在则创建）
    os.makedirs(output_folder, exist_ok=True)

    # 获取两个文件夹中的图片文件（只处理常见图片格式）
    image_extensions = (".jpg", ".jpeg", ".png", ".bmp", ".gif")
    files1 = [f for f in os.listdir(folder1) if f.lower().endswith(image_extensions)]
    files2 = [f for f in os.listdir(folder2) if f.lower().endswith(image_extensions)]

    # 确保文件列表按名称排序，保证一一对应
    files1.sort()
    files2.sort()

    # 检查两个文件夹的文件数量是否一致
    if len(files1) != len(files2):
        print(f"警告：两个文件夹的图片数量不一致！文件夹1有{len(files1)}张，文件夹2有{len(files2)}张")
        # 只处理到数量较少的文件夹的文件数
        min_count = min(len(files1), len(files2))
        files1 = files1[:min_count]
        files2 = files2[:min_count]

    # 设置标题字体（使用系统默认字体，避免字体缺失问题）
    try:
        # 尝试加载系统字体（Windows）
        font = ImageFont.truetype("simhei.ttf", 40)
    except:
        try:
            # 尝试加载系统字体（Linux/Mac）
            font = ImageFont.truetype("/Library/Fonts/Arial.ttf", 40)
        except:
            # 无指定字体时使用默认字体
            font = ImageFont.load_default(size=40)

    # 遍历并拼接每一对图片
    for idx, (file1, file2) in enumerate(zip(files1, files2)):
        try:
            # 构建文件完整路径
            path1 = os.path.join(folder1, file1)
            path2 = os.path.join(folder2, file2)

            # 读取图片
            img1 = Image.open(path1).convert("RGB")
            img2 = Image.open(path2).convert("RGB")

            # 统一图片宽度（取两张图中较大的宽度）
            target_width = max(img1.width, img2.width)

            # 调整图片尺寸（保持宽高比）
            img1 = img1.resize((target_width, int(img1.height * target_width / img1.width)))
            img2 = img2.resize((target_width, int(img2.height * target_width / img2.width)))

            # 创建新画布（高度为两张图高度之和 + 标题栏高度）
            title_height = 60
            new_height = img1.height + img2.height + title_height
            new_img = Image.new("RGB", (target_width, new_height), "white")

            # 粘贴第一张图片（标题栏下方）
            new_img.paste(img1, (0, title_height))
            # 粘贴第二张图片（第一张图片下方）
            new_img.paste(img2, (0, img1.height + title_height))

            # 添加标题（文件名）
            draw = ImageDraw.Draw(new_img)
            # 获取文件名（去除扩展名）
            file_name = os.path.splitext(file1)[0]
            # 计算文字位置（居中）
            text_bbox = draw.textbbox((0, 0), file_name, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = (target_width - text_width) // 2
            text_y = 10  # 标题栏内的垂直位置
            # 绘制文字
            draw.text((text_x, text_y), file_name, fill="black", font=font)

            # 保存新图片
            output_filename = f"merged_{file_name}.jpg"
            output_path = os.path.join(output_folder, output_filename)
            new_img.save(output_path, quality=95)

            print(f"成功处理：{file1} + {file2} -> {output_filename}")

        except Exception as e:
            print(f"处理第{idx+1}对图片失败（{file1}/{file2}）：{str(e)}")
            continue

    print(f"\n处理完成！共处理{len(files1)}对图片，保存至：{os.path.abspath(output_folder)}")


# -------------------------- 配置参数 --------------------------
# 请修改以下三个路径为你实际的文件夹路径
FOLDER1_PATH = "/Users/drhy/Desktop/temp/result_baseline/outputs/rank/"  # 第一个图片文件夹
FOLDER2_PATH = "/Users/drhy/Desktop/temp/result_method/outputs/rank/"  # 第二个图片文件夹
OUTPUT_FOLDER = "./merged_photos"  # 输出文件夹（当前目录下）
# -------------------------------------------------------------

# 执行图片拼接
if __name__ == "__main__":
    # 检查源文件夹是否存在
    if not os.path.exists(FOLDER1_PATH):
        print(f"错误：文件夹 {FOLDER1_PATH} 不存在！")
    elif not os.path.exists(FOLDER2_PATH):
        print(f"错误：文件夹 {FOLDER2_PATH} 不存在！")
    else:
        merge_images_vertically(FOLDER1_PATH, FOLDER2_PATH, OUTPUT_FOLDER)
