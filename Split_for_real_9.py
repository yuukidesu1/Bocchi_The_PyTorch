from PIL import Image, ImageOps
import os

# 创建保存分割后图片的文件夹
output_directory_raster = "output_images/raster"
output_directory_label = "output_images/label"
os.makedirs(output_directory_raster, exist_ok=True)
os.makedirs(output_directory_label, exist_ok=True)

# 获取输入图像的基础路径
base_path = "F:/PyCharm/Projects/UNet/DATASET/data_CitySegmentation/London/raster/london_{}_{}_raster.png"
base_label_path = "F:/PyCharm/Projects/UNet/DATASET/data_CitySegmentation/London/label/london_{}_{}_label.png"


num = 1
num_s = 1


# 遍历并分割图像
for i in range(5):
    for j in range(4):                                                              # B in 6
        # 构建图像文件和标签文件的路径
        image_path = base_path.format(i, j)
        label_path = base_label_path.format(i, j)

        # 打开输入图像和标签图像
        image = Image.open(image_path)
        label = Image.open(label_path)

        # 获取输入图像的尺寸
        width, height = image.size

        # 计算每块的宽度和高度
        block_width = width // 3
        block_height = height // 3

        # 计算镜像填充的像素数
        pad_width = (3 - width % 3) % 3
        pad_height = (3 - height % 3) % 3

        # 镜像填充图像
        image = ImageOps.expand(image, border=(pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))
        label = ImageOps.expand(label, border=(pad_width // 2, pad_height // 2, pad_width - pad_width // 2, pad_height - pad_height // 2))

        # 获取新的图像尺寸
        new_width, new_height = image.size

        # 初始化文件名变量
        if i == j == 0:
            filename_number = num
        else:
            filename_number = num_s
        num_s += 9

        # 遍历并分割图像
        for m in range(3):
            for n in range(3):
                # 计算分割后的区域
                left = m * (new_width // 3)
                upper = n * (new_height // 3)
                right = left + (new_width // 3)
                lower = upper + (new_height // 3)

                # 裁剪图像块
                image_block = image.crop((left, upper, right, lower))
                label_block = label.crop((left, upper, right, lower))

                # 构建保存文件名
                filename = f"{filename_number}.png"

                # 保存分割后的图像和标签到各自的文件夹
                image_block.save(os.path.join(output_directory_raster, filename))
                label_block.save(os.path.join(output_directory_label, f"label_{filename}"))

                # 增加文件名变量以准备下一次迭代
                filename_number += 1

print("分割完成，并保存在文件夹:", output_directory_raster, "和", output_directory_label)
