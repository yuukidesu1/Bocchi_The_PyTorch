from PIL import Image, ImageOps
import os

# 输入图像文件和标签文件的路径
image_path = "D:/PyCharm/UNet/DataSet/data_CitySegmentation/Berlin/raster/berlin_0_0_raster.png"
label_path = "D:/PyCharm/UNet/DataSet/data_CitySegmentation/Berlin/label/berlin_0_0_label.png"

# 创建保存分割后图片的文件夹
output_directory = "output_images"
os.makedirs(output_directory, exist_ok=True)

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

# 遍历并分割图像
for i in range(3):
    for j in range(3):
        # 计算分割后的区域
        left = i * (new_width // 3)
        upper = j * (new_height // 3)
        right = left + (new_width // 3)
        lower = upper + (new_height // 3)

        # 裁剪图像块
        image_block = image.crop((left, upper, right, lower))
        label_block = label.crop((left, upper, right, lower))

        # 构建保存文件名
        filename = f"{i * 3 + j + 1}.png"

        # 保存分割后的图像和标签
        image_block.save(os.path.join(output_directory, filename))
        label_block.save(os.path.join(output_directory, f"label_{filename}"))

print("分割完成，并保存在文件夹:", output_directory)
