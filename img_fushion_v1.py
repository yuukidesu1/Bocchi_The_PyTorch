import cv2

# 读取RGB和红外图像
rgb_image_path = 'D:/PyCharm/UNet/DataSet/2_Ortho_RGB/top_potsdam_3_12_RGB.tif '  # 替换为你的RGB图像路径
infrared_image_path = 'D:/PyCharm/UNet/DataSet/4_Ortho_RGBIR/top_potsdam_2_10_RGBIR.tif'  # 替换为你的红外图像路径

rgb_image = cv2.imread(rgb_image_path)
infrared_image = cv2.imread(infrared_image_path, cv2.IMREAD_GRAYSCALE)  # 红外图像通常是单通道的

# 将红外图像与RGB图像进行融合
blended_image = cv2.addWeighted(rgb_image, 1, cv2.cvtColor(infrared_image, cv2.COLOR_GRAY2BGR), 0, 0)

# 保存融合后的图像为文件
output_path = 'tif_images/out_putimg_3.png'  # 替换为你想要保存的图像路径及格式
cv2.imwrite(output_path, blended_image)
