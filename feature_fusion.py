import cv2
from matplotlib import pyplot as plt

# 读取RGB和红外图像
rgb_image_path = 'D:/PyCharm/UNet/DataSet/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'  # 替换为你的RGB图像路径
infrared_image_path = 'D:/PyCharm/UNet/DataSet/4_Ortho_RGBIR/top_potsdam_2_10_RGBIR.tif'  # 替换为你的红外图像路径

rgb_image = cv2.imread(rgb_image_path)
infrared_image = cv2.imread(infrared_image_path, cv2.IMREAD_GRAYSCALE)  # 红外图像通常是单通道的

# 显示RGB图像
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.title('RGB Image')
plt.imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
plt.axis('off')

# 显示红外图像
plt.subplot(1, 2, 2)
plt.title('Infrared Image')
plt.imshow(infrared_image, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()
