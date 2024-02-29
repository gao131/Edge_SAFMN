# from PIL import Image
#
# # 打开.bmp文件
# image_path = "C:\\Users\\22359\\Desktop\\超分辨率重建\\Train_images\\train4\\1201.BMP"
# image = Image.open(image_path)
# print(image)
# image.save("1.png")
#
# # 显示图像信息
# print("图像格式:", image.format)
# print("图像模式:", image.mode)
# print("图像尺寸:", image.size)
#
# # 显示图像
# image.show()
#
# # 关闭图像
# image.close()

# from PIL import Image
# import cv2
# import numpy as np
#
# def bicubic_interpolation(img, new_size):
#     return img.resize(new_size, Image.BICUBIC)
#
# def bilateral_filter(img, d, sigma_color, sigma_space):
#     img_np = np.array(img)
#     img_filtered_np = cv2.bilateralFilter(img_np, d, sigma_color, sigma_space)
#     img_filtered = Image.fromarray(img_filtered_np)
#     return img_filtered
#
# # 打开.bmp文件
# image_path = "1.png"
# image = Image.open(image_path).convert("RGB")
#
# # 获取图像原始尺寸
# original_size = image.size
# print("原始图像尺寸:", original_size)
#
# # 计算新尺寸（二分之一大小）
# new_size = (original_size[0] * 2, original_size[1] * 2)
# print("新图像尺寸:", new_size)
#
# # 进行双三次插值上采样
# resized_image = bicubic_interpolation(image, new_size)
#
# # 添加双边滤波
# d = 15  # 邻域直径
# sigma_color = 75  # 颜色空间标准差
# sigma_space = 75  # 坐标空间标准差
# filtered_image = bilateral_filter(resized_image, d, sigma_color, sigma_space)
#
# # 保存上采样并滤波后的图像
# filtered_image.save("2_filtered.png")

#
# import os
# from PIL import Image
# import shutil
# import numpy as np
# import pdb
#
#
#
#
# def add_gaussian_noise(img):
#     # 将图像转换为NumPy数组
#     img_array = np.array(img)
#
#     row, col, ch = img_array.shape
#     # 生成高斯噪声
#     # gauss = np.random.normal(scale=1,size=img_array.shape)
#     gauss = np.random.normal(scale=10,size=(row, col, int(ch/3)))
#
#     # # 生成泊松噪声
#     # poisson_intensity = 10
#     # poisson_noise = np.random.poisson(poisson_intensity,size=(row, col, int(ch/3)))
#
#     # nois = gauss + poisson_noise
#     nois = gauss
#     nois = np.concatenate([nois, nois, nois], axis=2)
#
#     # 将噪声添加到图像中
#     noisy_image_array = img_array + nois
#
#     # 将像素值限制在0和255之间
#     noisy_image_array = np.clip(noisy_image_array, 0, 255)
#
#     # 转换为整数类型
#     noisy_image_array = noisy_image_array.astype(np.uint8)
#
#     # 将NumPy数组转换回PIL图像
#     noisy_image = Image.fromarray(noisy_image_array)
#
#     return noisy_image
#
# def process_images(src_folder, dest_folder):
#     # 遍历源文件夹及其子文件夹
#     for root, dirs, files in os.walk(src_folder):
#         for file in files:
#             file_path = os.path.join(root, file)
#
#             # 检查文件是否为图像文件
#             if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                 # 读取图像
#                 image = Image.open(file_path).convert("RGB")
#
#                 # 构建目标文件夹路径
#                 dest_subfolder = os.path.relpath(root, src_folder)
#                 dest_subfolder_path = os.path.join(dest_folder, dest_subfolder)
#
#                 # 如果目标文件夹不存在，则创建它
#                 if not os.path.exists(dest_subfolder_path):
#                     os.makedirs(dest_subfolder_path)
#                 # file = file.replace("x2","")
#                 # file = file.replace("x4","")
#                 # file = file.split(".")[0].zfill(4) + ".png"
#                 # 构建目标图像文件路径
#                 dest_file_path = os.path.join(dest_subfolder_path, file)
#
#                 image = add_gaussian_noise(image)
#
#                 # 保存图像到目标文件夹
#                 image.save(dest_file_path)
#
#                 # 关闭图像文件
#                 image.close()
#
# if __name__ == "__main__":
#     # 指定源文件夹和目标文件夹
#     source_folder = "C:\\Users\\22359\\Desktop\\image\\DRSRD1_2D\\shuffled2D800\\shuffled2D_test_LR_default_X2"
#     destination_folder = "C:\\Users\\22359\\Desktop\\image\\DRSRD1_2D\\shuffled2D800\\add_noise_shuffled2D_test_LR_default_X2"
#
#     # 处理图像
#     process_images(source_folder, destination_folder)

# file_path = "C:\\Users\\22359\\Desktop\\超分辨率重建\\Train_images\\train1\\0001.bmp"
# image = Image.open(file_path).convert("RGB")
# image.save("1.png")


# import numpy as np
# import matplotlib.pyplot as plt
# import pywt
# from skimage import io, color
#
# def get_high_frequency(image, wavelet='db1', level=1):
#     # Convert the image to grayscale
#     if image.shape[-1] == 3:
#         image = color.rgb2gray(image)
#
#     # Wavelet transform
#     coeffs = pywt.wavedec2(image, wavelet, level=level)
#
#     # Get detail coefficients, which contain high-frequency information
#     high_frequency = coeffs[1:]
#
#     return high_frequency
#
# # Read the image
# image_path = "C:\\Users\\22359\\Desktop\\1.jpg"
# original_image = io.imread(image_path)
#
# # Get high-frequency information
# high_frequency = get_high_frequency(original_image, wavelet='db1', level=1)
#
# # Display the results
# plt.figure(figsize=(12, 4))
#
# plt.subplot(1, 3, 1)
# plt.imshow(original_image, cmap='gray')
# plt.title('Original Image')
#
# for i in range(len(high_frequency)):
#     channel = high_frequency[i]
#     plt.subplot(1, 3, i + 2)
#     plt.imshow(np.abs(channel), cmap='gray')
#     plt.title(f'High Frequency Channel {i + 1}')
#
# plt.show()

# import cv2
# import numpy as np
# import os
# import matplotlib.pyplot as plt
# import numpy as np
# import matplotlib
# import pywt
#
# # 读取图像
# img = cv2.imread("C:\\Users\\22359\\Desktop\\0001.jpg", cv2.IMREAD_GRAYSCALE)
#
# # 小波逐次分解
# coeffs = pywt.wavedec2(img, 'haar', level=1)
# cA, (cH, cV, cD) = coeffs  # cA,cH,cV,cD分别代表从低频到高频的部分（即左上到右下）
# coeffs2 = pywt.wavedec2(img, 'haar', level=2)
# cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs2
#
# # 显示小波分解效果图
# plt.figure(figsize=(15, 5))
# plt.figure(1)
# plt.subplot(141)
# img = np.array(img, dtype=np.uint8)
# plt.imshow(img, 'gray')
# plt.title('yuantu')
# plt.subplot(142)
# cA = np.array(cA, dtype=np.uint8)
# plt.imshow(cA, 'gray')
# plt.title('1-xiaobo')
# plt.subplot(143)
# cA2 = np.array(cA2, dtype=np.uint8)
# plt.title('2-xiaobo')
# plt.imshow(cA2, 'gray')
# plt.subplot(144)
# cV = np.array(cV, dtype=np.uint8)
# plt.title('3-xiaobo')
# plt.imshow(cV, 'gray')
# plt.show()

# import torch
#
#
# x = torch.randn([2,1,4,4])
# print(x.shape)
# x = torch.cat([x]*3,dim=1)
# print(x.shape)


# from PIL import Image
#
# def check_channels_equal(image_path):
#     # 打开图像
#     img = Image.open(image_path)
#
#     # 将图像转换为RGB模式（如果不是RGB模式）
#     img = img.convert('RGB')
#
#     # 获取图像的像素数据
#     pixels = img.load()
#
#     # 遍历图像的每个像素
#     for x in range(img.width):
#         for y in range(img.height):
#             # 获取当前像素的RGB值
#             r, g, b = pixels[x, y]
#
#             # 检查三个通道的值是否相等
#             if r != g or g != b:
#                 return False
#
#     # 如果所有像素的三个通道的值都相等，则返回True
#     return True
#
# # 图片路径
# image_path = "C:\\Users\\22359\\Desktop\\10849x4_gray154SAFMN.png"
#
# # 检查三通道的值是否相等
# result = check_channels_equal(image_path)
#
# if result:
#     print("图像的三通道值相等。")
# else:
#     print("图像的三通道值不相等。")

# from PIL import Image
# import numpy as np
#
# def add_gaussian_noise(img):
#     # 将图像转换为NumPy数组
#     img_array = np.array(img)
#
#     # 生成高斯噪声
#     gauss = np.random.normal(size=img_array.shape)
#
#     # 将噪声添加到图像中
#     noisy_image_array = img_array + gauss
#
#     # 将像素值限制在0和255之间
#     noisy_image_array = np.clip(noisy_image_array, 0, 255)
#
#     # 转换为整数类型
#     noisy_image_array = noisy_image_array.astype(np.uint8)
#
#     # 将NumPy数组转换回PIL图像
#     noisy_image = Image.fromarray(noisy_image_array)
#
#     return noisy_image
#
# # 图像路径
# image_path = '1.png'
#
# # 打开图像
# original_image = Image.open(image_path)
#
# # 添加高斯噪声并保存结果
# noisy_image = add_gaussian_noise(original_image)
# noisy_image.save('3.png')

# import torch
# import torch.nn as nn
# from torchvision import transforms
# from PIL import Image
# import matplotlib.pyplot as plt
#
# # Read the image
# image_path = "1.png"
# image = Image.open(image_path)
#
# # Convert to PyTorch tensor
# tensor_image = transforms.ToTensor()(image).unsqueeze(0)  # Add a batch dimension
#
# # Define Sobel filter
# sobel_filter = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1, bias=False)
# sobel_filter.weight.data = torch.FloatTensor([[[1, 0, -1], [2, 0, -2], [1, 0, -1]]])
#
# # Apply Sobel filter for edge detection
# print(tensor_image.shape)
# # edges = sobel_filter(tensor_image)
#
# # # Display the original image and the edges
# # plt.figure(figsize=(10, 5))
# #
# # # Original image
# # plt.subplot(1, 2, 1)
# # plt.imshow(tensor_image.squeeze().permute(1, 2, 0), cmap='gray')
# # plt.title('Original Image')
# #
# # # Sobel edges
# # plt.subplot(1, 2, 2)
# # plt.imshow(edges.squeeze().detach().permute(1, 2, 0), cmap='gray')  # Detach to convert to numpy array
# # plt.title('Sobel Edges')
# #
# # plt.show()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class SobelConv2d(nn.Module):
#
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
#                  padding=0, dilation=1, groups=1, bias=True, requires_grad=True):
#         assert kernel_size % 2 == 1, 'SobelConv2d\'s kernel_size must be odd.'
#         assert out_channels % 4 == 0, 'SobelConv2d\'s out_channels must be a multiple of 4.'
#         assert out_channels % groups == 0, 'SobelConv2d\'s out_channels must be a multiple of groups.'
#
#         super(SobelConv2d, self).__init__()
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.kernel_size = kernel_size
#         self.stride = stride
#         self.padding = padding
#         self.dilation = dilation
#         self.groups = groups
#
#         # In non-trainable case, it turns into normal Sobel operator with fixed weight and no bias.
#         self.bias = bias if requires_grad else False
#
#         if self.bias:
#             self.bias = nn.Parameter(torch.zeros(size=(out_channels,), dtype=torch.float32), requires_grad=True)
#         else:
#             self.bias = None
#
#         self.sobel_weight = nn.Parameter(torch.zeros(
#             size=(out_channels, int(in_channels / groups), kernel_size, kernel_size)), requires_grad=False)
#
#         # Initialize the Sobel kernal
#         kernel_mid = kernel_size // 2
#         for idx in range(out_channels):
#             if idx % 4 == 0:
#                 self.sobel_weight[idx, :, 0, :] = -1
#                 self.sobel_weight[idx, :, 0, kernel_mid] = -2
#                 self.sobel_weight[idx, :, -1, :] = 1
#                 self.sobel_weight[idx, :, -1, kernel_mid] = 2
#             elif idx % 4 == 1:
#                 self.sobel_weight[idx, :, :, 0] = -1
#                 self.sobel_weight[idx, :, kernel_mid, 0] = -2
#                 self.sobel_weight[idx, :, :, -1] = 1
#                 self.sobel_weight[idx, :, kernel_mid, -1] = 2
#             elif idx % 4 == 2:
#                 self.sobel_weight[idx, :, 0, 0] = -2
#                 for i in range(0, kernel_mid + 1):
#                     self.sobel_weight[idx, :, kernel_mid - i, i] = -1
#                     self.sobel_weight[idx, :, kernel_size - 1 - i, kernel_mid + i] = 1
#                 self.sobel_weight[idx, :, -1, -1] = 2
#             else:
#                 self.sobel_weight[idx, :, -1, 0] = -2
#                 for i in range(0, kernel_mid + 1):
#                     self.sobel_weight[idx, :, kernel_mid + i, i] = -1
#                     self.sobel_weight[idx, :, i, kernel_mid + i] = 1
#                 self.sobel_weight[idx, :, 0, -1] = 2
#
#         # Define the trainable sobel factor
#         if requires_grad:
#             self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
#                                              requires_grad=True)
#         else:
#             self.sobel_factor = nn.Parameter(torch.ones(size=(out_channels, 1, 1, 1), dtype=torch.float32),
#                                              requires_grad=False)
#
#     def forward(self, x):
#         if torch.cuda.is_available():
#             self.sobel_factor = self.sobel_factor.cuda()
#             if isinstance(self.bias, nn.Parameter):
#                 self.bias = self.bias.cuda()
#
#         sobel_weight = self.sobel_weight * self.sobel_factor
#
#         if torch.cuda.is_available():
#             sobel_weight = sobel_weight.cuda()
#
#         out = F.conv2d(x, sobel_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
#
#         return out

#
# import cv2
# import numpy as np
# import pywt
# import torch
#
# # 加载图像
# image = cv2.imread('9601x4.png', cv2.IMREAD_GRAYSCALE)
# # image = cv2.imread("C:\\Users\\22359\\Desktop\\新建文件夹 (2)\\DeepRockSR-2D\\shuffled2D\\shuffled2D_valid_LR_default_X4\\9601x4.png", cv2.IMREAD_GRAYSCALE)
#
# # 小波变换
# wavelet = 'haar'
# mode = 'per'
# print(image)
# # image = torch.from_numpy(image1)
# # print(image.shape)
# # image = image.numpy()
# # print(image==image1)
# coeffs = pywt.dwt2(image, wavelet, mode)
#
# print(coeffs[0].shape)
# print(coeffs[1])
# # 设定阈值
# threshold = 2
#
# # 阈值处理
# coeffs = tuple(map(lambda x: pywt.threshold(x, threshold, mode='soft'), coeffs))
#
#
# # print(coeffs)
#
# # 小波逆变换
# denoised_image = pywt.idwt2(coeffs, wavelet, mode)
#
# # 显示原始图像和去噪后的图像
# cv2.imshow('Original Image', image)
# print(image.shape)
# cv2.imshow('Denoised Image', denoised_image.astype(np.uint8))
# print(denoised_image.shape)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# import torch
#
# def dwt_init(x):
#     # # Adjust dimensions if necessary
#     # input_shape = x.shape
#     # adjusted_shape = [s + 1 if s % 2 != 0 else s for s in input_shape[2:4]]
#     # x = torch.nn.functional.pad(x, (0, adjusted_shape[1] - input_shape[3], 0, adjusted_shape[0] - input_shape[2]))
#
#     x01 = x[:, :, 0::2, :] / 2
#     x02 = x[:, :, 1::2, :] / 2
#     x1 = x01[:, :, :, 0::2]
#     x2 = x02[:, :, :, 0::2]
#     x3 = x01[:, :, :, 1::2]
#     x4 = x02[:, :, :, 1::2]
#
#     x_LL = x1 + x2 + x3 + x4
#     x_HL = -x1 - x2 + x3 + x4
#     x_LH = -x1 + x2 - x3 + x4
#     x_HH = x1 - x2 - x3 + x4
#
#     return torch.cat((x_LL, x_HL, x_LH, x_HH), 1)
#
#
# def iwt_init(x):
#     r = 2
#     in_batch, in_channel, in_height, in_width = x.size()
#     # print([in_batch, in_channel, in_height, in_width])
#     out_batch, out_channel, out_height, out_width = in_batch, int(
#         in_channel / (r ** 2)), r * in_height, r * in_width
#     x1 = x[:, 0:out_channel, :, :] / 2
#     x2 = x[:, out_channel:out_channel * 2, :, :] / 2
#     x3 = x[:, out_channel * 2:out_channel * 3, :, :] / 2
#     x4 = x[:, out_channel * 3:out_channel * 4, :, :] / 2
#
#     h = torch.zeros([out_batch, out_channel, out_height, out_width]).float().cuda()
#
#     h[:, :, 0::2, 0::2] = x1 - x2 - x3 + x4
#     h[:, :, 1::2, 0::2] = x1 - x2 + x3 - x4
#     h[:, :, 0::2, 1::2] = x1 + x2 - x3 - x4
#     h[:, :, 1::2, 1::2] = x1 + x2 + x3 + x4
#
#     return h
#
# # 指定tensor的形状
# shape = (3, 4, 126, 126)  # 一个形状为 (3, 4, 5, 2) 的四维tensor
#
# # 使用torch.rand()创建随机tensor
# random_tensor = torch.rand(*shape)
#
# x = dwt_init(random_tensor)
#
# x1 = iwt_init(x)
#
# print(random_tensor)
# print(x1.cpu())
#
# # # 打印结果
# # print(random_tensor[:,0:1,:,:].shape)

######################################################################上采样#################################################
# import os
# import torch
# import torch.nn.functional as F
# from torchvision import transforms
# from PIL import Image
#
# class InterpolationModel(torch.nn.Module):
#     def __init__(self, scale_factor):
#         super(InterpolationModel, self).__init__()
#         self.scale_factor = scale_factor
#
#     def forward(self, x):
#         output = F.interpolate(x, scale_factor=self.scale_factor, mode="bicubic", align_corners=True)
#         return output
#
# def interpolate_image(input_path, output_path, scale_factor):
#     # Read the image
#     original_image = Image.open(input_path).convert('RGB')
#
#     # Transform the image to a PyTorch tensor
#     transform = transforms.ToTensor()
#     input_tensor = transform(original_image).unsqueeze(0)
#
#     # Create and run the interpolation model
#     interpolation_model = InterpolationModel(scale_factor)
#     output_tensor = interpolation_model(input_tensor)
#
#     # Convert the output tensor back to a PIL image
#     output_image = transforms.ToPILImage()(output_tensor.squeeze(0))
#
#     # Save the interpolated image
#     output_image.save(output_path)
#
# if __name__ == "__main__":
#     input_folder = "C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_LR_default_X4"  # Change this to the path of your input image folder
#     output_folder = "C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_LR_default_X4_upsample"  # Change this to the path where you want to save the output images
#     scale_factor = 4
#
#     # Create the output folder if it doesn't exist
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # Interpolate all images in the input folder
#     for filename in os.listdir(input_folder):
#         if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp','.BMP')):
#             input_path = os.path.join(input_folder, filename)
#             # output_path = os.path.join(output_folder, f"{filename.replace('x4','')}")
#             output_path = os.path.join(output_folder, f"{filename.replace('.BMP','.bmp')}")
#             output_path = output_path.replace('.bmp','.png')
#             interpolate_image(input_path, output_path, scale_factor)
######################################################################################################


###################################################计算psnr值###################################
# import os
# import cv2
# import numpy as np
# from skimage.metrics import structural_similarity as ssim
#
#
# def calculate_psnr(original_image, compressed_image,my_psnr,my_str):
#     crop_border = 4
#     img1 = cv2.imread(original_image)
#     img2 = cv2.imread(compressed_image)
#
#     img_ycrcb = cv2.cvtColor(img1, cv2.COLOR_BGR2YCrCb)
#     img1 = img_ycrcb[:,:,0]
#     print(img1.shape)
#     img_ycrcb2 = cv2.cvtColor(img2, cv2.COLOR_BGR2YCrCb)
#     img2 = img_ycrcb2[:,:,0]
#
#     # img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     # img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     # print(img1_gray[crop_border:-crop_border,crop_border:-crop_border].shape)
#     # mse = np.mean((img1_gray[crop_border:-crop_border] - img2_gray[crop_border:-crop_border]) ** 2)
#     ssim_index, _ = ssim(img1, img2, full=True)
#     mse = np.mean((img1[crop_border:-crop_border].astype(np.float64) - img2[crop_border:-crop_border].astype(np.float64)) ** 2)
#     # mse = np.mean((img1 - img2) ** 2)
#
#     if mse == 0:
#         return float('inf')
#
#     psnr = 20 * np.log10(255.0 / np.sqrt(mse))
#     if psnr < my_psnr:
#         my_psnr = psnr
#         my_str = compressed_image
#     return psnr,my_psnr,ssim_index
#
# def calculate_psnr_for_folder(original_folder, compressed_folder):
#     psnr_sum = 0.0
#     ssim_sum = 0.0
#     total_images = 0
#     my_psnr = 100000
#     my_str = ""
#
#     for filename in os.listdir(original_folder):
#         if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
#             original_path = os.path.join(original_folder, filename)
#             compressed_path = os.path.join(compressed_folder, filename.replace(".png","x4.png"))
#             # compressed_path = os.path.join(compressed_folder, filename.replace(".png","x4.png"))
#             if os.path.exists(compressed_path):
#                 psnr_value,my_psnr,ssim = calculate_psnr(original_path, compressed_path,my_psnr,my_str)
#                 print(f"PSNR for {filename}: {psnr_value} dB")
#
#                 psnr_sum += psnr_value
#                 ssim_sum += ssim
#                 total_images += 1
#     print(my_str)
#     if total_images == 0:
#         print("No valid images found for comparison.")
#     else:
#         average_psnr = psnr_sum / total_images
#         average_ssim = ssim_sum / total_images
#         print(f"\nAverage PSNR for {total_images} images: {average_psnr} dB")
#         print(f"\nAverage SSIM for {total_images} images: {average_ssim} dB")
#
# if __name__ == "__main__":
#     original_folder = "C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_HR"
#     compressed_folder = "C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_LR_default_X4_upsample"
#     # compressed_folder = "C:\\Users\\22359\\Desktop\\generate\\gray552SAFMN\\visualization\\Set5"
#     # compressed_folder = "C:\\Users\\22359\\Desktop\\generate\\2gray154SAFMN\\visualization\\Set5"
#
#
#     calculate_psnr_for_folder(original_folder, compressed_folder)




#########################################切分训练集和验证集###############################################################
# import os
# import shutil
# import random
#
# def split_images(input_folder, output_folder1, output_folder2, split_ratio=0.5,string = "test1"):
#     # 获取输入文件夹中的所有图像文件
#     image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
#
#     random.shuffle(image_files)
#     # 计算分割的数量
#     split_index = int(len(image_files) * split_ratio)
#
#     # 确保输出文件夹存在，否则创建它们
#     if not os.path.exists(output_folder1):
#     # os.makedirs(output_folder)
#         os.makedirs(output_folder1)
#     if not os.path.exists(output_folder2):
#         os.makedirs(output_folder2)
#
#     # 将图像按照分割比例分别复制到两个输出文件夹中
#     for i, image_file in enumerate(image_files):
#         source_path = os.path.join(input_folder, image_file)
#         if i < split_index:
#             destination_path = os.path.join(output_folder1, string+image_file)
#         else:
#             destination_path = os.path.join(output_folder2, string+image_file)
#         shutil.copy(source_path, destination_path)
#
# if __name__ == "__main__":
#     # 输入文件夹路径
#     input_folder_path = "C:\\Users\\22359\\Desktop\\超分辨率重建\\testset\\test5"
#     string = "5test"
#     # 输出文件夹路径
#     output_folder1_path = "C:\\Users\\22359\\Desktop\\超分辨率重建\\testset\\train"
#     output_folder2_path = "C:\\Users\\22359\\Desktop\\超分辨率重建\\testset\\valid"
#
#     # 分割比例（可调整）
#     split_ratio = 0.5
#
#     # 执行分割操作
#     split_images(input_folder_path, output_folder1_path, output_folder2_path, split_ratio,string)


######################################################合并数据集############################
# import os
# import shutil
#
# def merge_folders(input_folders, output_folder):
#     # 确保输出文件夹存在，否则创建它
#     os.makedirs(output_folder, exist_ok=True)
#
#     # 遍历输入文件夹列表
#     for input_folder in input_folders:
#         # 获取输入文件夹中的所有图像文件
#         image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]
#
#         # 将图像复制到输出文件夹中
#         for image_file in image_files:
#             source_path = os.path.join(input_folder, image_file)
#             destination_path = os.path.join(output_folder, image_file.lower())
#             shutil.copy(source_path, destination_path)
#
# if __name__ == "__main__":
#     # 输入文件夹列表
#     input_folders = ["C:\\Users\\22359\\Desktop\\超分辨率重建\\Train_images\\train1", "C:\\Users\\22359\\Desktop\\超分辨率重建\\Train_images\\train2", "C:\\Users\\22359\\Desktop\\超分辨率重建\\Train_images\\train3"
#                      ,"C:\\Users\\22359\\Desktop\\超分辨率重建\\Train_images\\train4","C:\\Users\\22359\\Desktop\\超分辨率重建\\Train_images\\train5"]
#
#     # 输出文件夹路径
#     output_folder_path = "C:\\Users\\22359\\Desktop\\超分辨率重建\\Train_images\\train"
#
#     # 执行合并操作
#     merge_folders(input_folders, output_folder_path)



##################################################转三通道###########################
# from PIL import Image
# import os
#
# def convert_to_rgb(input_folder, output_folder):
#     # 确保输出文件夹存在
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
#
#     # 遍历输入文件夹中的所有文件和子文件夹
#     for root, dirs, files in os.walk(input_folder):
#         for filename in files:
#             if filename.lower().endswith(('.bmp', '.BMP')):
#                 # 构建输入文件的完整路径
#                 input_path = os.path.join(root, filename)
#
#                 # 打开图像并将其转为RGB模式
#                 img = Image.open(input_path).convert('RGB')
#
#                 # 构建输出文件的相对路径，将文件扩展名改为.png
#                 output_relative_path = os.path.relpath(input_path, input_folder)
#                 output_path = os.path.join(output_folder, os.path.splitext(output_relative_path)[0] + '.png')
#
#                 # 确保输出文件夹的子文件夹结构存在
#                 output_subfolder = os.path.dirname(output_path)
#                 if not os.path.exists(output_subfolder):
#                     os.makedirs(output_subfolder)
#
#                 # 保存图像
#                 img.save(output_path)
#
# if __name__ == "__main__":
#     # 指定输入和输出文件夹
#     input_folder = "C:\\Users\\22359\\Desktop\\add1\\valid"
#     output_folder = "C:\\Users\\22359\\Desktop\\add1\\valid1"
#
#     # 执行转换
#     convert_to_rgb(input_folder, output_folder)


# from PIL import Image
# import os
#
# def convert_images(input_folder, output_folder):
#     # 遍历输入文件夹及其子文件夹中的所有文件
#     for root, dirs, files in os.walk(input_folder):
#         for file in files:
#             # 检查文件是否是图像文件
#             if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
#                 # 构建输入文件的完整路径
#                 input_path = os.path.join(root, file)
#
#                 # 打开图像文件
#                 image = Image.open(input_path)
#
#                 # 转换为单通道图像
#                 image = image.convert('L')
#
#                 # 构建输出文件的完整路径，将后缀改为.bmp
#                 output_path = os.path.join(output_folder, os.path.relpath(input_path, input_folder))
#                 output_path = os.path.splitext(output_path)[0] + '.bmp'
#
#                 # 确保输出文件夹存在
#                 os.makedirs(os.path.dirname(output_path), exist_ok=True)
#
#                 # 保存单通道图像
#                 image.save(output_path)
#
#                 print(f"Converted {input_path} to {output_path}")
#
# if __name__ == "__main__":
#     input_folder = "C:\\Users\\22359\\Desktop\\add1"  # 替换为你的输入文件夹路径
#     output_folder = "C:\\Users\\22359\\Desktop\\add"  # 替换为你的输出文件夹路径
#
#     convert_images(input_folder, output_folder)

#
# from skimage.metrics import structural_similarity as ssim
# import cv2
#
# def are_images_equal(image_path1, image_path2):
#     # Read images
#     img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
#     img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)
#
#     # Check if images are loaded successfully
#     if img1 is None or img2 is None:
#         print("Error: Unable to load images.")
#         return
#
#     # Compute SSIM
#     ssim_index, _ = ssim(img1, img2, full=True)
#     print(ssim_index)
#     # SSIM index ranges from -1 to 1, where 1 indicates identical images
#     if ssim_index > 0.95:
#         print("Images are similar.")
#     else:
#         print("Images are not similar.")
#
# if __name__ == "__main__":
#     image_path1 = "C:\\Users\\22359\\Desktop\\add\\three_train\\0001.bmp"  # Replace with the path of the first image
#     image_path2 = "C:\\Users\\22359\\Desktop\\SR\\Train_images\\train\\0001.bmp"  # Replace with the path of the second image
#
#     are_images_equal(image_path1, image_path2)

############################################################使用Sobel算子对图像进行分割######################################
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # 读取图像
# image_path = "C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_HR\\10803.png"  # 替换为你的图像路径
# # image_path = "C:\\Users\\22359\\Desktop\\valid\\results\\0055B4SAFMN\\visualization\\Set5\\10803x4_0055B4SAFMN.png"  # 替换为你的图像路径
# original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
# # 使用Sobel算子进行边缘检测
# sobel_x = cv2.Sobel(original_image, cv2.CV_64F, 1, 0, ksize=3)
# sobel_y = cv2.Sobel(original_image, cv2.CV_64F, 0, 1, ksize=3)
#
# # 计算梯度幅值和方向
# gradient_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
# # 保存梯度幅值图
# output_path = "carbon.png"  # 替换为你希望保存的路径
# cv2.imwrite(output_path, gradient_magnitude)
# # print("Gradient Magnitude image saved at:", output_path)
#
# gradient_direction = np.arctan2(sobel_y, sobel_x)
#
# # 显示结果
# plt.figure(figsize=(12, 6))
#
# plt.subplot(2, 3, 1), plt.imshow(original_image, cmap='gray')
# plt.title('Original Image'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(2, 3, 2), plt.imshow(sobel_x, cmap='gray')
# plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(2, 3, 3), plt.imshow(sobel_y, cmap='gray')
# plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(2, 3, 4), plt.imshow(gradient_magnitude, cmap='gray')
# plt.title('Gradient Magnitude'), plt.xticks([]), plt.yticks([])
#
# plt.subplot(2, 3, 5), plt.imshow(gradient_direction, cmap='gray')
# plt.title('Gradient Direction'), plt.xticks([]), plt.yticks([])
#
# plt.show()


# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class CustomConvolution(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, learnable_fraction=0.5):
#         super(CustomConvolution, self).__init__()
#
#         self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
#
#         # Determine the number of parameters that should be learned
#         total_params = self.conv.weight.numel()
#         learnable_params = int(total_params * learnable_fraction)
#
#         # Create learnable parameters and non-learnable parameters
#         self.learnable_weights = nn.Parameter(torch.randn(learnable_params))
#         self.non_learnable_weights = torch.randn(total_params - learnable_params)
#
#         # Concatenate learnable and non-learnable parameters
#         self.weights = torch.cat([self.learnable_weights, self.non_learnable_weights])
#
#         # Assign the concatenated parameters to the convolution layer
#         self.conv.weight = nn.Parameter(self.weights.view_as(self.conv.weight))
#
#     def forward(self, x):
#         return self.conv(x)
#
#
# # Example usage
# in_channels = 3
# out_channels = 64
# kernel_size = 3
#
# custom_conv = CustomConvolution(in_channels, out_channels, kernel_size)
#
# # Print the model's parameters
# for name, param in custom_conv.named_parameters():
#     print(name, param.size(), param.requires_grad)
#
# F.conv2d()


# import torch.nn as nn
# import torch
#
#
# class MultiheadAttention(nn.Module):
#     # n_heads：多头注意力的数量
#     # hid_dim：每个词输出的向量维度
#     def __init__(self, hid_dim, n_heads, dropout):
#         super(MultiheadAttention, self).__init__()
#         self.hid_dim = hid_dim
#         self.n_heads = n_heads
#
#         # 强制 hid_dim 必须整除 h
#         assert hid_dim % n_heads == 0
#         # 定义 W_q 矩阵
#         self.w_q = nn.Linear(hid_dim, hid_dim)
#         # 定义 W_k 矩阵
#         self.w_k = nn.Linear(hid_dim, hid_dim)
#         # 定义 W_v 矩阵
#         self.w_v = nn.Linear(hid_dim, hid_dim)
#         self.fc = nn.Linear(hid_dim, hid_dim)
#         self.do = nn.Dropout(dropout)
#         # 缩放
#         self.scale = torch.sqrt(torch.FloatTensor([hid_dim // n_heads]))
#
#     def forward(self, query, key, value, mask=None):
#         # K: [64,12,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
#         # V: [64,12,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
#         # Q: [64,12,300], batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
#         bsz = query.shape[0]
#         Q = self.w_q(query)
#         K = self.w_k(key)
#         V = self.w_v(value)
#         # 这里把 K Q V 矩阵拆分为多组注意力，变成了一个 4 维的矩阵
#         # 最后一维就是是用 self.hid_dim // self.n_heads 来得到的，表示每组注意力的向量长度, 每个 head 的向量长度是：300/6=50
#         # 64 表示 batch size，6 表示有 6组注意力，10 表示有 10 词，50 表示每组注意力的词的向量长度
#         # K: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
#         # V: [64,10,300] 拆分多组注意力 -> [64,10,6,50] 转置得到 -> [64,6,10,50]
#         # Q: [64,12,300] 拆分多组注意力 -> [64,12,6,50] 转置得到 -> [64,6,12,50]
#         # 转置是为了把注意力的数量 6 放到前面，把 10 和 50 放到后面，方便下面计算
#         Q = Q.view(bsz, -1, self.n_heads, self.hid_dim //
#                    self.n_heads).permute(0, 2, 1, 3)
#         K = K.view(bsz, -1, self.n_heads, self.hid_dim //
#                    self.n_heads).permute(0, 2, 1, 3)
#         V = V.view(bsz, -1, self.n_heads, self.hid_dim //
#                    self.n_heads).permute(0, 2, 1, 3)
#
#         # 第 1 步：Q 乘以 K的转置，除以scale
#         # [64,6,12,50] * [64,6,50,10] = [64,6,12,10]
#         # attention：[64,6,12,10]
#         attention = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
#
#         # 把 mask 不为空，那么就把 mask 为 0 的位置的 attention 分数设置为 -1e10
#         if mask is not None:
#             attention = attention.masked_fill(mask == 0, -1e10)
#
#         # 第 2 步：计算上一步结果的 softmax，再经过 dropout，得到 attention。
#         # 注意，这里是对最后一维做 softmax，也就是在输入序列的维度做 softmax
#         # attention: [64,6,12,10]
#         attention = self.do(torch.softmax(attention, dim=-1))
#
#         # 第三步，attention结果与V相乘，得到多头注意力的结果
#         # [64,6,12,10] * [64,6,10,50] = [64,6,12,50]
#         # x: [64,6,12,50]
#         x = torch.matmul(attention, V)
#
#         # 因为 query 有 12 个词，所以把 12 放到前面，把 5 和 60 放到后面，方便下面拼接多组的结果
#         # x: [64,6,12,50] 转置-> [64,12,6,50]
#         x = x.permute(0, 2, 1, 3).contiguous()
#         # 这里的矩阵转换就是：把多组注意力的结果拼接起来
#         # 最终结果就是 [64,12,300]
#         # x: [64,12,6,50] -> [64,12,300]
#         x = x.view(bsz, -1, self.n_heads * (self.hid_dim // self.n_heads))
#         x = self.fc(x)
#         return x
#
#
# # batch_size 为 64，有 12 个词，每个词的 Query 向量是 300 维
# query = torch.rand(64, 12, 300)
# # batch_size 为 64，有 12 个词，每个词的 Key 向量是 300 维
# key = torch.rand(64, 12, 300)
# # batch_size 为 64，有 12 个词，每个词的 Value 向量是 300 维
# value = torch.rand(64, 12, 300)
# attention = MultiheadAttention(hid_dim=300, n_heads=6, dropout=0.1)
# output = attention(query, key, value)
# ## output: torch.Size([64, 12, 300])
# print(output.shape)

#
#
# import torch
# import torch.nn as nn
#
# # 定义一个简单的模型，使用9x9的核的转置卷积进行上采样
# class UpsampleModel(nn.Module):
#     def __init__(self, upsample_factor):
#         super(UpsampleModel, self).__init__()
#         self.transposed_conv = nn.ConvTranspose2d(in_channels=3, out_channels=3, kernel_size=9, stride=upsample_factor, padding=4)
#
#     def forward(self, x):
#         x = self.transposed_conv(x)
#         return x
#
# # 创建一个上采样倍数为4的模型实例
# upsample_factor = 4
# model = UpsampleModel(upsample_factor)
#
# # 输入一个4x4的随机张量
# input_tensor = torch.rand(1, 3, 4, 4)
#
# # 使用模型进行上采样
# output = model(input_tensor)
#
# # 打印输入和输出的形状
# print("Input shape:", input_tensor.shape)
# print("Output shape:", output.shape)

##########################################################灰度直方图#########################################
# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import find_peaks
#
# # 读取图像
# # image = cv2.imread('0901.png', cv2.IMREAD_GRAYSCALE)
# # image = cv2.imread("C:\\Users\\22359\\Desktop\\valid\\SR_latest\\SuperRestoration-master\\SRCNN\\test_res\\test_x4_Set5\\x4\\10803.png", cv2.IMREAD_GRAYSCALE)
# # image = cv2.imread("C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_HR\\10803.png", cv2.IMREAD_GRAYSCALE)
# # image = cv2.imread("C:\\Users\\22359\\Desktop\\valid\\SR_latest\\SuperRestoration-master\\FSRCNN\\test_res\\x4\\10803_FSRCNN_x4.png", cv2.IMREAD_GRAYSCALE)
# # image = cv2.imread("C:\\Users\\22359\\Desktop\\valid\\results\\0055B4SAFMN\\visualization\\Set5\\10803x4_0055B4SAFMN.png", cv2.IMREAD_GRAYSCALE)
# image = cv2.imread("C:\\Users\\22359\\Desktop\\valid\\results\\155B4SAFMN\\visualization\\Set5\\10803x4_155B4SAFMN.png", cv2.IMREAD_GRAYSCALE)
#
# # 计算灰度直方图
# hist, bins = np.histogram(image.flatten(), 256, [0, 256])
# print(hist)
# print(bins)
#
# # 找到直方图中的波峰值
# peaks, _ = find_peaks(hist)
#
# # 绘制图像和直方图
# plt.subplot(1, 2, 1)
# plt.imshow(image, cmap='gray')
# plt.title('Original Image')
#
# plt.subplot(1, 2, 2)
# plt.plot(hist)
# for peak in peaks:
#     print("peaks:",peak)
#     print("hist:",hist[peak])
#     print("")
# # plt.plot(peaks, hist[peaks], "x")
# plt.title('Histogram with Peaks')
# print(sum(hist[0:77])/sum(hist))
#
# plt.show()

import cv2
import numpy as np

import os

def get_image_names(folder_path):
    image_names = []
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif', '.bmp')):  # 只处理图片文件
            name_without_extension = os.path.splitext(filename)[0]  # 获取文件名并去除后缀
            image_names.append(name_without_extension)
    return image_names

# 指定文件夹路径
folder_path = "C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_HR"
nums = get_image_names(folder_path)



def calculate_threshold(image_path):
    # 读取图像
    image = cv2.imread(image_path, 0)  # 第二个参数0表示以灰度模式读取图像

    # 设定初始阈值
    threshold = 255
    previous_threshold = 0

    # 迭代直到阈值不再变化
    while abs(previous_threshold - threshold) > 0.01 :
        previous_threshold = threshold

        # 根据当前阈值进行二值化处理
        _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)

        # 计算新的阈值
        threshold = (cv2.mean(image)[0] + cv2.mean(image, mask=~binary_image)[0]) / 2
    return threshold

# nums = ["10803","10820","10826","10804","10821","10838","10807","10831","11984"]
image_paths = ["C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_HR\\","C:\\Users\\22359\\Desktop\\valid\\SR_latest\\SuperRestoration-master\\SRCNN\\test_res\\test_x4_Set5\\x4\\","C:\\Users\\22359\\Desktop\\valid\\SR_latest\\SuperRestoration-master\\FSRCNN\\test_res\\x4\\","C:\\Users\\22359\\Desktop\\valid\\results\\0055B4SAFMN\\visualization\\Set5\\","C:\\Users\\22359\\Desktop\\valid\\results\\155B4SAFMN\\visualization\\Set5\\"]
image_names =[".png",".png","_FSRCNN_x4.png","x4_0055B4SAFMN.png","x4_155B4SAFMN.png"]
voidages = np.zeros(len(image_paths))
# image_path = "C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_HR\\10803.png"
# image_path = "C:\\Users\\22359\\Desktop\\valid\\SR_latest\\SuperRestoration-master\\SRCNN\\test_res\\test_x4_Set5\\x4\\10803.png"
# image_path = "C:\\Users\\22359\\Desktop\\valid\\SR_latest\\SuperRestoration-master\\FSRCNN\\test_res\\x4\\10803_FSRCNN_x4.png"
# image_path = "C:\\Users\\22359\\Desktop\\valid\\results\\0055B4SAFMN\\visualization\\Set5\\10803x4_0055B4SAFMN.png"
# image_path = "C:\\Users\\22359\\Desktop\\valid\\results\\155B4SAFMN\\visualization\\Set5\\10803x4_155B4SAFMN.png"
original_voidage = 1
for i in range(len(nums)):
    for j in range(len(image_paths)):
        image_path = image_paths[j] + nums[i] + image_names[j]
        image = cv2.imread(image_path, 0)  # 第二个参数0表示以灰度模式读取图像
        hist, bins = np.histogram(image.flatten(), 256, [0, 255])
        # # print(hist)
        # print(len(bins))
        if j == 0:
            # 读取图像
            threshold = calculate_threshold(image_path)
            # print(int(threshold+1))
            rate1 = sum(hist[0:int(threshold+1)]) / sum(hist)
            rate2 = sum(hist[int(threshold+1):]) / sum(hist)
            original_voidage = rate1 if rate1 < rate2 else rate2
        else:
            rate1 = sum(hist[0:int(threshold+1)]) / sum(hist)
            rate2 = sum(hist[int(threshold+1):]) / sum(hist)
            voidage = rate1 if rate1 < rate2 else rate2
            voidages[j] += abs(voidage - original_voidage) / original_voidage
print(voidages/len(nums))

# # 设定初始阈值
# threshold = 127
# previous_threshold = 0
#
# # 迭代直到阈值不再变化
# while previous_threshold != threshold:
#     previous_threshold = threshold
#
#     # 根据当前阈值进行二值化处理
#     _, binary_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY)
#
#     # 计算新的阈值
#     threshold = int((cv2.mean(image)[0] + cv2.mean(image, mask=~binary_image)[0]) / 2)
#
# # 显示分割后的图像
# cv2.imshow('Segmented Image', binary_image)
# cv2.waitKey(0)
# hist, bins = np.histogram(image.flatten(), 256, [0, 256])
# print(threshold)
# # print(sum(hist[0:int(threshold)])/sum(hist))
# voidage = sum(hist[0:97])/sum(hist)
# print(voidage)
# print(abs(voidage-0.174856)/0.174856)
# cv2.destroyAllWindows()

######################################################获取图像的鸟瞰图############################################
# import cv2
# import numpy as np
#
# def get_birdseye_view(image_path, output_path, height=500, width=500):
#     # 读取图像
#     image = cv2.imread(image_path)
#
#     # 定义图像的四个角点
#     pts1 = np.float32([[200, 50], [300, 50], [200, 150], [300, 150]])#carbon
#     # pts1 = np.float32([[350, 30], [450, 30], [350, 120], [450, 120]])#coal
#     # pts1 = np.float32([[360, 20], [460, 20], [360, 120], [460, 120]])#standstone
#
#
#     # 定义鸟瞰图的四个角点
#     pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
#
#     # 计算变换矩阵
#     matrix = cv2.getPerspectiveTransform(pts1, pts2)
#
#     # 进行透视变换
#     result = cv2.warpPerspective(image, matrix, (width, height))
#
#     # 保存鸟瞰图
#     cv2.imwrite(output_path, result)
#
# if __name__ == "__main__":
#     # input_image_path = "C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_HR\\10803.png"
#     # input_image_path = "C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_LR_default_X4_upsample\\10803x4.png"
#     # input_image_path = "C:\\Users\\22359\\Desktop\\valid\\results\\0055B4SAFMN\\visualization\\Set5\\10803x4_0055B4SAFMN.png"
#     # input_image_path = "C:\\Users\\22359\\Desktop\\valid\\results\\155B4SAFMN\\visualization\\Set5\\10803x4_155B4SAFMN.png"
#     # input_image_path = "C:\\Users\\22359\\Desktop\\valid\\SR_latest\\SuperRestoration-master\\FSRCNN\\test_res\\x4\\10803_FSRCNN_x4.png"
#     input_image_path = "C:\\Users\\22359\\Desktop\\valid\\SR_latest\\SuperRestoration-master\\SRCNN\\test_res\\test_x4_Set5\\x4\\10803.png"
#     output_image_path = "carbon_SRCNN_birdseye.jpg"
#
#     get_birdseye_view(input_image_path, output_image_path)

##################################框出选中区域################################
# import cv2
# import numpy as np
#
# # 读取图像
# image = cv2.imread("C:\\Users\\22359\\Desktop\\image\\DeepRockSR-2D\\shuffled2D\\shuffled2D_test_HR\\10803.png")
#
# # 定义矩形框的坐标和大小 (x, y, width, height)
# rectangle_coordinates = (200, 50, 100, 100)
#
# # 在图像上绘制矩形框
# cv2.rectangle(image, (rectangle_coordinates[0], rectangle_coordinates[1]),
#               (rectangle_coordinates[0] + rectangle_coordinates[2], rectangle_coordinates[1] + rectangle_coordinates[3]),
#               (0, 0, 255), 2)  # 最后一个参数是框的颜色和线宽
#
# # 显示带有矩形框的图像
# cv2.imshow('Image with Rectangle', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

























