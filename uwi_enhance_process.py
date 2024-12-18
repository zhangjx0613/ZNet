import numpy as np
import cv2
import os

root = 'data/SUIM/' # change to your data folder path
data_f = ['train_image/images/', 'test_image/images/']
mask_f = ['train_image/masks/', 'test_image/masks/']
enhance_f = ['train_image/red/', 'test_image/red/']
set_size = [1525, 110]
save_name = ['train', 'test']

height = 192 # 384
width = 256 # 512

for j in range(2):

	print('processing ' + data_f[j] + '......')
	count = 0
	length = set_size[j]
	imgs = np.uint8(np.zeros([length, height, width, 3]))
	enhances = np.uint8(np.zeros([length, height, width, 3]))
	depths = np.uint8(np.zeros([length, height, width]))
	masks = np.uint8(np.zeros([length, height, width]))
	mask_RGBs = np.uint8(np.zeros([length, height, width, 3]))
	filenames = np.chararray(length, itemsize=100, unicode=True)

	path = root + data_f[j]
	enhance_p = root + enhance_f[j]
	mask_p = root + mask_f[j]

	for i in os.listdir(path):

		# if len(i.split('_'))==2:
		img = cv2.imread(path+i)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = cv2.resize(img, (width, height))

		e_path = enhance_p + i.replace('.jpg', '_red.jpg')
		enhance = cv2.imread(e_path)
		enhance = cv2.cvtColor(enhance, cv2.COLOR_BGR2RGB)
		enhance = cv2.resize(enhance, (width, height))

		d_path = enhance_p + i.replace('.jpg', '_red_trans_out.jpg')
		depth = cv2.imread(d_path, 0)
		depth = cv2.resize(depth, (width, height))

		m_path = mask_p + i.replace('.jpg', '.bmp')
		# mask = cv2.imread(m_path, 0)
		mask_RGB = cv2.imread(m_path)
		mask_RGB = cv2.cvtColor(mask_RGB, cv2.COLOR_BGR2RGB)
		mask_RGB = cv2.resize(mask_RGB, (width, height), interpolation=cv2.INTER_NEAREST)

		# 定义颜色调色板，格式为(R, G, B)
		color_palette = [
			(0, 0, 0),  # BW
			(0, 0, 255),  # HD
			(0, 255, 255),  # WR
			(255, 0, 0),  # RO
			(255, 0, 255),  # RI
			(255, 255, 0),  # FV
			(0, 255, 0),  # PF
			(255, 255, 255),  # SR
		]
		# 创建一个集合来存储不同的颜色索引
		color_indices = {}
		# 将颜色调色板的RGB值映射到单通道的颜色索引
		for k, color in enumerate(color_palette):
			color_indices[color] = k
		# 创建一个与图像数组形状相同的单通道数组
		palette_l = np.zeros_like(mask_RGB[:, :, 0], dtype=np.uint8)
		# 将每个像素的RGB值映射到颜色索引并存储到单通道数组中
		for color, index in color_indices.items():
			# if color == (0, 0, 0) or color == (255, 255, 255) or color == (0, 255, 0):
			# 	mask = np.all(np.isclose(mask_RGB[:, :, :3], np.array(color), atol=5), axis=-1)
			# 	palette_l[mask] = 0
			# else:
			mask = np.all(np.isclose(mask_RGB[:, :, :3], np.array(color), atol=5), axis=-1)
			palette_l[mask] = index

		imgs[count] = img
		enhances[count] = enhance
		depths[count] = depth
		masks[count] = palette_l
		# filenames[count] = i
		if save_name[j]=='test':
			mask_RGBs[count] = mask_RGB
			filenames[count] = i

		count +=1
		print(count)


	np.save('{}/uwi_data_{}.npy'.format(root+"uwi_enhance_name/", save_name[j]), imgs)
	np.save('{}/uwi_enhance_{}.npy'.format(root+"uwi_enhance_name/", save_name[j]), enhances)
	np.save('{}/uwi_depth_{}.npy'.format(root+"uwi_enhance_name/", save_name[j]), depths)
	np.save('{}/uwi_mask_{}.npy'.format(root+"uwi_enhance_name/", save_name[j]), masks)
	if save_name[j] == 'test':
		np.save('{}/uwi_maskrgb_{}.npy'.format(root + "uwi_enhance_name/", save_name[j]), mask_RGBs)
		np.save('{}/uwi_testname_{}.npy'.format(root + "uwi_enhance_name/", save_name[j]), filenames)