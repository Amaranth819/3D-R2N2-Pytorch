from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import torch
import os
import numpy as np
import voxel
import random

def get_all_files_with_postfix(top_dir, postfix):
	files = []
	for r, d, f in os.walk(top_dir):
		dire = []
		for each_file in f:
			if postfix in each_file:
				dire.append(os.path.join(r, each_file))
		if len(dire) > 0:
			files.append(dire)
	return files

def create_data(img_paths, model_paths):
	return zip(img_paths, model_paths)

# Transform and dataset
img_transforms = transforms.Compose([
	transforms.RandomCrop((127, 127)),
	transforms.ToTensor()
])

# model_transform = transforms.Compose([
# 	transforms.ToTensor()
# ])
model_transform = None

class ReadShapeNet(Dataset):
	def __init__(self, img_top_dir, img_postfix, model_top_dir, model_postfix, seq_len, img_transform = None, model_transform = None):
		super(ReadShapeNet, self).__init__()
		self.seq_len = seq_len
		self.img_transform = img_transform
		self.model_transform = model_transform

		# img and model
		self.data_pairs = create_data(
			get_all_files_with_postfix(img_top_dir, img_postfix),
			get_all_files_with_postfix(model_top_dir, model_postfix)
		)

	def __getitem__(self, index):
		pair = self.data_pairs[index]

		indices = random.sample(range(0, 24), self.seq_len)
		img_paths = [pair[0][idx] for idx in indices]

		if self.img_transform is not None:
			img_data = np.stack([self.img_transform(Image.open(ip).convert('RGB')) for ip in img_paths], axis = 0)
		else:
			img_data = np.stack([Image.open(ip).convert('RGB') for ip in img_paths], axis = 0)

		model_data = voxel.read_voxel_data(pair[1][0])
		if self.model_transform is not None:
			model_data = self.model_transform(model_data)
		voxel_data = np.zeros((2, 32, 32, 32))
		voxel_data[0, ...] = model_data < 1
		voxel_data[1, ...] = model_data

		return img_data, voxel_data

	def __len__(self):
		return len(self.data_pairs)

# dataset = ReadShapeNet('../../3dr2n2/code/dataset/test/imgs/02691156/', '.png', \
# 	'../../3dr2n2/code/dataset/test/models/02691156', '.binvox', 5, \
# 	img_transform = img_transform, model_transform = None)
# loader = DataLoader(dataset, batch_size = 8, shuffle = True)
# i = 0
# for data in loader:
# 	if i > 5:
# 		break
# 	print(data[0].size()) # (8, 5, 127, 127, 3)
# 	print(data[1].size()) # (8, 32, 32, 32)
# 	i += 1