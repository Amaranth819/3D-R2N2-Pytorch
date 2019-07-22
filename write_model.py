import sys
sys.path.append('./lib')
sys.path.append('./network')

import torch
import argparse
import os
from PIL import Image
from res_net_gru import Res_Gru_Net
from voxel import write_binvox_file
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable
from data import ReadShapeNet, img_transforms, model_transform

def write_model(config):
	pretrained_model_path = config.pmp
	dataset_img_path = config.dip
	img_postfix = config.ip
	dataset_model_path = config.dmp
	model_postfix = config.mp
	threshold = config.th
	result_root_path = config.rrp
	seq_len = config.sl
	batch_size = config.bs
	num_batches = config.num

	# dataset
	if os.path.exists(dataset_img_path) is False and os.path.exists(dataset_model_path) is False:
		raise ValueError('Unknown dataset path.')
	eval_dataset = ReadShapeNet(dataset_img_path, img_postfix, dataset_model_path, model_postfix, seq_len, img_transform = img_transforms, model_transform = model_transform)
	eval_dataset_loader = DataLoader(eval_dataset, batch_size = batch_size, shuffle = True)

	# Save objs
	if os.path.exists(result_root_path) is False:
		os.makedirs(result_root_path)

	# model
	is_cuda = torch.cuda.is_available()
	model = Res_Gru_Net(seq_len)
	if is_cuda is True:
		model = model.cuda()
	if os.path.exists(pretrained_model_path) is False:
		raise ValueError('Unknown model path.')
	checkpoint = torch.load(pretrained_model_path)
	model.load_state_dict(checkpoint['state_dict'])
	model.eval()

	# Start writing models
	idx = 0
	total_num = num_batches * batch_size
	print("Start!")
	for eval_data in eval_dataset_loader:
		if idx >= total_num:
			break
		eval_img_data = Variable(eval_data[0].cuda()) if is_cuda is True else Variable(eval_data[0])
		# eval_model_data = Variable(eval_model_data[1].cuda()) if is_cuda is True else Variable(eval_model_data[1])

		# Write models
		pred = model(eval_img_data).cpu().detach().numpy()
		pred = pred[:, 1, ...] >= threshold
		for b in range(batch_size):
			x = pred[b]
			file_name = result_root_path + '%d%s' % (idx, model_postfix)
			write_binvox_file(x, file_name)
			idx += 1
	print("End!")

def main():
	parser = argparse.ArgumentParser()

	# Parameters
	parser.add_argument('--pmp', type = str, help = 'Pretrained model path.', default = './model/model.pkl')
	parser.add_argument('--dip', type = str, help = 'Image path.', default = '/home/xulin/Documents/dataset/shapenet/dataset/test/imgs/03001627/')
	parser.add_argument('--ip', type = str, help = 'Image postfix.', default = '.png')
	parser.add_argument('--dmp', type = str, help = 'Model path.', default = '/home/xulin/Documents/dataset/shapenet/dataset/test/models/03001627/')
	parser.add_argument('--mp', type = str, help = 'Model postfix.', default = '.binvox')
	parser.add_argument('--th', type = int, help = 'Threshold.', default = 0.4)
	parser.add_argument('--rrp', type = str, help = 'Root path for storing models.', default = './objs/03001627/')
	parser.add_argument('--sl', type = int, help = 'Sequence length.', default = 5)
	parser.add_argument('--bs', type = int, help = 'Batch size.', default = 16)
	parser.add_argument('--num', type = int, help = 'The number of batches for evaluation.', default = 5)

	config = parser.parse_args()
	write_model(config)


if __name__ == '__main__':
	main()