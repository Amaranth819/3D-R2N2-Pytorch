import torch
import torch.nn as nn
import argparse
import os
import sys
import numpy as np
sys.path.append('./lib')
sys.path.append('./network')

# from net import Res_GRU_Net, weight_init
from res_net_gru import Res_Gru_Net, weight_init, calculate_iou
from layer import SoftmaxLoss3D
from data import ReadShapeNet, img_transforms, model_transform
from torch.utils.data import DataLoader
from torch.autograd.variable import Variable

# def calculate_iou(pred, model_data, threshold, is_cuda):
# 	occupy = pred[:, 1, ...] >= threshold
# 	model_data = model_data[:, 1, ...]
# 	occupy = occupy.type(torch.cuda.ByteTensor) if is_cuda is True else occupy.type(torch.ByteTensor)
# 	model_data = model_data.type(torch.cuda.ByteTensor) if is_cuda is True else model_data.type(torch.ByteTensor)
# 	logic_and = torch.sum(occupy & model_data)
# 	logic_or = torch.sum(occupy | model_data)
# 	iou = logic_and.float() / logic_or.float()
# 	return iou

def start(config):
	# Configuration
	bs = config.bs
	seq_len = config.sl
	epoch = config.e
	init_lr = config.lr
	lr_decay = config.lrd
	lr_decay_step = config.lrds
	eval_step = config.es
	threshold = config.t
	save_model_step = config.ss
	model_path = config.mp
	model_name = config.mn
	log_path = config.lp
	log_name = config.ln
	train_img_root = config.train_img_root
	train_model_root = config.train_model_root
	test_img_root = config.test_img_root
	test_model_root = config.test_model_root
	img_postfix = config.img_pf
	model_postfix = config.model_pf

	# Create the dataset
	if os.path.exists(train_img_root) is False and os.path.exists(train_model_root) is False:
		raise ValueError('Unknown training dataset!')
	train_dataset = ReadShapeNet(train_img_root, img_postfix, train_model_root, model_postfix, seq_len, img_transform = img_transforms, model_transform = model_transform)
	print("%d training samples." % len(train_dataset))
	train_dataset_loader = DataLoader(train_dataset, batch_size = bs, shuffle = True)

	if os.path.exists(test_img_root) is False and os.path.exists(test_model_root) is False:
		raise ValueError('Unknown testing dataset!')
	test_dataset = ReadShapeNet(test_img_root, img_postfix, test_model_root, model_postfix, seq_len, img_transform = img_transforms, model_transform = model_transform)
	print("%d testing samples." % len(test_dataset))
	test_dataset_loader = DataLoader(test_dataset, batch_size = bs, shuffle = False)

	# Build the network
	is_cuda = torch.cuda.is_available()
	model = Res_Gru_Net(seq_len)
	model = model.float()
	if is_cuda is True:
		model = model.cuda()
	loss_func = SoftmaxLoss3D()

	full_model_path = model_path + model_name + '.pkl'
	optimizer = torch.optim.Adam(model.parameters(), lr = init_lr, weight_decay = 1e-4)
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = lr_decay_step, gamma = lr_decay)
	if os.path.exists(full_model_path) is False:
		print('Start initializing the model...')
		# First time training
		weight_init(model)
		start_epoch = 0
		print('OK!')
	else:
		# Continue training
		print('Start loading the pretrained model...')
		checkpoint = torch.load(full_model_path)
		model.load_state_dict(checkpoint['state_dict'])
		start_epoch = checkpoint['epoch'] + 1
		# load the optimizer
		optimizer.load_state_dict(checkpoint['optimizer'])
		if is_cuda is True:
			for state in optimizer.state.values():
				for k, v in state.items():
					if isinstance(v, torch.Tensor):
						state[k] = v.cuda()
		# load the scheduler
		scheduler.load_state_dict(checkpoint['scheduler'])
		print('OK!')
	model.train()

	# Model and log path
	if os.path.exists(model_path) is False:
		os.makedirs(model_path)
	if os.path.exists(log_path) is False:
		os.makedirs(log_path)

	# Start training
	for e in range(epoch):
		# train
		log_content = '[Epoch %d] Start training!' % (e + start_epoch)
		with open(log_path + log_name, 'a') as f:
			f.write(log_content + '\n')
			print(log_content)
		bs_train = 0
		for data in train_dataset_loader:
			img_data = Variable(data[0].cuda()) if is_cuda is True else Variable(data[0])
			model_data = Variable(data[1].float().cuda()) if is_cuda is True else Variable(data[1].float())

			pred = model(img_data)
			# loss
			loss = loss_func(pred, model_data)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			# iou
			iou = calculate_iou(pred, model_data, threshold)

			log_content = '[Epoch %d|Batch %d] Training: loss = %.6f, IoU = %.6f' % (e + start_epoch, bs_train, loss.item(), iou)
			print(log_content)
			with open(log_path + log_name, 'a') as f:
				f.write(log_content + '\n')
			bs_train += 1

		# validate
		if (e + 1) % eval_step == 0:
			with open(log_path + log_name, 'a') as f:
				f.write('[Epoch %d] Start validation!' % (e + start_epoch))
				model.eval()
				eval_iou = []
				bs_test = 0
				for data in test_dataset_loader:
					img_data = Variable(data[0].cuda()) if is_cuda is True else Variable(data[0])
					model_data = Variable(data[1].float().cuda()) if is_cuda is True else Variable(data[1].float())

					pred = model(img_data)
					iou = calculate_iou(pred, model_data, threshold)
					log_content = '[Epoch %d|Batch %d] Testing: IoU = %.6f' % (e + start_epoch, bs_test, iou)
					print(log_content)
					f.write(log_content + '\n')
					eval_iou.append(iou.item())
					bs_test += 1
				log_content = '[Epoch %d] The average testing IoU = %.6f' % (e + start_epoch, np.mean(eval_iou))
				print(log_content)
				f.write(log_content)
				model.train()

		# save model
		if (e + 1) % save_model_step == 0:
			states = {
				'epoch' : e + start_epoch,
				'state_dict' : model.state_dict(),
				'optimizer' : optimizer.state_dict(),
				'scheduler' : scheduler.state_dict()
			}
			torch.save(states, full_model_path)
			log_content = 'Save the model successfully!'
			print(log_content)
			f.write(log_content)

		# learning rate decays
		scheduler.step()

def main():
	parser = argparse.ArgumentParser()

	# Add arguments
	parser.add_argument('--bs', type = int, help = 'Batch size', default = 12)
	parser.add_argument('--sl', type = int, help = 'The length of sequence', default = 5)
	parser.add_argument('--e', type = int, help = 'Epoch', default = 40)
	parser.add_argument('--lr', type = float, help = 'Learning rate', default = 1e-4)
	parser.add_argument('--lrd', type = float, help = 'How much the learning rate decays each time', default = 0.1)
	parser.add_argument('--lrds', type = float, help = 'How many steps between two decays', default = 15)
	parser.add_argument('--es', type = int, help = 'How many steps between two evaluations', default = 1)
	parser.add_argument('--t', type = float, help = 'Threshold of voxelization', default = 0.4)
	parser.add_argument('--ss', type = int, help = 'Save model step', default = 5)
	parser.add_argument('--mp', type = str, help = 'Model path', default = './model/')
	parser.add_argument('--mn', type = str, help = 'Model name', default = 'model')
	parser.add_argument('--lp', type = str, help = 'Log file path', default = './log/03001627/')
	parser.add_argument('--ln', type = str, help = 'Log file name', default = 'log.txt')
	parser.add_argument('--train_img_root', type = str, help = 'The root of training image dataset', default = '/home/xulin/Documents/dataset/shapenet/dataset/train/imgs/03001627/')
	parser.add_argument('--train_model_root', type = str, help = 'The root of training model dataset', default = '/home/xulin/Documents/dataset/shapenet/dataset/train/models/03001627/')
	parser.add_argument('--test_img_root', type = str, help = 'The root of testing image dataset', default = '/home/xulin/Documents/dataset/shapenet/dataset/test/imgs/03001627/')
	parser.add_argument('--test_model_root', type = str, help = 'The root of testing model dataset', default = '/home/xulin/Documents/dataset/shapenet/dataset/test/models/03001627/')
	parser.add_argument('--img_pf', type = str, help = 'The postfix of images', default = '.png')
	parser.add_argument('--model_pf', type = str, help = 'The postfix of models', default = '.binvox')

	config = parser.parse_args()
	print("==========Configuration==========")
	print(config)
	print("=================================")
	start(config)

if __name__ == '__main__':
	main()