import sys
sys.path.append('../lib')

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from layer import Residual_Block, BN_FCConv3D, Unpool3D
from torch.utils.checkpoint import checkpoint

class Encoder(nn.Module):
	def __init__(self, seq_len, n_gru_vox, deconv_filters):
		super(Encoder, self).__init__()

		# Parameters
		self.conv_filters = [3, 96, 128, 256, 256, 256, 256]
		self.fc_layers_size = [1024]
		self.ceil_mode = True
		self.img_h, self.img_w = 127, 127
		self.n_gru_vox = n_gru_vox
		self.deconv_filters = deconv_filters
		self.seq_len = seq_len

		# Build the network
		self.build()

	def build(self):
		'''
			Encoder
		'''
		# layer 1
		self.layer1 = Residual_Block(self.conv_filters[0], self.conv_filters[1], ks = 7, pad = 3)
		self.layer1_pool = nn.MaxPool2d(2, ceil_mode = self.ceil_mode)

		# layer 2
		self.layer2_sc = nn.Sequential(
			nn.Conv2d(self.conv_filters[1], self.conv_filters[2], kernel_size = 1),
			nn.BatchNorm2d(self.conv_filters[2])
		)
		self.layer2 = Residual_Block(self.conv_filters[1], self.conv_filters[2], shortcut = self.layer2_sc)
		self.layer2_pool = nn.MaxPool2d(2, ceil_mode = self.ceil_mode)

		# layer 3
		self.layer3_sc = nn.Sequential(
			nn.Conv2d(self.conv_filters[2], self.conv_filters[3], kernel_size = 1),
			nn.BatchNorm2d(self.conv_filters[3])
		)
		self.layer3 = Residual_Block(self.conv_filters[2], self.conv_filters[3], shortcut = self.layer3_sc)
		self.layer3_pool = nn.MaxPool2d(2, ceil_mode = self.ceil_mode)

		# layer 4
		self.layer4 = Residual_Block(self.conv_filters[3], self.conv_filters[4])
		self.layer4_pool = nn.MaxPool2d(2, ceil_mode = self.ceil_mode)

		# layer 5
		self.layer5_sc = nn.Sequential(
			nn.Conv2d(self.conv_filters[4], self.conv_filters[5], kernel_size = 1),
			nn.BatchNorm2d(self.conv_filters[5])
		)
		self.layer5 = Residual_Block(self.conv_filters[4], self.conv_filters[5], shortcut = self.layer5_sc)
		self.layer5_pool = nn.MaxPool2d(2, ceil_mode = self.ceil_mode)

		# layer 6
		self.layer6_sc = nn.Sequential(
			nn.Conv2d(self.conv_filters[5], self.conv_filters[6], kernel_size = 1),
			nn.BatchNorm2d(self.conv_filters[6])
		)
		self.layer6 = Residual_Block(self.conv_filters[5], self.conv_filters[6], shortcut = self.layer6_sc)
		self.layer6_pool = nn.MaxPool2d(2, ceil_mode = self.ceil_mode)

		# final layer size
		fh, fw = self.fm_size()
		fcs_size = fh * fw * self.conv_filters[-1]

		# fc layers
		self.fcs = nn.Linear(fcs_size, self.fc_layers_size[0])


		'''
			GRU3d
		'''
		conv3d_filter_shape = (self.deconv_filters[0], self.deconv_filters[0], 3, 3, 3)
		self.gru3d_u = BN_FCConv3D(self.fc_layers_size[-1], \
			conv3d_filter_shape, self.deconv_filters[0], self.n_gru_vox, self.seq_len)
		self.gru3d_r = BN_FCConv3D(self.fc_layers_size[-1], \
			conv3d_filter_shape, self.deconv_filters[0], self.n_gru_vox, self.seq_len)
		self.gru3d_rs = BN_FCConv3D(self.fc_layers_size[-1], \
			conv3d_filter_shape, self.deconv_filters[0], self.n_gru_vox, self.seq_len)
		self.gru3d_sigmoid = nn.Sigmoid()
		self.gru3d_tanh = nn.Tanh()

	def fm_size(self):
		h = self.img_h
		w = self.img_w
		for i in range(len(self.conv_filters) - 1):
			if self.ceil_mode is True:
				h = math.ceil(h / 2)
				w = math.ceil(w / 2)
			else:
				h = math.floor(h / 2)
				w = math.floor(w / 2)
		return int(h), int(w)

	def forward(self, x, h, u, idx):
		bs = x.size()[0]
		# encoder
		x = self.layer1(x)
		x = self.layer1_pool(x)
		x = self.layer2(x)
		x = self.layer2_pool(x)
		x = self.layer3(x)
		x = self.layer3_pool(x)
		x = self.layer4(x)
		x = self.layer4_pool(x)
		x = self.layer5(x)
		x = self.layer5_pool(x)
		x = self.layer6(x)
		x = self.layer6_pool(x)
		x = x.view(bs, -1)
		x = self.fcs(x)

		# gru
		update = self.gru3d_sigmoid(self.gru3d_u(x, h, idx))
		reset = self.gru3d_sigmoid(self.gru3d_r(x, h, idx))
		rs = self.gru3d_tanh(self.gru3d_rs(x, reset * h, idx))
		x = update * h + (1.0 - update) * rs

		return x, update


class Decoder(nn.Module):
	def __init__(self, deconv_filters):
		super(Decoder, self).__init__()

		# Parameter
		self.deconv_filters = deconv_filters

		self.build()

	def build(self):
		self.decoder_unpool0 = nn.ConvTranspose3d(self.deconv_filters[0], self.deconv_filters[0], kernel_size = 2, stride = 2)
		self.decoder_block0 = nn.Sequential(
			nn.Conv3d(self.deconv_filters[0], self.deconv_filters[1], 3, padding = 1),
			nn.BatchNorm3d(self.deconv_filters[1]),
			nn.LeakyReLU(negative_slope = 0.1, inplace = True),
			nn.Conv3d(self.deconv_filters[1], self.deconv_filters[1], 3, padding = 1),
			nn.BatchNorm3d(self.deconv_filters[1])
		)

		self.decoder_unpool1 = nn.ConvTranspose3d(self.deconv_filters[1], self.deconv_filters[1], kernel_size = 2, stride = 2)
		self.decoder_block1 = nn.Sequential(
			nn.Conv3d(self.deconv_filters[1], self.deconv_filters[2], 3, padding = 1),
			nn.BatchNorm3d(self.deconv_filters[2]),
			nn.LeakyReLU(negative_slope = 0.1, inplace = True),
			nn.Conv3d(self.deconv_filters[2], self.deconv_filters[2], 3, padding = 1),
			nn.BatchNorm3d(self.deconv_filters[2])
		)

		self.decoder_unpool2 = nn.ConvTranspose3d(self.deconv_filters[2], self.deconv_filters[2], kernel_size = 2, stride = 2)
		self.decoder_block2 = nn.Sequential(
			nn.Conv3d(self.deconv_filters[2], self.deconv_filters[3], 3, padding = 1),
			nn.BatchNorm3d(self.deconv_filters[3]),
			nn.LeakyReLU(negative_slope = 0.1, inplace = True),
			nn.Conv3d(self.deconv_filters[3], self.deconv_filters[3], 3, padding = 1),
			nn.BatchNorm3d(self.deconv_filters[3])
		)
		self.decoder_block2_shortcut = nn.Sequential(
			nn.Conv3d(self.deconv_filters[2], self.deconv_filters[3], 1),
			nn.BatchNorm3d(self.deconv_filters[3])
		)

		self.decoder_block3 = nn.Sequential(
			nn.Conv3d(self.deconv_filters[3], self.deconv_filters[4], 3, padding = 1),
			nn.BatchNorm3d(self.deconv_filters[4]),
			nn.LeakyReLU(negative_slope = 0.1, inplace = True),
			nn.Conv3d(self.deconv_filters[4], self.deconv_filters[4], 3, padding = 1),
			nn.BatchNorm3d(self.deconv_filters[4]),
			nn.LeakyReLU(negative_slope = 0.1, inplace = True),
			nn.Conv3d(self.deconv_filters[4], self.deconv_filters[4], 3, padding = 1),
			nn.BatchNorm3d(self.deconv_filters[4])
		)
		self.decoder_block3_shortcut = nn.Sequential(
			nn.Conv3d(self.deconv_filters[3], self.deconv_filters[4], 1),
			nn.BatchNorm3d(self.deconv_filters[4])
		)
		
		self.decoder_block4 = nn.Sequential(
			nn.Conv3d(self.deconv_filters[4], self.deconv_filters[5], 3, padding = 1),
			nn.LeakyReLU(negative_slope = 0.1, inplace = True)
		)

	def forward(self, x):
		x = self.decoder_unpool0(x)
		p = self.decoder_block0(x)
		x = F.leaky_relu(x + p, inplace = True)

		x = self.decoder_unpool1(x)
		p = self.decoder_block1(x)
		x = F.leaky_relu(x + p, inplace = True)

		x = self.decoder_unpool2(x)
		p1 = self.decoder_block2(x)
		p2 = self.decoder_block2_shortcut(x)
		x = F.leaky_relu(p1 + p2, inplace = True)

		p1 = self.decoder_block3(x)
		p2 = self.decoder_block3_shortcut(x)
		x = F.leaky_relu(p1 + p2, inplace = True)

		x = self.decoder_block4(x)

		return x


class Res_Gru_Net(nn.Module):
	def __init__(self, seq_len):
		super(Res_Gru_Net, self).__init__()

		self.deconv_filters = [128, 128, 128, 64, 32, 2]
		self.n_gru_vox = 4
		self.seq_len = seq_len
		self.encoder = Encoder(seq_len, self.n_gru_vox, self.deconv_filters)
		self.decoder = Decoder(self.deconv_filters)

	def hidden_init(self, shape):
		h = torch.zeros(shape).type(torch.FloatTensor)
		if torch.cuda.is_available() is True:
			h = h.type(torch.cuda.FloatTensor) 
		return torch.autograd.Variable(h)

	def forward(self, x):
		'''
			x: [bs, seq_len, c, h, w]
		'''
		# encoder
		bs, seq_len = x.size()[:2]
		h_shape = (bs, self.deconv_filters[0], self.n_gru_vox, self.n_gru_vox, self.n_gru_vox)
		h = self.hidden_init(h_shape)
		u = self.hidden_init(h_shape)

		for idx in range(self.seq_len):
			h, u = self.encoder(x[:, idx, ...], h, u, idx)

		# decoder
		h = self.decoder(h)
		return h

def calculate_iou(pred, model_data, threshold):
	occupy = pred[:, 1, ...] >= threshold
	model_data = model_data[:, 1, ...]
	if torch.cuda.is_available() is True:
		occupy = occupy.type(torch.cuda.ByteTensor)
		model_data = model_data.type(torch.cuda.ByteTensor)
	else:
		occupy = occupy.type(torch.ByteTensor)
		model_data = model_data.type(torch.ByteTensor)
	logic_and = torch.sum(occupy & model_data)
	logic_or = torch.sum(occupy | model_data)
	iou = logic_and.float() / logic_or.float()
	return iou

def weight_init(network):
	for each_module in network.modules():
		if isinstance(each_module, (nn.Conv2d, nn.Conv3d)):
			torch.nn.init.xavier_uniform_(each_module.weight)
			if each_module.bias is not None:
				each_module.bias.data.zero_()
		elif isinstance(each_module, (nn.BatchNorm2d, nn.BatchNorm3d)):
			each_module.weight.data.fill_(1.)
			if each_module.bias is not None:
				each_module.bias.data.zero_()
		elif isinstance(each_module, nn.Linear):
			each_module.weight.data.normal_(0, 0.01)
			if each_module.bias is not None:
				each_module.bias.data.zero_()

if __name__ == '__main__':
	x = torch.zeros((16, 5, 3, 127, 127), dtype = torch.float16)
	x = torch.autograd.Variable(x)
	net = Res_Gru_Net(5)
	weight_init(net)
	net = net.half()
	if torch.cuda.is_available() is True:
		x = x.cuda()
		net = net.cuda()
	y = net(x)
	print(y.size())