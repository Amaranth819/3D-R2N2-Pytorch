import glob
import shutil
import math
import os
import argparse

def split_dataset(cate, root_dir, img_top_dir, model_top_dir, scale):
	train_root_img_dir = root_dir + 'dataset/train/imgs/'
	train_root_model_dir = root_dir + 'dataset/train/models/'
	test_root_img_dir = root_dir + 'dataset/test/imgs/'
	test_root_model_dir = root_dir + 'dataset/test/models/'

	all_img_dirs = os.listdir(img_top_dir + '/' + cate)
	all_model_dirs = os.listdir(model_top_dir + '/' + cate)
	num_imgs = len(all_img_dirs)
	num_models = len(all_model_dirs)

	# Eliminate those having either images or models.
	intersection = list(set(all_img_dirs).intersection(set(all_model_dirs)))
	num_intersection = len(intersection)
	print('%s: originally %d images and %d models. The intersection has %d pairs of images and models.' % (cate, num_imgs, num_models, num_intersection))

	training_size = int(num_intersection * scale)

	for sub_cate in intersection[:training_size]:
		source_img_path = '%s/%s/%s' % (img_top_dir, cate, sub_cate)
		source_model_path = '%s/%s/%s' % (model_top_dir, cate, sub_cate)
		des_img_path = '%s/%s/%s' % (train_root_img_dir, cate, sub_cate)
		des_model_path = '%s/%s/%s' % (train_root_model_dir, cate, sub_cate)
		shutil.copytree(source_img_path, des_img_path)
		shutil.copytree(source_model_path, des_model_path)

	for sub_cate in intersection[training_size:]:
		source_img_path = '%s/%s/%s' % (img_top_dir, cate, sub_cate)
		source_model_path = '%s/%s/%s' % (model_top_dir, cate, sub_cate)
		des_img_path = '%s/%s/%s' % (test_root_img_dir, cate, sub_cate)
		des_model_path = '%s/%s/%s' % (test_root_model_dir, cate, sub_cate)
		shutil.copytree(source_img_path, des_img_path)
		shutil.copytree(source_model_path, des_model_path)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--root', help = 'The root directory of ShapeNet dataset.', default = '/home/xulin/Documents/dataset/shapenet/')
	config = parser.parse_args()
	print(config)

	root_dir = config.root
	img_top_dir, model_top_dir = root_dir + 'ShapeNetRendering', root_dir + 'ShapeNetVox32'

	if os.path.exists(img_top_dir) is False and os.path.exists(model_top_dir) is False:
		raise ValueError('Unknown dataset path!')

	cates_model = os.listdir(model_top_dir)
	cates_imgs = os.listdir(img_top_dir)
	cates = list(set(cates_model).intersection(set(cates_imgs)))
	for cate in cates:
		split_dataset(cate, root_dir, img_top_dir, model_top_dir, 0.9)

if __name__ == '__main__':
	main()