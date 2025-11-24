#edit by guodong fan
import torchvision
from model.netrgb_max import Network
import argparse
import os
import random
import warnings
from PIL import Image
import cv2
from torchvision import transforms
warnings.filterwarnings("ignore")
from iqa import comput
import torch
import numpy as np
def resize_image_by_long_edge(image):
	height, width = image.shape[:2]
	i = image.shape[0]
	j = image.shape[1]
	if image.shape[0] % 4 != 0:
		while i % 4 != 0:
			i += 1
	if image.shape[1] % 4 != 0:
		while j % 4 != 0:
			j += 1
	image = cv2.resize(image, (j, i))
	lq_img = (np.asarray(image) / 255.0)
	lq_img = torch.from_numpy(lq_img).float().permute(2, 0, 1).cuda().unsqueeze(0)
	return lq_img,(height,width)

def test(args):
	t = 0.
	InputImages = os.listdir(args.TestFolderPath + '/')
	os.makedirs(args.toPath + '/', exist_ok=True)
	l = []
	for num in range(len(InputImages)):
		print("\nImages Processed: %d/ %d  \r" % (num + 1, len(InputImages)))
		Input = np.array(Image.open(args.TestFolderPath + '/' + InputImages[num]).convert('RGB'))
		Input,target_size = resize_image_by_long_edge(Input)
		transform = transforms.Resize(target_size,interpolation=transforms.InterpolationMode.NEAREST_EXACT)
		Input = Input.to(device)
		losslist = []
		print(args.TestFolderPath + '/' + InputImages[num])
		_model = Network(args.gamma, args.down, args.denoise,)
		_model.res_conv1.apply(_model.weights_init)
		_model.to(device)
		_model.train()
		optimizer = torch.optim.AdamW(_model.parameters(), lr=args.lr, betas=(0.99, 0.999), eps=1e-08,
		                              weight_decay=1e-2)
		for k in range(args.iter):
			loss,I = _model._loss(Input)
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		with torch.no_grad():
			_add, out, I,_max = _model.test(Input)
			out = transform(out)
			torchvision.utils.save_image(out, args.toPath + '/' + InputImages[num])

		l.append(losslist)


if __name__ == '__main__':
	modelname = 'netrgb'
	dataset = 'DICM'
	parser = argparse.ArgumentParser()
	parser.add_argument('--TestFolderPath', type=str, default=r'Inputdir/')
	parser.add_argument('--toPath', type=str, default='# Outputdir/' )
	parser.add_argument('--iter', type=int, default=40)
	parser.add_argument('--pretrain', type=int, default=0)
	parser.add_argument('--lr', type=float, default=0.001)
	parser.add_argument('--gamma', type=float, default=1)
	parser.add_argument('--down', type=float, default=0.5)
	parser.add_argument('--denoise', type=bool, default=False)#for LOL
	parser.add_argument('--exp', type=str, default="0.5size")#
	parser.add_argument('--comput', type=bool, default=True)
	parser.add_argument('--metric_mode', type=str, default='noref')#
	parser.add_argument('--path', type=str, default='weights_1_64.pt')#
	args = parser.parse_args()
	torch.manual_seed(1143)
	torch.cuda.manual_seed_all(1143)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	device = 'cuda'
	np.random.seed(1143)
	random.seed(1143)
	test(args)

	if args.comput:
		if args.metric_mode == 'ref':
			comput(args.toPath, 'ssim', dataset, modelname)
			comput(args.toPath, 'psnr', dataset, modelname)
		else:
			comput(args.toPath, 'niqe', dataset, modelname)
