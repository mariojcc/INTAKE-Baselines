import argparse as arg
import torch
import numpy as np
import random as rd
import os
import torch
import adamod
import pathlib
from datetime import datetime
from torch.utils.data import DataLoader

from utils.dataset import AscDatasets
from utils.dataset import AscDataset
from utils.trainer import Trainer
from utils.trainer import Tester
from models.stconvs2s import STConvS2S, STConvS2S_U
from models.stfd import ST_RFD, ST_CFD 
from models.convlstm import STConvLSTM
from sklearn.model_selection import TimeSeriesSplit

class RMSELoss(torch.nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.eps = eps
        
    def forward(self,yhat,y):
        loss = torch.sqrt(self.mse(yhat,y) + self.eps)
        return loss

def prepare_dataset(args):
	dataPath = 'data'
	dataDestination = '/medianmodel_acc'
	val_split = 0
	test_split = args.split
	if (not args.cross_validation):
		val_split = args.split
	data = AscDatasets(dataPath, dataDestination, args.region_division, args.input_region, args.scale, val_split, test_split, args.x_sequence_len, args.forecasting_horizon)
	mask = data.get_mask_land()
	return data, mask

def create_loaders(args, train_data, test_data, val_data = None):
	params = {'batch_size': args.batch, 'num_workers': args.workers, 'worker_init_fn': init_seed}
	params_test = {'batch_size': 1, 'num_workers': args.workers, 'worker_init_fn': init_seed}
	train_loader = DataLoader(dataset=train_data, shuffle=True, **params)
	if (val_data != None):
		val_loader = DataLoader(dataset=val_data, shuffle=False, **params)
	else:
		val_loader = None
	test_loader = DataLoader(dataset=test_data, shuffle=False, **params_test)
	return train_loader, test_loader, val_loader

def set_seed(iteration):
	seed = (iteration * 10) + 1000
	np.random.seed(seed)
	rd.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic=True
	torch.set_deterministic(True)
	return seed

def init_seed(iteration):
	seed = (iteration * 10) + 1000
	np.random.seed(seed)

def create_path(model, version):
	newDirectory = os.path.join('models', model)
	os.makedirs(newDirectory, exist_ok=True)
	model_path = os.path.join(newDirectory, model + '_' + str(version) + '_' + datetime.now().strftime('m%md%d-h%Hm%Ms%S') + '.pth.tar')
	return model_path

def build_model(args, device, iteration):
	print(f'Iteration: {iteration}')
	seed = set_seed(iteration)
	models = {
		'st-rfd': ST_RFD,
		'st-cfd': ST_CFD,
		'stconvs2s': STConvS2S_U,
		'convlstm': STConvLSTM
	}
	data, mask = prepare_dataset(args)
	train_data = data.get_train()
	val_data = data.get_val()
	test_data = data.get_test()

	model = models[args.model](train_data.x.shape, args.num_layers, args.hidden_dim, args.kernel_size, args.dropout, args.forecasting_horizon, args.version, device)
	model.to(device)
	print(model)

	criterion = torch.nn.MSELoss()

	opt_params = {'lr': 0.001, 'beta3': 0.999}
	optimizer = adamod.AdaMod(model.parameters(), **opt_params)

	train_loader, test_loader, val_loader = create_loaders(args, train_data, test_data, val_data)

	model_path = create_path(args.model, args.version)
	if (args.model == 'st-rfd'):
		recurrent_model = True
	else:
		recurrent_model = False
	if (args.version == 1):
		trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, args.epoch, device, model_path, args.patience, mask, recurrent_model= recurrent_model)
	if (args.version == 3):
		trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, args.epoch, device, model_path, args.patience, mask, recurrent_model= recurrent_model, lilw = True)

	if (args.cross_validation):
		tscv = TimeSeriesSplit()
		fold = 0
		total_val_loss = 0.0
		for train_index, val_index in tscv.split(train_data.x):
			fold += 1
			x_train_fold, x_val_fold = train_data.x[train_index], train_data.x[val_index]
			y_train_fold, y_val_fold = train_data.y[train_index], train_data.y[val_index]
			train = AscDataset(x_train_fold, y_train_fold, data_format = 'tensor')
			val = AscDataset(x_val_fold, y_val_fold, data_format = 'tensor')
			train_loader, val_loader, _ = create_loaders(args, train, val)
			trainer.train_data = train_loader
			trainer.val_data = val_loader
			trainer.earlyStop.reset(args.patience)
			print(f'---------------Fold {fold} ---------------')
			print("-----Train-----")
			print("Index:", train_index)
			print("X : ", train.x.shape)
			print("Y : ", train.y.shape)
			print("-----Val-----")
			print("Index:", val_index)
			print("X : ", val.x.shape)
			print("Y : ", val.y.shape)
			train_losses, val_losses = trainer.train_evaluate()
			'''model = models[args.model](train_data.x.shape, args.num_layers, args.hidden_dim, args.kernel_size, args.dropout, args.forecasting_horizon, args.version, device)
			model.to(device)
			trainer.model = model
			optimizer = adamod.AdaMod(model.parameters(), **opt_params)
			trainer.optimizer = optimizer
			total_val_loss += val_losses[-1]'''
		#print(f"Validation Loss after {tscv.get_n_splits()} splits: {total_val_loss / tscv.get_n_splits()}")
	else:
		print("-----Train-----")
		print("X : ", train_data.x.shape)
		print("Y : ", train_data.y.shape)
		print("-----Val-----")
		print("X : ", val_data.x.shape)
		print("Y : ", val_data.y.shape)
		print("-----Test-----")
		print("X : ", test_data.x.shape)
		print("Y : ", test_data.y.shape)
		train_losses, val_losses = trainer.train_evaluate()

	if (args.disjoint):
		test_data.x = test_data.x[::args.x_sequence_len, :, :, :, :]
		test_data.y = test_data.y[::args.x_sequence_len, :, :, :, :]
		assert test_data.x.shape[0] == test_data.y.shape[0]
		params = {'batch_size': args.batch, 'num_workers': args.workers, 'worker_init_fn': init_seed}
		test_loader = DataLoader(dataset=test_data, shuffle=False, **params)
		print("-----Test-----")
		print("X : ", test_data.x.shape)
		print("Y : ", test_data.y.shape)
		_, test_loader, _ = create_loaders(args, train_data, test_data, val_data)

	tester = Tester(model, optimizer, criterion, test_loader, device, False, args.model + '_' + str(args.version) + '_' + str(args.input_region),  mask, recurrent_model=False)
	if (args.scale):
		rmse,mae,r2 = tester.load_and_test(trainer.path, data)
	else:
		rmse, mae, r2 = tester.load_and_test(trainer.path)

	return rmse,mae,r2



def run_model(args, device):
	test_rmse, test_mae, test_r2 = [],[],[]
	for i in range(args.iteration):
		rmse, mae, r2 = build_model(args, device, i)
		test_rmse.append(rmse)
		test_mae.append(mae)
		test_r2.append(r2)
	mean_rmse, std_rmse = np.mean(test_rmse), np.std(test_rmse)
	mean_mae, std_mae = np.mean(test_mae), np.std(test_mae)
	mean_r2, std_r2 = np.mean(test_r2), np.std(test_r2)
	print('============================================')
	print(f'Test set MSE mean: {mean_rmse}, std: {std_rmse}')
	print(f'Test set MAE mean: {mean_mae}, std: {std_mae}')
	print(f'Test set R2 mean: {mean_r2}, std: {std_r2}')
	print('============================================')

if __name__ == '__main__':
	parser = arg.ArgumentParser()
	parser.add_argument('-rd', '--region-division', type=int, choices = [1,2,3,4,5], default=1)
	parser.add_argument('-ir', '--input-region', type=int, default=1)
	parser.add_argument('-e',  '--epoch', type=int, default=100)
	parser.add_argument('-b',  '--batch', type=int, default=15)
	parser.add_argument('-p',  '--patience', type=int, default=10)
	parser.add_argument('-w',  '--workers', type=int, default=4)
	parser.add_argument('-m',  '--model', type=str, choices=['st-rfd', 'st-cfd', 'stconvs2s', 'convlstm'], default='st-rfd')
	parser.add_argument('-l',  '--num-layers', type=int, dest='num_layers', default=3)
	parser.add_argument('-d',  '--hidden-dim', type=int, dest='hidden_dim', default=32)
	parser.add_argument('-k',  '--kernel-size', type=int, dest='kernel_size', default=5)
	parser.add_argument('-dr', '--dropout', type=float, default=0.0)
	parser.add_argument('-iw', '--input-window', type=int, default = 5)
	parser.add_argument('-fh', '--forecasting-horizon', type=int, default=5)
	parser.add_argument('-i',  '--iteration', type=int, default=1)
	parser.add_argument('-cv', '--cross-validation', type=bool, default=False)
	parser.add_argument('-sp', '--split', type=float, default = 0.2)
	# 1 = base model, 2 = base + gridmask, 3 = base + lilw, 4 = base + gridmask + lilw
	parser.add_argument('-v',  '--version', type=int, choices=[1,2,3,4], default=1)
	parser.add_argument('-s',  '--scale', type=bool, default=False)
	parser.add_argument('-xsl',  '--x-sequence-len', type=int, default=5)
	parser.add_argument('-dj',  '--disjoint', type=bool, default=False)

	args = parser.parse_args()

	if (args.input_region > args.region_division):
		parser.error('Current input region has to be lower or equal to number of input sub-regions')

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	print(f'Model: {args.model.upper()}')
	print(f'Device: {device}') 
	print(f'Settings: {args}')

	run_model(args, device)


