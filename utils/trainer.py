import torch
import sys
import os
import torch.nn.functional as F
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import numpy as np

class Trainer():
	def __init__(self, model, train_data, val_data, criterion, optimizer, max_epochs, device, path, patience, mask,
	 lilw=False, online_learning_epochs = 5):
		self.model = model
		self.train_data = train_data
		self.val_data = val_data
		self.criterion = criterion
		self.max_epochs = max_epochs
		self.device = device
		self.optimizer = optimizer
		self.path = path
		self.lilw = lilw
		self.mask_land = mask.to(self.device)
		self.online_learning_epochs = online_learning_epochs
		self.earlyStop = EarlyStop(patience, self.path)

	def train_evaluate(self):
		train_losses = []
		val_losses = []
		for epoch in range(self.max_epochs):
			self.train(train_losses, self.train_data)
			print('Train - Epoch %d, Epoch Loss: %f' % (epoch, train_losses[epoch]))
			self.evaluate(val_losses)
			print('Val Avg. Loss: %f' % (val_losses[epoch]))
			if (torch.cuda.is_available()):
				torch.cuda.empty_cache()
			if (self.earlyStop.check_stop_condition(epoch, self.model, self.optimizer, val_losses[epoch])):
				break
		#Fine-tune trained model with validation data so model is trained on data as close to prediction as possible
		self.load_model(self.path)
		for e in range(self.online_learning_epochs):
			self.train(train_losses, self.val_data)
			print('Online Training - Epoch %d, Epoch Loss: %f' % (e, train_losses[epoch+e+1]))
		self.earlyStop.save_model(epoch, self.model, self.optimizer, val_losses[epoch])
		return train_losses, val_losses

	def train(self, train_losses, train_set):
		train_loss = self.model.train()
		epoch_train_loss = 0.0
		for i, (x, y) in enumerate(train_set):
			x,y = x.to(self.device), y.to(self.device)
			self.optimizer.zero_grad()
			if (self.lilw):
				output = self.model(x, original_x = x)
			else:
				output = self.model(x)
			#Disregard undefined pixels
			output = output*self.mask_land
			if (self.cut_output and not self.recurrent_model):
				loss = self.criterion(output[:,:,0,:,:], y[:,:,0,:,:])
			else:
				loss = self.criterion(output, y)
			loss.backward()
			self.optimizer.step()
			epoch_train_loss += loss.detach().item()
		avg_epoch_loss = epoch_train_loss/len(train_set)
		train_losses.append(avg_epoch_loss)

	def evaluate(self, val_losses):
		epoch_val_loss = 0.0
		self.model.eval()
		with torch.no_grad():
			for i, (x, y) in enumerate(self.val_data):
				x,y = x.to(self.device), y.to(self.device)
				output = self.model(x)
				#Disregard undefined pixels
				output = output * self.mask_land
				loss = self.criterion(output, y)
				epoch_val_loss += loss.detach().item()
		avg_val_loss = epoch_val_loss/len(self.val_data)
		val_losses.append(avg_val_loss)

	def load_model(self, path):
		assert os.path.isfile(path)
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		print(f'Loaded model at path {path}, best epoch: {epoch}, best loss: {loss}')



class EarlyStop:
	def __init__(self, threshold, path):
		self.min_loss = sys.float_info.max
		self.count = 0
		self.threshold = threshold
		self.path = path
		
	def check_stop_condition(self, epoch, model, optimizer, loss):
		if (loss < self.min_loss):
			self.save_model(epoch, model, optimizer, loss)
			self.min_loss = loss
			self.count = 0
			return False
		else:
			self.count += 1
			if (self.count >= self.threshold):
				return True
			return False

	def reset(self, threshold):
		#self.min_loss = sys.float_info.max
		self.count = 0
		self.threshold = threshold

	def save_model(self, epoch, model, optimizer, loss):
		torch.save({
			'epoch': epoch,
			'model_state_dict': model.state_dict(),
			'optimizer_state_dict': optimizer.state_dict(),
			'loss': loss,
			}, self.path)
		print ('=> Saving a new best')

class Tester():
	def __init__(self, model, optimizer, criterion, test_data, device, model_name, mask, online_learning_epochs = 5):
		self.model = model
		self.model_name = model_name
		self.optimizer = optimizer
		self.criterion = criterion
		self.test_data = test_data
		self.device = device
		self.mask_land = mask.to(self.device)
		self.online_learning_epochs = online_learning_epochs

	def load_and_test(self, path):
		self.load_model(path)
		return self.test_model()

	def test_model(self):
		batch_rmse_loss = 0.0
		batch_mae_loss = 0.0
		batch_r2 = 0.0
		preds = []
		preds_border = []
		for i, (x, x_border, y, y_border) in enumerate(self.test_data):
			x,x_border,y,y_border = x.to(self.device), x_border.to(self.device), y.to(self.device), y_border.to(self.device)
			self.model.eval()
			with torch.no_grad():
				output = self.model(x)
				#Disregard undefined pixels
				output = output * self.mask_land
				loss_rmse = self.criterion(output, y)
				loss_mae = F.l1_loss(output, y)
				r2,ar2 = self.report_r2(output.cpu(), y.cpu())
				for j in range(output.shape[0]):
					preds.append(output[j,0,0,:,:].cpu().numpy())
				batch_rmse_loss += loss_rmse.detach().item()
				batch_mae_loss += loss_mae.detach().item()
				batch_r2 += ar2
			for e in range(self.online_learning_epochs):
				loss = self.online_learning(x,y)
		rmse_loss = batch_rmse_loss/len(self.test_data)
		mae_loss = batch_mae_loss/len(self.test_data)
		r2_metric = batch_r2/len(self.test_data)
		self.save_prediction(preds, "region_3")
		return rmse_loss, mae_loss, r2_metric

	def online_learning(self, x, y):
		self.model.train()
		self.optimizer.zero_grad()
		x_in = x
		output = self.model(x_in, original_x = x)
		output = output*self.mask_land
		loss = self.criterion(output, y)
		loss.backward()
		self.optimizer.step()
		return loss

	def report_r2(self, y_pred, y_true):
		batch, ch, time, lat, lon = y_true.shape
		r2 = 0
		ar2 = 0
		for i in range(batch):
			for j in range(time):
				mse = metrics.mean_squared_error(y_true[i,0,j,:,:], y_pred[i,0,j,:,:]) 
				r2 += metrics.r2_score(y_true[i,0,j,:,:], y_pred[i,0,j,:,:])
				ar2 +=  1.0 - ( mse / y_true[i,0,j,:,:].var() )
		r2 = r2/(batch*time)
		ar2 = ar2 / (batch*time)
		return r2, ar2

	def load_model(self, path):
		assert os.path.isfile(path)
		checkpoint = torch.load(path)
		self.model.load_state_dict(checkpoint['model_state_dict'])
		self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		loss = checkpoint['loss']
		print(f'Loaded model at path {path}, best epoch: {epoch}, best loss: {loss}')

	def save_prediction(self, prediction, name):
		dataPath = 'data/medianmodel_acc_pred_'
		with open(dataPath+name+'.asc', 'wb') as f:
			np.save(f, prediction, allow_pickle=False)

