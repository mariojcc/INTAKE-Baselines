import torch
from torch.utils.data import Dataset
import numpy as np
import os.path
from os import path
from sklearn.preprocessing import MinMaxScaler

class NCDFDatasets():
	def __init__(self, data, val_split, test_split, cut_y=False,  data_type='Prediction'):
		self.train_data = NCDFDataset(data, data.sample.size, test_split, val_split, data_type, False, False, cut_y)
		self.val_data = NCDFDataset(data, data.sample.size, test_split, val_split, data_type, False, True, cut_y)
		self.test_data = NCDFDataset(data, data.sample.size, test_split, val_split, data_type, True, False, cut_y)

	def get_train(self):
		return self.train_data
	def get_val(self):
		return self.val_data
	def get_test(self):
		return self.test_data

class NCDFDataset(Dataset):
	def __init__(self, data, sampleSize, test_split, val_split, data_type, is_test=False, is_val=False, cut_y=False):
		super(NCDFDataset, self).__init__()
		self.cut_y = cut_y
		self.reconstruction = True if data_type == 'Reconstruction' else False 

		splitter = DataSplitter(data, sampleSize, test_split, val_split)
		if (is_test):
			dataset = splitter.split_test()
		elif (is_val):
			dataset = splitter.split_val()
		else:
			dataset = splitter.split_train()

		#batch, channel, time, lat, lon
		self.x = torch.from_numpy(dataset.x.values).float().permute(0, 4, 1, 2, 3)
		if (self.cut_y):
			self.y = torch.from_numpy(dataset.y.values).float().permute(0, 4, 1, 2, 3)[:,:,0,:,:]
		else:
			self.y = torch.from_numpy(dataset.y.values).float().permute(0, 4, 1, 2, 3)
		del dataset

		if (self.reconstruction):
			data_cat = torch.cat((self.x, self.y), 2)
			self.y = data_cat.clone().detach()
			self.x, self.removed = self.removeObservations(data_cat.clone().detach())

	def __getitem__(self, index):
		if (self.reconstruction):
			return (self.x[index,:,:,:,:], self.y[index,:,:,:,:], self.removed[index])
		elif (self.cut_y):
			return (self.x[index,:,:5,:,:], self.y[index,:,:,:])
		else:
			return (self.x[index,:,:5,:,:], self.y[index,:,:,:,:])

	def __len__(self):
		return self.x.shape[0]

	def removeObservations(self, data):
		removed_observations = torch.zeros(data.shape[0], dtype=torch.long)
		new_data = torch.zeros(data.shape[0], data.shape[1], data.shape[2]-1, data.shape[3], data.shape[4])
		for i in range(data.shape[0]):
			index = np.random.randint(0, data.shape[2])
			#new_data[i] = torch.cat([data[i, :, :index, :, :], data[i, :, index+1:, :, :]], dim=1)
			data[i,:,index,:,:] = torch.empty(data.shape[1], data.shape[3], data.shape[4]).fill_(-1)
			removed_observations[i] = index
		return data, removed_observations

class AscDatasets():
	def __init__(self, dataPath, dataDestination, subregions, current_region, scale, val_split, test_split, x_seq_len, y_seq_len):
		self.dataPath = dataPath
		self.dataDestination = dataDestination
		self.val_split = val_split
		self.test_split = test_split
		self.subregions = subregions
		self.current_region = current_region
		self.scale = scale
		self.x_seq_len = x_seq_len
		self.y_seq_len = y_seq_len
		self.cutoff = None

		if (path.exists(self.dataPath + self.dataDestination + '_x.asc')):
			self.dataX, self.dataY = self.load_data()
		else:
			self.dataX, self.dataY = self.processData()
			self.save_data()
		self.dataX, self.dataY = self.replace_missing_values(self.dataX, self.dataY, 0.0)
		self.split()
		if (self.scale):
			self.scale_data()
		print(self.dataX.shape)
		print(self.dataY.shape)
		self.train_data = AscDataset(self.train_data_x, self.train_data_y)
		if (self.val_split == 0):
			self.val_data = None
		else:
			self.val_data = AscDataset(self.val_data_x, self.val_data_y)
		self.test_data = AscDataset(self.test_data_x, self.test_data_y, border_data_x = self.test_data_border_x, border_data_y = self.test_data_border_y)

	def get_train(self):
		return self.train_data
	def get_val(self):
		return self.val_data
	def get_test(self):
		return self.test_data
	def get_test_border(self):
		return self.test_data_border

	def processData(self):
		months = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec','Jan','Feb']
		months31Days = ['Aug', 'Jul','Mar','May','Oct', 'Dec','Jan']
		days=[]
		for i in range(1,32):
			if (i < 10):
				i = '0' + str(i)
			days.append(str(i))
		dataX = []
		dataY = []
		data = []
		count = 0
		singleSequenceX = []
		singleSequenceY = []
		dataPrefix = self.dataPath + '/medianmodel_'
		numberFiles = len(months)*(len(days)-1)
		xSeqDone = False
		for i in range(len(months)):
			for j in range(len(days)):
				if (not j == 30 or months[i] in months31Days):
					if (months[i] in ['Jan','Feb']):
						dataPath = dataPrefix + days[j] + '-' + months[i] + '-2021.asc'
					else:
						dataPath = dataPrefix + days[j] + '-' + months[i] + '-2020.asc'
					if (not path.exists(dataPath)):
						continue
					data.append(np.genfromtxt(dataPath, dtype=None, skip_header = 6))
					count += 1
					if ("01-Sep-2020" in dataPath):
						index = len(data)-1
		data = np.array(data)
		print("FILE COUNT:" + str(count))
		for i in range(data.shape[0]):
			if (i+self.x_seq_len+self.y_seq_len <= data.shape[0]):
				singleSequenceX = data[i:i+self.x_seq_len, :, :]
				singleSequenceY = data[i+self.x_seq_len+self.y_seq_len-1:i+self.x_seq_len+self.y_seq_len, :, :]
				if (i+self.x_seq_len+self.y_seq_len-1 == index):
					print("Defining cutoff")
					self.cutoff = i
				dataX.append(singleSequenceX)
				dataY.append(singleSequenceY)
		assert len(dataX) == len(dataY)
		npDataX = np.array(dataX)
		npDataY = np.array(dataY)
		#assert (npDataX[0,1,:,:].all() == npDataX[1,0,:,:].all())
		#assert(npDataY[0,0,:,:].all() == npDataX[1,self.x_seq_len-1,:,:].all())
		print(npDataX.shape)
		print(npDataY.shape)
		return npDataX,npDataY

	def process_data_arima(self):
		months = ['Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec', 'Jan', 'Feb']
		months31Days = ['Aug', 'Jul','Mar','May','Oct', 'Dec', 'Jan']
		days=[]
		for i in range(1,32):
			if (i < 10):
				i = '0' + str(i)
			days.append(str(i))
		dataX = []
		dataY = []
		data = []
		dataPrefix = self.dataPath + '/medianmodel_'
		numberFiles = len(months)*(len(days)-1)
		for i in range(len(months)):
			for j in range(len(days)):
				if (not j == 30 or months[i] in months31Days):
					if (months[i] in ['Jan', 'Feb']):
						dataPath = dataPrefix + days[j] + '-' + months[i] + '-2021.asc'
					else:
						dataPath = dataPrefix + days[j] + '-' + months[i] + '-2020.asc'
					if (not path.exists(dataPath)):
						continue
					data.append(np.genfromtxt(dataPath, dtype=None, skip_header = 6))
					if ("01-Sep-2020" in dataPath):
						index = len(data)-1
		data = np.array(data)
		data_ravel = []
		for i in range(data.shape[0]):
			data_ravel.append(data[i].ravel())
		data_ravel = np.array(data_ravel)
		with open(self.dataPath+self.dataDestination+'_arima.asc', 'wb') as f:
			np.save(f, data_ravel, allow_pickle=False)
		return data_ravel

	def save_data(self):
		with open(self.dataPath+self.dataDestination+'_x.asc', 'wb') as f:
			np.save(f, self.dataX, allow_pickle=False)
		with open(self.dataPath+self.dataDestination+'_y.asc', 'wb') as f:
			np.save(f, self.dataY, allow_pickle=False)

	def load_data(self):
		with open(self.dataPath+self.dataDestination+'_x.asc', 'rb') as f:
			dataX = np.load(f)
		with open(self.dataPath+self.dataDestination+'_y.asc', 'rb') as f:
			dataY = np.load(f)
		return dataX, dataY

	def load_data_arima(self):
		path_arima = self.dataPath+self.dataDestination+'_arima.asc'
		if (path.exists(path_arima)):
			with open(path_arima, 'rb') as f:
				data_arima = np.load(f)
		else:
			data_arima = self.process_data_arima()
		return data_arima

	def split(self):
		#test_cutoff = int(self.dataX.shape[0] * self.test_split)
		if (self.cutoff is None):
			#Index for Sep 1st prediction
			self.cutoff = 152
		val_cutoff = int(self.cutoff * self.val_split)
		#instances, sequence, height, width
		train_data_x = self.dataX[0:self.cutoff - val_cutoff]
		train_data_y = self.dataY[0:self.cutoff - val_cutoff]
		self.train_data_x, self.train_data_y = self.calculate_sub_regions(train_data_x, train_data_y)
		assert self.train_data_x.shape[0] == self.train_data_y.shape[0]
		val_data_x = self.dataX[self.cutoff - val_cutoff:self.cutoff]
		val_data_y = self.dataY[self.cutoff - val_cutoff:self.cutoff]
		self.val_data_x, self.val_data_y = self.calculate_sub_regions(val_data_x, val_data_y)
		assert self.val_data_x.shape[0] == self.val_data_y.shape[0]
		test_data_x = self.dataX[self.cutoff: self.dataX.shape[0]]
		test_data_y = self.dataY[self.cutoff: self.dataY.shape[0]]
		self.test_data_x, self.test_data_y = self.calculate_sub_regions(test_data_x, test_data_y)
		assert self.test_data_x.shape[0] == self.test_data_y.shape[0]
		self.test_data_border_x, self.test_data_border_y = self.calculate_sub_regions(test_data_x, test_data_y, 10)
		assert self.test_data_border_x.shape[0] == self.test_data_border_y.shape[0]

	def calculate_sub_regions(self, data_x, data_y, step = 0):
		print(data_x.shape)
		cut_height = int(data_x.shape[2] / self.subregions)
		remainder = data_x.shape[2] % self.subregions
		start = cut_height * (self.current_region-1)
		if (remainder > 0 and self.current_region == self.subregions):
			cut_height += remainder
		data_x = data_x[:,:,start+step:start+cut_height+step,:]
		data_y = data_y[:,:,start+step:start+cut_height+step,:]


		'''cut_width = int(data_x.shape[3] / self.subregions)
		remainder = data_x.shape[3] % self.subregions
		start = cut_width * (self.current_region-1)
		if (remainder > 0 and self.current_region == self.subregions):
			cut_width += remainder
		data_x = data_x[:,:,:,start:start+cut_width]
		data_y = data_y[:,:,:,start:start+cut_width]'''
		return data_x, data_y

	def replace_missing_values(self, dataX, dataY, value):
		dataX[dataX == -999.250] = value
		dataY[dataY == -999.250] = value
		return dataX, dataY

	def scale_data(self):
		self.scaler = MinMaxScaler(feature_range=(-1,1))
		totalTrainingData = np.append(self.train_data_x, self.train_data_y).reshape(-1, 1)
		self.scaler.fit(totalTrainingData)
		batch, time, height, width = self.train_data_x.shape
		for i in range(batch):
			for j in range(time):	
				self.train_data_x[i,j,:,:] = self.scaler.transform(self.train_data_x[i,j,:,:])
				if (j < self.train_data_y.shape[1]):
					self.train_data_y[i,j,:,:] = self.scaler.transform(self.train_data_y[i,j,:,:])
				if (i < self.val_data_x.shape[0]):
					self.val_data_x[i,j,:,:] = self.scaler.transform(self.val_data_x[i,j,:,:])
					if (j < self.val_data_y.shape[1]):
						self.val_data_y[i,j,:,:] = self.scaler.transform(self.val_data_y[i,j,:,:])
				if (i < self.test_data_x.shape[0]):
					self.test_data_x[i,j,:,:] = self.scaler.transform(self.test_data_x[i,j,:,:])
					if (j < self.test_data_y.shape[1]):
						self.test_data_y[i,j,:,:] = self.scaler.transform(self.test_data_y[i,j,:,:])

	def unscale_data(self, data):
		batch,ch,time,height,width = data.shape
		for i in range(batch):
			for j in range(time):
				data[i,0,j,:,:] = self.scaler.inverse_transform(data[i,0,j,:,:])
		return data

	def get_mask_land(self):
		filename = 'mask.npy'
		mask_land = np.load(filename)
		mask_land = torch.from_numpy(mask_land).float()
		cut_height = int(mask_land.shape[3] / self.subregions)
		remainder = mask_land.shape[3] % self.subregions
		start = cut_height * (self.current_region-1)
		if (remainder > 0 and self.current_region == self.subregions):
			cut_height += remainder
		mask_land = mask_land[:,:,:,start:start+cut_height,:]
		print(mask_land.shape)
		return mask_land

class AscDataset(Dataset):
	def __init__(self, dataX, dataY, data_format='numpy', border_data_x = None, border_data_y = None):
		#batch, channel, time, width, height
		self.border_x = None
		self.border_y = None
		if (data_format == 'numpy'):
			self.x = torch.from_numpy(dataX).float().unsqueeze(1)
			self.y = torch.from_numpy(dataY).float().unsqueeze(1)
			if not (border_data_x is None):
				self.border_x = torch.from_numpy(border_data_x).float().unsqueeze(1)
				self.border_y = torch.from_numpy(border_data_y).float().unsqueeze(1)
		elif (data_format == 'tensor'):
			self.x = dataX
			self.y = dataY
			if not (border_data_x is None):
				self.border_x = border_data_x
				self.border_y = border_data_y
		else:
			raise ValueError("Invalid Data Format")


	def normalize(self, x, min_range, max_range):
		min_val = np.amin(x)
		max_val = np.amax(x)
		return min_range + ((x-min_val)*(max_range - min_range))/(max_val - min_val)

	def __getitem__(self, index):
		if not (self.border_x is None):
			return (self.x[index,:,:,:,:], self.border_x[index,:,:,:,:], self.y[index,:,:,:,:], self.border_y[index,:,:,:,:])
		return (self.x[index,:,:,:,:], self.y[index,:,:,:,:])

	def __len__(self):
		return self.x.shape[0]


class DataSplitter():
	def __init__(self, data, sampleSize, val_split=0, test_split=0):
		self.val_split = val_split
		self.test_split = test_split
		self.data = data
		self.sampleSize = sampleSize

	def split_train(self):
		test_cutoff = int(self.sampleSize * self.test_split)
		val_cutoff = int(self.sampleSize * self.val_split)
		return self.data[dict(sample=slice(0, self.data.sample.size - val_cutoff - test_cutoff))]

	def split_val(self):
		test_cutoff = int(self.sampleSize * self.test_split)
		val_cutoff = int(self.sampleSize * self.val_split)
		return self.data[dict(sample=slice(self.data.sample.size - val_cutoff - test_cutoff, self.data.sample.size - test_cutoff))] 

	def split_test(self):
		test_cutoff = int(self.sampleSize * self.test_split)
		return self.data[dict(sample=slice(self.data.sample.size - test_cutoff, None))]


