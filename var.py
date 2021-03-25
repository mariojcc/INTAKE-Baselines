from utils.dataset import AscDatasets
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool 
from functools import partial
from statsmodels.tsa.api import VAR

def save_prediction(prediction):
	dataPath = 'data/medianmodel_acc_pred_var'
	with open(dataPath+'.asc', 'wb') as f:
		np.save(f, prediction, allow_pickle=False)

if __name__ == '__main__':
	dataset = AscDatasets('data', '/medianmodel_acc', 3, 1, False, 0.2, 0.2, 7, 7)
	data = dataset.load_data_arima()
	data[data == -999.25] = np.nan
	data = np.diff(np.diff(data))

	train_data = data[:-188] #165 #158
	test_data = data[-181:]
	#train = train_data.T
	#test = test_data.T
	mask = np.all(np.isnan(train_data), axis=0)
	train = train_data[:,~mask]
	mask = np.all(np.isnan(test_data), axis=0)
	test = test_data[:,~mask]
	print(train.shape)
	print(test.shape)
	for i in range(181):
		preds = []
		forecasting_model = VAR(train)
		results = forecasting_model.fit(7)
		results = results.forecast(y = train, steps=7)
		print(results.shape)
		preds.append(results[-1])
		print(len(preds[0]))
		train = np.append(train, test[:,i:i+1], 1)
		print(train.shape)
	save_prediction(preds)