from utils.dataset import AscDatasets
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool 
from functools import partial
from statsmodels.tsa.api import VAR

def save_prediction(prediction):
	dataPath = 'data/medianmodel_acc_pred_var_'
	with open(dataPath+'.asc', 'wb') as f:
		np.save(f, prediction, allow_pickle=False)

if __name__ == '__main__':
	dataset = AscDatasets('data', '/medianmodel_acc', 3, 1, False, 0.2, 0.2, 14, 14)
	data = dataset.load_data_arima()
	data[data == -999.25] = np.nan
	mask = np.all(np.isnan(data), axis=0)
	data = data[:,~mask]
	print("Data shape:")
	print(data.shape)

	train = data[:-188]
	test = data[-181:]
	print("Train shape:")
	print(train.shape)
	print("Test shape:")
	print(test.shape)
	preds = []
	for i in range(181):
		forecasting_model = VAR(train)
		results = forecasting_model.fit(4)
		results = results.forecast(y = train, steps=7)
		preds.append(results[-1])
		train = np.append(train, test[i:i+1,:], 0)
	preds = np.array(preds)
	save_prediction(preds)
