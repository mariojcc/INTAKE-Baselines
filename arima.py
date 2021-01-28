from utils.dataset import AscDatasets
import statsmodels.api as sm
import numpy as np
from sklearn.metrics import mean_squared_error
from multiprocessing import Pool 
from functools import partial

def save_prediction(prediction):
	dataPath = 'data/medianmodel_acc_pred_arima_1'
	with open(dataPath+'.asc', 'wb') as f:
		np.save(f, prediction, allow_pickle=False)

def rmse(y_actual, y_predicted):
    return sqrt(mean_squared_error(y_actual, y_predicted) + 1e-6)

def run_arima(data, train_data, i):
	if (-999.250 in train_data):
		preds = [-999.250 for i in range(122)]
		return preds
	else:
		print(i)
		print(train_data.shape)
		in_x = np.copy(train_data)
		preds = []
		for j in range(122):
			start_index = len(train_data) + j
			end_index = start_index + 6
			arima = sm.tsa.statespace.SARIMAX(in_x, order=(7,0,1), enforce_stationarity=False) #try with MA as (1,0,0,0,0,0,1) 
			results = arima.fit(disp=False, maxiter=100) 
			pred_sequence = results.predict(start=start_index, end=end_index, dynamic=False)
			preds.append(pred_sequence[-1])
			in_x = np.append(in_x, data[start_index, i])
		return preds

if __name__ == '__main__':
	dataset = AscDatasets('data', '/medianmodel_acc', 3, 1, False, 0.2, 0.2, 7, 7)
	data = dataset.load_data_arima()

	train_data = data[:-129] #165 #158
	test_data = data[-122:]
	#preds = np.empty([122, test_data.shape[1]])
	train = train_data.T[int(train_data.shape[1]/2):]
	idxs = list(range(len(train)))
	idxs = [x+int(train_data.shape[1]/2) for x in idxs]
	with Pool() as pool:
		func = partial(run_arima, data)
		pred_matrix = pool.starmap(func, zip(train, idxs))
		preds = np.array(pred_matrix)
		#shape 122 x time series
		preds = preds.T
		print(preds.shape)
		pool.close()
		pool.join()
	'''for i in range(test_data.shape[1]): #calculate for each time-series
		if (-999.250 in train_data[:,i]):
			preds[:,i] = -999.250
			continue
		else:
			in_x = train_data[:,i]
			print(i)
			for j in range(test_data.shape[0]):
				start_index = len(train_data) + j
				end_index = start_index + 6
				arima = sm.tsa.statespace.SARIMAX(in_x, order=(7,0,1), enforce_stationarity=False) #try with MA as (1,0,0,0,0,0,1) 
				results = arima.fit(disp=False, maxiter=100) 
				pred_sequence = results.predict(start=start_index, end=end_index, dynamic=False)
				preds[j, i] = pred_sequence[-1]
				in_x = np.append(in_x, data[start_index, i])
			save_prediction(preds)'''
	#preds = preds.reshape(preds.shape[0], 288, 141)
	save_prediction(preds)
	#rmse = sqrt(mean_squared_error(test_data, preds))
	#mse = mean_squared_error(test_data, preds)
	#print('Test RMSE: %.3f' % rmse)
	#print('Test MSE: %.3f' % mse)