import torch
import torch.nn.functional as F
from extras.evonorm import EvoNorm3D
from extras.conv2dlocal import Conv2dLocal
from extras.gridmask import GridMask

class FeedBackwardConv3d(torch.nn.Conv3d):
		def __init__(self, in_channels, out_channels, kernel_size, stride=1,
								 padding=0, dilation=1,
								 bias=False, padding_mode='zeros', weight=None):
				super().__init__(in_channels, out_channels, kernel_size, stride=stride,
								 padding=padding, dilation=dilation,
								 bias=bias, padding_mode=padding_mode)
				
		def forward(self,input, weight=None):
				if (weight is not None):
						return F.conv3d(input, weight.permute(1,0,2,3,4), self.bias, self.stride,
												self.padding, self.dilation)
				else:
						return F.conv3d(input, self.weight, self.bias, self.stride,
												self.padding, self.dilation)

class Encoder(torch.nn.Module):
	def __init__(self, layer_size, hidden_dim, kernel_size, dropout_rate, device, lilw = 0, reduce_layer = False):
		super(Encoder, self).__init__()
		self.device = device
		self.layer_size = layer_size
		self.reduce_layer = reduce_layer
		self.conv_layers = torch.nn.ModuleList()
		self.bn_layers = torch.nn.ModuleList()
		self.decode_bn_layers = torch.nn.ModuleList()
		self.dropout_layers = torch.nn.ModuleList()
		self.lilw = lilw
		
		self.kernel_size = [1, kernel_size, kernel_size]
		self.padding = [0, kernel_size // 2, kernel_size // 2]
		
		in_channels = 1 + self.lilw
		out_channels = hidden_dim 
		for i in range(self.layer_size):
			self.conv_layers.append(FeedBackwardConv3d(in_channels, out_channels,self.kernel_size, padding = self.padding))
			self.bn_layers.append(EvoNorm3D(out_channels, version = 'B0_3D', sequence=5))
			self.dropout_layers.append(torch.nn.Dropout(dropout_rate))
			self.decode_bn_layers.append(EvoNorm3D(32, version = 'B0_3D', sequence=5))
			in_channels = out_channels + self.lilw
		self.conv_reduce = FeedBackwardConv3d(in_channels, 1, 1)

	def forward(self, x, lilw = None, decode=False):
		if (decode):
			if (self.reduce_layer):
				x = self.conv_reduce(x, self.conv_reduce.weight)
			for i in range(self.layer_size-1, -1, -1):
				if (self.lilw != 0):
					x = x[:, 0:32, :, :, :]
				x = self.decode_bn_layers[i](x)
				x = self.dropout_layers[i](x)
				x = self.conv_layers[i](x, self.conv_layers[i].weight)
		else:
			for i in range(self.layer_size):
				x = self.conv_layers[i](x)
				x = self.bn_layers[i](x)
				if (self.lilw != 0):
					x = torch.cat((x,lilw), 1)
				x = self.dropout_layers[i](x)
			if (self.reduce_layer):
				x = self.conv_reduce(x)
		return x

class DecoderSTCFD(torch.nn.Module):
		
	def __init__(self, layer_size, hidden_dim, kernel_size, dropout_rate, device, lilw = 0,):
		super(DecoderCNN, self).__init__()
		self.padding = kernel_size // 2
		self.conv_layers = torch.nn.ModuleList()
		self.batch_layers = torch.nn.ModuleList()
		self.dropout_layers = torch.nn.ModuleList()
		self.lilw = lilw

		temporal_kernel_size =  [kernel_size, 1, 1]
		temporal_padding =  [self.padding, 0, 0]
		
		out_channels = hidden_dim
		in_channels = hidden_dim + self.lilw
		for i in range(layer_size):
			self.conv_layers.append(
					torch.nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
										kernel_size=temporal_kernel_size, padding=temporal_padding, bias=False)
			)
			self.batch_layers.append(EvoNorm3D(out_channels, version = 'B0_3D', sequence=5))
			self.dropout_layers.append(torch.nn.Dropout(dropout_rate))
			in_channels = out_channels + self.lilw

	def forward(self, x, lilw=None):
		for conv, batch, drop in zip(self.conv_layers, self.batch_layers, self.dropout_layers):
			x = conv(x)[:,:,:-self.padding,:,:]
			x = batch(x)
			if (self.lilw != 0):
				x = torch.cat((x,lilw), 1)
			x = drop(x)
		return x

class ST_CFD(torch.nn.Module):
	def __init__(self, input_shape, layer_size, hidden_dim, kernel_size, dropout_rate, forecasting_horizon, version, device):
		super(ST_CFD, self).__init__()
		self.device = device
		self.forecasting_horizon = forecasting_horizon
		self.encoder = Encoder(layer_size, hidden_dim, kernel_size, dropout_rate, device)
		self.decoder = DecoderSTCFD(layer_size, hidden_dim, kernel_size, dropout_rate, device)
		temporal_kernel_size =  [kernel_size, 1, 1]
		padding_final = [kernel_size // 2, 0, 0]
		self.conv_final = torch.nn.Conv3d(in_channels=1, out_channels=1, 
					kernel_size=temporal_kernel_size, padding=padding_final, bias=True)

		if (self.version in [3,4]):
			self.encoder.lilw = 2
			self.li_layer = Conv2dLocal(input_shape[3], input_shape[4], in_channels = input_shape[2], out_channels = 2, kernel_size = 1, bias = False)
			self.lw_layer = Conv2dLocal(input_shape[3], input_shape[4], in_channels = 1, out_channels = 2, kernel_size = 1, bias = False)   

	def forward(self, x):
		batch, channel, time, height, width = x.size()
		z = x
		li = torch.ones(z.squeeze(1).shape).to(self.device)
		li = self.li_layer(li).unsqueeze(2).expand(-1,-1,time,-1,-1)
		z = z.view(batch*time, channel, height, width)
		lw = self.lw_layer(z).contiguous().view(batch, self.encoder.lilw, time, height, width)
		lilw = torch.cat((li,lw), 1)
		x = torch.cat((x,lilw), 1)

		x = self.encoder(x, lilw)  
		x = self.decoder(x, lilw)
		x = self.encoder(x, decode=True)
		x = self.conv_final(x)
		return x

class ST_RFD(torch.nn.Module):
	def __init__(self, input_shape, layer_size, hidden_dim, kernel_size, dropout_rate, forecasting_horizon, version, device):
		super(ST_RFD, self).__init__()
		self.version = version
		self.device = device
		self.forecasting_horizon = forecasting_horizon
		self.recurrent_encoder = torch.nn.LSTMCell(input_shape[3]*input_shape[4], input_shape[3]*input_shape[4]); 
		if (self.version == 1):
			self.encoder = Encoder(layer_size, hidden_dim, kernel_size, dropout_rate, device,  reduce_layer=True)
		if (self.version in [3,4]):
			#Update to send number of LI and LW in arguments
			self.encoder = Encoder(layer_size, hidden_dim, kernel_size, dropout_rate, device, lilw = 4,  reduce_layer=True)
			self.li_layer = Conv2dLocal(input_shape[3], input_shape[4], in_channels = input_shape[2], out_channels = 2, kernel_size = 1, bias = False)
			self.lw_layer = Conv2dLocal(input_shape[3], input_shape[4], in_channels = 1, out_channels = 2, kernel_size = 1, bias = False)

		 

	def forward(self, x, states, original_x = None):
		batch, channel, time, height, width = x.size()
		if (self.version in [3,4]):
			#z = original_x if self.training else x
			z = x
			li = torch.ones(z.squeeze(1).shape).to(self.device)
			li = self.li_layer(li).unsqueeze(2).expand(-1,-1,time,-1,-1)
			z = z.view(batch*time, channel, height, width)
			lw = self.lw_layer(z).contiguous().view(batch, 2, time, height, width)
			lilw = torch.cat((li,lw), 1)
			x = torch.cat((x,lilw), 1)
			x = self.encoder(x, lilw)
		else:
			x = self.encoder(x)
		
		x = x.squeeze().view(batch, time, -1)
		h = states[0]
		c = states[1]
		for i in range(time):
			h,c = self.recurrent_encoder(x[:,i,:],(h,c))
				
		outputs = torch.zeros(batch, self.forecasting_horizon, height*width, device=self.device)    
		inputLSTM = torch.zeros(h.size(), device=self.device)
		for i in range(self.forecasting_horizon):
			h,c = self.recurrent_encoder(inputLSTM,(h,c))
			inputLSTM = h
			outputs[:,i,:] = h
				
		x = outputs.contiguous().view(batch, channel, self.forecasting_horizon, height, width)
		x = self.encoder(x, decode=True)
		if (self.version in [3,4]):
			x = x[:,0:1,:,:,:]
		x = x.view(batch, channel, self.forecasting_horizon, height, width)
		return x