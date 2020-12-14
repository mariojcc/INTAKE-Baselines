import torch
import torch.nn as nn
from extras.evonorm import EvoNorm3D

class EncoderSTCNN(nn.Module):
	
	def __init__(self, layer_size, hidden_dim, kernel_size, dropout_rate, device):
		super(EncoderSTCNN, self).__init__()
		self.padding = kernel_size // 2
		self.conv_layers = nn.ModuleList()
		self.batch_layers = nn.ModuleList()
		self.dropout_layers = nn.ModuleList()
		
		spatial_kernel_size =  [1, kernel_size, kernel_size]
		spatial_padding =  [0, self.padding, self.padding]
		
		out_channels = hidden_dim
		in_channels = 1
		for i in range(layer_size):
			self.conv_layers.append(
				nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
						  kernel_size=spatial_kernel_size, padding=spatial_padding, bias=False)
			)
			self.batch_layers.append(EvoNorm3D(out_channels, version = 'B0_3D'))
			self.dropout_layers.append(nn.Dropout(dropout_rate))
			in_channels = out_channels
		
	def forward(self, x):
		for conv, batch, drop in zip(self.conv_layers, 
										   self.batch_layers, self.dropout_layers):
			x = conv(x)
			x = batch(x)
			x = drop(x)
			
		return x

class DecoderSTCNN(nn.Module):
	
	def __init__(self, layer_size, hidden_dim, kernel_size, dropout_rate, device):
		super(DecoderSTCNN, self).__init__()
		self.padding = kernel_size - 1
		self.conv_layers = nn.ModuleList()
		self.batch_layers = nn.ModuleList()
		self.dropout_layers = nn.ModuleList()

		temporal_kernel_size =  [kernel_size, 1, 1]
		temporal_padding =  [self.padding, 0, 0]
		
		out_channels = hidden_dim
		in_channels = hidden_dim
		for i in range(layer_size):
			self.conv_layers.append(
				nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
						  kernel_size=temporal_kernel_size, padding=temporal_padding, bias=False)
			)
			self.batch_layers.append(EvoNorm3D(out_channels, version = 'B0_3D'))
			self.dropout_layers.append(nn.Dropout(dropout_rate))
			in_channels = out_channels

		padding_final = [kernel_size // 2, 0, 0]
		self.conv_final = nn.Conv3d(in_channels=in_channels, out_channels=1, 
			  kernel_size=temporal_kernel_size, padding=padding_final, bias=True)
		
	def forward(self, x):
		for conv, batch, drop in zip(self.conv_layers, 
										   self.batch_layers, self.dropout_layers):
			x = conv(x)[:,:,:-self.padding,:,:]
			x = batch(x)
			x = drop(x)
			
		out = self.conv_final(x)
		return out

class STConvS2S(nn.Module):
	
	def __init__(self, input_shape, layer_size, hidden_dim, kernel_size, dropout_rate, forecasting_horizon, device):
		super(STConvS2S, self).__init__()
		self.device = device


		self.encoder = EncoderSTCNN(layer_size, hidden_dim, kernel_size, dropout_rate, device)
		self.decoder = DecoderSTCNN(layer_size, hidden_dim, kernel_size, dropout_rate, device)
		
	def forward(self, x):
		out = self.encoder(x)
		return self.decoder(out)