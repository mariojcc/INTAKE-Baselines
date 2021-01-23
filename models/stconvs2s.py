import torch
import torch.nn as nn
from extras.evonorm import EvoNorm3D
from extras.conv2dlocal import Conv2dLocal

class EncoderSTCNN(nn.Module):
	
	def __init__(self, layer_size, hidden_dim, kernel_size, dropout_rate, device, lilw):
		super(EncoderSTCNN, self).__init__()
		self.padding = kernel_size // 2
		self.conv_layers = nn.ModuleList()
		self.batch_layers = nn.ModuleList()
		self.dropout_layers = nn.ModuleList()
		self.lilw = lilw
		
		spatial_kernel_size =  [1, kernel_size, kernel_size]
		spatial_padding =  [0, self.padding, self.padding]
		
		out_channels = hidden_dim
		in_channels = 1 + lilw
		for i in range(layer_size):
			self.conv_layers.append(
				nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
						  kernel_size=spatial_kernel_size, padding=spatial_padding, bias=False)
			)
			self.batch_layers.append(EvoNorm3D(out_channels, version = 'B0_3D', sequence=7))
			self.dropout_layers.append(nn.Dropout(dropout_rate))
			in_channels = out_channels + lilw
		
	def forward(self, x, lilw=None):
		for conv, batch, drop in zip(self.conv_layers, 
										   self.batch_layers, self.dropout_layers):
			x = conv(x)
			x = batch(x)
			if (self.lilw != 0):
				x = torch.cat((x,lilw), 1)
			x = drop(x)
			
		return x

class DecoderSTCNN(nn.Module):
	
	def __init__(self, layer_size, hidden_dim, kernel_size, dropout_rate, device, lilw):
		super(DecoderSTCNN, self).__init__()
		self.padding = kernel_size - 1
		self.conv_layers = nn.ModuleList()
		self.batch_layers = nn.ModuleList()
		self.dropout_layers = nn.ModuleList()
		self.lilw = lilw

		temporal_kernel_size =  [kernel_size, 1, 1]
		temporal_padding =  [self.padding, 0, 0]
		
		out_channels = hidden_dim
		in_channels = hidden_dim + self.lilw
		for i in range(layer_size):
			self.conv_layers.append(
				nn.Conv3d(in_channels=in_channels, out_channels=out_channels, 
						  kernel_size=temporal_kernel_size, padding=temporal_padding, bias=False)
			)
			self.batch_layers.append(EvoNorm3D(out_channels, version = 'B0_3D', sequence=7))
			self.dropout_layers.append(nn.Dropout(dropout_rate))
			in_channels = out_channels + self.lilw

		padding_final = [kernel_size // 2, 0, 0]
		self.conv_final = nn.Conv3d(in_channels=in_channels, out_channels=1, 
			  kernel_size=temporal_kernel_size, padding=padding_final, bias=True)
		
	def forward(self, x, lilw = None):
		for conv, batch, drop in zip(self.conv_layers, 
										   self.batch_layers, self.dropout_layers):
			x = conv(x)[:,:,:-self.padding,:,:]
			x = batch(x)
			if (self.lilw != 0):
				x = torch.cat((x,lilw), 1)
			x = drop(x)
			
		out = self.conv_final(x)
		return out

class STConvS2S(nn.Module):
	
	def __init__(self, input_shape, layer_size, hidden_dim, kernel_size, dropout_rate, forecasting_horizon, version, device):
		super(STConvS2S, self).__init__()
		self.device = device
		self.version = version
		lilw = 0
		if (self.version in [3,4]):
			lilw = 4
			self.li_layer = Conv2dLocal(input_shape[3], input_shape[4], in_channels = input_shape[2], out_channels = 2, kernel_size = 1, bias = False)
			self.lw_layer = Conv2dLocal(input_shape[3], input_shape[4], in_channels = 1, out_channels = 2, kernel_size = 1, bias = False)


		self.encoder = EncoderSTCNN(layer_size, hidden_dim, kernel_size, dropout_rate, device, lilw)
		self.decoder = DecoderSTCNN(layer_size, hidden_dim, kernel_size, dropout_rate, device, lilw)

	def forward(self, x, original_x = None):
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
			out = self.decoder(x, lilw)
		else:
			x = self.encoder(x)
			out = self.decoder(x)
		return out[:,:,-1:,:,:]