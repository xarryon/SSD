import torch.nn as nn

class TConvINRelu(nn.Module):
	"""
	A sequence of Convolution, Instance Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride):
		super(TConvINRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.ConvTranspose2d(channels_in, channels_out, 2, stride, padding=0),
			nn.InstanceNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)

class ConvINRelu(nn.Module):
	"""
	A sequence of Convolution, Instance Normalization, and ReLU activation
	"""

	def __init__(self, channels_in, channels_out, stride):
		super(ConvINRelu, self).__init__()

		self.layers = nn.Sequential(
			nn.Conv2d(channels_in, channels_out, 3, stride, padding=1),
			nn.InstanceNorm2d(channels_out),
			nn.ReLU(inplace=True)
		)

	def forward(self, x):
		return self.layers(x)


class ConvBlock(nn.Module):
	'''
	Network that composed by layers of ConvINRelu
	'''

	def __init__(self, in_channels, out_channels, blocks=1, stride=1):
		super(ConvBlock, self).__init__()

		layers = [ConvINRelu(in_channels, out_channels, stride)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = ConvINRelu(out_channels, out_channels, 1)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)


class TConvBlock(nn.Module):
	'''
	Network that composed by layers of TConvINRelu
	'''

	def __init__(self, in_channels, out_channels, blocks=1, stride=2):
		super(TConvBlock, self).__init__()

		layers = [TConvINRelu(in_channels, out_channels, stride)] if blocks != 0 else []
		for _ in range(blocks - 1):
			layer = TConvINRelu(out_channels, out_channels, 2)
			layers.append(layer)

		self.layers = nn.Sequential(*layers)

	def forward(self, x):
		return self.layers(x)