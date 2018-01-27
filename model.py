import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		# 3 input image channel, 6 output channels, 5x5 square convolution
		self.conv1 = nn.Conv2d(3, 6, 5)
		# 6 input image channel, 16 output channels, 5x5 square convolution
		self.conv2 = nn.Conv2d(6, 16, 5)
		#fully connected layer with 16*5*5 input and 120 output neuros
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		#fully connected layer with 120 input and 84 output neuros
		self.fc2 = nn.Linear(120, 84)
		#fully connected layer with 84 input and 10 output neuros
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		#Max pooling a (2,2) window(relu over conv output)
		x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
		#Max pooling a (2,2) window(relu over conv output) 
		x = F.max_pool2d(F.relu(self.conv2(x)), 2) #2 used to specify (2,2) square
		#feature flatning
		x = x.view(-1, self.num_flat_features(x))#-1 to approriately adjust size
		#relu activation
		x =F.relu(self.fc1(x))
		#relu activation
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

	def num_flat_features(self, x):
		size = x.size()[1:]
		num_features = 1
		for s in size:
			num_features *= s 
		return num_features







