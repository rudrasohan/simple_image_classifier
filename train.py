import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from torch.autograd import Variable 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from model import Net
from get_data import *

#show image using matplotlib
def imshow(img):
	img = img/2 + 0.5 #unnormalize
	npimg = img.numpy()
	plt.imshow(np.transpose(npimg, (1, 2, 0)))
	#plt.show() #for displaying


#random images
dataiter = iter(trainloader)
images, labels = dataiter.next()
imshow(torchvision.utils.make_grid(images))
#display labels
print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

#initialize the net
net = Net()
criterion = nn.CrossEntropyLoss() #loss function
optimizer = optim.Adam(net.parameters(), lr=0.001, betas=(0.9,0.999), eps=1e-08) #optimizer

#running the net
for epoch in range(2): #set the no of epochs
	
	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		#get inputs
		inputs, labels = data
		#wrap them in Variables
		inputs, labels = Variable(inputs), Variable(labels)
		#initialize parameter gradients
		optimizer.zero_grad()
		#backprop
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()
		#print stats
		running_loss += loss.data[0]
		if i % 2000 == 1999: #print every 2000 mini batches
			print('[%d, %5d] loss: %.3f' %(epoch +1,i +1, running_loss/2000))
print('Finished Training')

torch.save(net, 'img_classify.pt')#saving model
