import torch
import torch.nn as nn
from torch.autograd import Variable
from custom_alex import *
import torchvision.models as models 
from loss_fn import *


# Batch size and input dimensions

BATCH_SIZE = 20
IMAGE_SIZE = 224
IMAGE_CHAN = 3
EPOCHS = 1


# Input and Labels Batch Initialisation

input_batch = Variable(torch.randn(BATCH_SIZE,IMAGE_CHAN,IMAGE_SIZE,IMAGE_SIZE),requires_grad = False)
label_coarse = Variable(torch.randn(BATCH_SIZE,1),requires_grad = False)
label_fine = Variable(torch.randn(BATCH_SIZE,1),requires_grad = False)


model = CustomAlex(model_id = "A")

learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for i in range(EPOCHS):
	# Update the input batch

	output = model.forward(input_batch)
	total_loss = total_loss(output.data,label_fine.data)
	total_grad = torch.from_numpy(total_grad(output.data,label_fine.data))
	optimizer.zero_grad()
	output.backward(gradient = total_grad)
	optimizer.step()












