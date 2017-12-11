import torch
import torch.nn as nn
from torch.autograd import Variable
from custom_alex import *
import torchvision.models as models 
from loss_fn import *
from torch.utils.data import *
import pdb
import json
# Batch size and input dimensions

BATCH_SIZE = 32
EPOCHS = 50

from birdsnap_dataset import *

train_dataset = BirdsnapDataset(root_dir='train',coarsef='birdsnap34.txt',transform=True)
test_dataset = BirdsnapDataset(root_dir='test',coarsef='birdsnap34.txt',transform=True)


train_iter = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_iter = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

model = nn.Sequential(
		CustomAlex(model_id = "A"),
		nn.Softmax(),
	).cuda()


optimizer =  torch.optim.SGD(model.parameters(), lr=0.01, dampening=0.001, momentum=0.9)

# Training Set
losses = []
	
for i in range(EPOCHS):
	losses.append([])
	for index, batch in enumerate(train_iter):
		print("Batch %d"%index)
		output = model.forward(Variable(batch["image"]).cuda())
		total_loss_val = total_loss(output, batch["fine_label"])
		total_grad_val = (total_grad(output, batch["fine_label"]))
		optimizer.zero_grad()
		# total_loss_val.backward()
		output.backward(gradient=total_grad_val)
		# pdb.set_trace()
		optimizer.step()
		print("Loss at Epoch %d Batch %d : %f , %f" % (i+1, index, total_loss_val.data[0], torch.sum(total_grad_val)))
		losses[-1].append(total_loss_val.data[0])

with open('losses.txt', 'w+') as f:
	f.write(json.dumps(losses))












