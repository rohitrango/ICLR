import torch
import torch.nn as nn
from torch.autograd import Variable
from custom_alex import *
import torchvision.models as models 
from loss_fn import *
from torch.utils.data import *
import pdb

# Batch size and input dimensions

BATCH_SIZE = 20
EPOCHS = 1

from birdsnap_dataset import *

train_dataset = BirdsnapDataset(root_dir='train',coarsef='birdsnap34.txt',transform=True)
test_dataset = BirdsnapDataset(root_dir='test',coarsef='birdsnap34.txt',transform=True)


train_iter = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_iter = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

model = nn.Sequential(
		CustomAlex(model_id = "A"),
		nn.Softmax(),
	)

learning_rate = 1e-3

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training Set
for i in range(EPOCHS):
	for index,batch in enumerate(train_iter):
		output = model.forward(Variable(batch["image"]))
		total_loss_val = total_loss(output.data,batch["fine_label"])
		total_grad_val = torch.from_numpy(total_grad(output.data,batch["fine_label"])).float()
		optimizer.zero_grad()
		output.backward(gradient=total_grad_val)
		optimizer.step()
		print("Loss at Epoch %d Batch %d : %f" % (i+1,index+1,total_loss_val))
	












