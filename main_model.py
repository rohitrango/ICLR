import torch
import torch.nn as nn
from torch.autograd import Variable
from custom_alex import *
import torchvision.models as models 
from loss_fn import *
from torch.utils.data import *
import pdb

# Batch size and input dimensions

BATCH_SIZE = 16
EPOCHS = 50

from birdsnap_dataset import *

train_dataset = BirdsnapDataset(root_dir='train',coarsef='birdsnap34.txt',transform=True)
test_dataset = BirdsnapDataset(root_dir='test',coarsef='birdsnap34.txt',transform=True)


train_iter = DataLoader(train_dataset,batch_size=BATCH_SIZE,shuffle=True)
test_iter = DataLoader(test_dataset,batch_size=BATCH_SIZE,shuffle=True)

model = nn.Sequential(
		CustomAlex(model_id = "A"),
		nn.Softmax(),
	)


optimizer =  torch.optim.SGD(model.parameters(), lr = 0.01, momentum=0.9)



# Training Set
for i in range(EPOCHS):
	for index,batch in enumerate(train_iter):
		output = model.forward(Variable(temp["image"]))
		total_loss_val = total_loss(output.data,temp["fine_label"])
		total_grad_val = torch.from_numpy(total_grad(output.data,temp["fine_label"])).float()
		optimizer.zero_grad()
		output.backward(gradient=total_grad_val)
		pdb.set_trace()
		optimizer.step()
		print("Loss at Epoch %d Batch %d : %f" % (i+1,index,total_loss_val))
			












