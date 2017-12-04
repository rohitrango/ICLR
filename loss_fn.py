import numpy as np
import pdb

EPSILON = 1e-35
MARGIN = 2


def calc_loss(p,q,is_simi,embeddings):
	log_1 = np.log(embeddings[p] + EPSILON)
	log_2 = np.log(embeddings[q] + EPSILON)
	log_f = log_1 - log_2
	loss = np.multiply(embeddings[p],log_f)
	loss = loss.numpy()
	if not is_simi:
		loss = MARGIN-loss
		loss = loss.clip(min=0)
	loss = np.sum(loss)
	return loss


def total_loss(embeddings,labels):
	loss = 0.0 
	for i in range(embeddings.size(0)):
		for j in range(embeddings.size(0)):
			similar = labels[i,0] == labels[j,0]
			loss += calc_loss(i,j,is_simi=similar,embeddings= embeddings)
	return loss


def calc_grad(p,q,factor,embeddings):
	val_1 = embeddings[p]
	val_2 = embeddings[q] + EPSILON
	value =  factor * np.divide(val_1,val_2)
	value = value.numpy()
	return value	


def total_grad(embeddings,labels):
	total_grad = np.zeros((embeddings.size(0),4096),dtype=np.float32)
	for i in range(embeddings.size(0)):
		for j in range(embeddings.size(0)):
			if labels[i,0] == labels[j,0]:
				factor = -1
			else:
				factor = 1
			total_grad[i] += calc_grad(j,i,factor = factor,embeddings= embeddings)
	pdb.set_trace()
	return total_grad


