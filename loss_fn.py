import numpy as np
import pdb
import torch

EPSILON = 1e-35
MARGIN = 2

def kldiv(arr1,arr2):
	log_1 = np.log(arr1 + EPSILON)
	log_2 = np.log(arr2 + EPSILON)
	log_f = log_1 - log_2
	loss = np.multiply(arr1,log_f)
	loss = loss.numpy()
	loss = np.sum(loss)
	return loss


def calc_loss(p,q,is_simi,embeddings):
	loss = kldiv(embeddings[p],embeddings[q])
	if not is_simi:
		loss = MARGIN-loss
		loss = loss.clip(min=0)
	return loss


def total_loss(embeddings,labels):
	loss = 0.0 
	for i in range(embeddings.size(0)):
		for j in range(embeddings.size(0)):
			if i != j:
				similar = labels[i] == labels[j]
				loss += calc_loss(i,j,is_simi=similar,embeddings= embeddings)
	return loss


def calc_grad(p,q,factor,embeddings):
	val_1 = embeddings[p]
	val_2 = embeddings[q] + EPSILON
	value =  factor * torch.div(val_1,val_2)
	loss = kldiv(embeddings[p],embeddings[q])
	value = value.numpy()
	if factor == 1:
		if loss > MARGIN:
			value = np.zeros((4096))

	return value	


def total_grad(embeddings,labels):
	total_grad = np.zeros((embeddings.size(0),4096),dtype=np.float32)
	for i in range(embeddings.size(0)):
		for j in range(embeddings.size(0)):
			if i != j:
				if labels[i] == labels[j]:
					factor = -1
				else:
					factor = 1
				total_grad[i] += calc_grad(j,i,factor = factor,embeddings= embeddings)
	return total_grad


