import numpy as np
import pdb
import torch

EPSILON = 1e-35
MARGIN = 200

def kldiv(arr1,arr2):
	log_1 = torch.log(arr1 + EPSILON)
	log_2 = torch.log(arr2 + EPSILON)
	log_f = log_1 - log_2
	loss = torch.mul(arr1,log_f)
	# loss = loss.numpy()
	loss = torch.sum(loss)
	return loss


def calc_loss(p,q,is_simi,embeddings):
	loss = kldiv(embeddings[p],embeddings[q])
	if not is_simi:
		loss = MARGIN-loss
		# loss = loss.clip(min=0)
		loss = torch.clamp(loss, min=0.0)
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
	value =  factor * torch.div(val_1, val_2).data
	loss = kldiv(embeddings[p],embeddings[q]).data
	if factor == 1:
		if loss[0] > MARGIN:
			value = torch.zeros((1024)).cuda()
	return value	


def total_grad(embeddings, labels):
	total_grad = torch.zeros((embeddings.size(0), 1024)).cuda()
	for i in range(embeddings.size(0)):
		for j in range(embeddings.size(0)):
			if i != j:
				if labels[i] == labels[j]:
					factor = -1
				else:
					factor = 1
				total_grad[i] += calc_grad(j,i,factor = factor,embeddings= embeddings)
	return total_grad


