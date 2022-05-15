import numpy as np
import matplotlib.pyplot as plt
# import scipy
import trnn_utils as U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
torch.manual_seed(0)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# model train
def model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y, learning_rate):

	x_train = torch.Tensor(train_data_x)
	y_train = torch.Tensor(train_data_y)
	train_dataset = TensorDataset(x_train, y_train)
	train_loader = DataLoader(train_dataset, batch_size = 20, shuffle = True)
	data = torch.Tensor(all_data)
	word_embed = torch.Tensor(word_embed)

	net = U.Text_Encoder(data, word_embed, learning_rate, device)
	net.to(device)
	criterion = nn.CrossEntropyLoss().to(net.device)
	accuracy = []
	epochs = []
	for epoch in range(100):
		train_loss = 0

		net.train()
		for id_batch, y in train_loader:
			id_batch = id_batch.clone().detach().long().to(net.device)
			y = y.clone().detach().long().to(net.device)
			net.opt.zero_grad()
			y_pred = net(id_batch)
			loss = criterion(y_pred, y)
			loss.backward()
			net.opt.step()
			train_loss += loss.item()
		test_accuracy = model_test(test_data_x, test_data_y, net, epoch)
		accuracy.append(test_accuracy)
		epochs.append(epoch)
		if epoch%10 == 0:
			print("Epoch: {0:d} \tTest Accuracy: {1:.3f}".format(epoch, test_accuracy))
	return accuracy, epochs

# model test: can be called directly in model_train 
def model_test(test_data_x, test_data_y, net, epoch_num):
	x_test = torch.Tensor(test_data_x)
	y_test = torch.Tensor(test_data_y)
	dataset = TensorDataset(x_test, y_test)
	testLoader = DataLoader(dataset, batch_size = 20, shuffle = True)
	net.eval()
	with torch.no_grad():
		correct = 0
		total = 0
		for batch_id, y in testLoader:
			batch_id = batch_id.clone().detach().long().to(net.device)
			y = y.clone().detach().long().to(net.device)
			y_pred = net(batch_id)
			y_pred = y_pred.argmax(axis = 1)[:,None].squeeze().long()
			correct += (y_pred == y).sum().item()
			total += y.size(0)

	return correct/total

if __name__ == '__main__':
	# load datasets
	input_data = U.input_data()

	all_data, train_data, test_data = input_data.load_text_data()
	train_data_x = train_data[0] # map content by id
	train_data_y = train_data[1]
	test_data_x = test_data[0] # map content by id
	test_data_y = test_data[1]

	word_embed = input_data.load_word_embed()

	learning_rate = 0.0025

	# model train (model test function can be called directly in model_train)
	accuracy, epochs = model_train(all_data, word_embed, train_data_x, train_data_y, test_data_x, test_data_y, learning_rate)
	plt.plot(epochs, accuracy, label = "last hidden")
	plt.xlabel('epochs')
	plt.ylabel('accuracy')
	plt.legend()







