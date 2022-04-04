import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from NumDataset import NumbersDataset


trainingSet = NumbersDataset()
testingSet = NumbersDataset() #test set (same as training set for now)
trainingLoader = torch.utils.data.DataLoader(trainingSet, batch_size=64, shuffle=True)
testingLoader = torch.utils.data.DataLoader(testingSet, batch_size=64, shuffle=True)

input_size = 45**2 #45**2 neurons for first layer since images are 45x45 
hidden_sizes = [128, 64] #hidden layers
output_size = 15

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()
#create iterator for images & labels
images, labels = next(iter(trainingLoader))
images = images.view(images.shape[0], -1)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

loss.backward()
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time() #track time taken
running_loss = 0
for images, labels in trainingLoader:
        #flatten the image
        images = images.view(images.shape[0], -1)
    
        #training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #backpropagating
        loss.backward()
        
        #Optimize the weights
        optimizer.step()
        
        running_loss += loss.item()
else:
    print("Epoch - Training loss: {}".format(running_loss/len(trainingLoader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)

torch.save(model, 'model1.pt')