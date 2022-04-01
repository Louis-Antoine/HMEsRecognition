import numpy as np
import torch
import torchvision
import matplotlib.pyplot as plt
from time import time
from torchvision import datasets, transforms
from torch import nn, optim
from NumDataset import NumbersDataset

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainingSet = NumbersDataset()
trainingSet.tranform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
valuesSet = NumbersDataset() #test set (same as training set for now)
trainingLoader = torch.utils.data.DataLoader(trainingSet, batch_size=64, shuffle=True)
valueLoader = torch.utils.data.DataLoader(valuesSet, batch_size=64, shuffle=True)
dataiter = iter(trainingLoader)
images, labels = dataiter.next()
# print(images.shape)
# print(labels.shape)
figure = plt.figure()
num_of_images = 60
for index in range(1, num_of_images + 1):
    plt.subplot(6, 10, index)
    plt.axis('off')
    plt.imshow(images[index].numpy().squeeze(), cmap='gray_r')
plt.show()

input_size = 45**2
hidden_sizes = [128, 64]
output_size = 14

model = nn.Sequential(nn.Linear(input_size, hidden_sizes[0]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                      nn.ReLU(),
                      nn.Linear(hidden_sizes[1], output_size),
                      nn.LogSoftmax(dim=1))
# print(model)

criterion = nn.NLLLoss()
images, labels = next(iter(trainingLoader))
# print(images.shape)
# print(labels.shape)
images = images.view(images.shape[0], -1)
# print(images.shape[0])
# print(labels)

logps = model(images) #log probabilities
loss = criterion(logps, labels) #calculate the NLL loss

print('Before backward pass: \n', model[0].weight.grad)
loss.backward()
print('After backward pass: \n', model[0].weight.grad)
optimizer = optim.SGD(model.parameters(), lr=0.003, momentum=0.9)
time0 = time()
running_loss = 0
for images, labels in trainingLoader:
        # Flatten MNIST images into a 784 long vector
        images = images.view(images.shape[0], -1)
    
        # Training pass
        optimizer.zero_grad()
        
        output = model(images)
        loss = criterion(output, labels)
        
        #This is where the model learns by backpropagating
        loss.backward()
        
        #And optimizes its weights here
        optimizer.step()
        
        running_loss += loss.item()
else:
    print("Epoch - Training loss: {}".format(running_loss/len(trainingLoader)))
print("\nTraining Time (in minutes) =",(time()-time0)/60)