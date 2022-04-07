import numpy as np
import torch

from TestingDataset import TestingDataset

#load testing data set
testingSet = TestingDataset()
testingLoader = torch.utils.data.DataLoader(testingSet, batch_size=64, shuffle=True)

#load model
model = torch.load('model.pt')

#keep track of accuracy
correct = 0
total = 0

for images,labels in testingLoader:
  for i in range(len(labels)):
    img = images[i].view(1, 45**2)
    with torch.no_grad():
        logps = model(img)

    
    ps = torch.exp(logps)
    probab = list(ps.numpy()[0])
    prediction = probab.index(max(probab))
    actual = labels.numpy()[i]
    #check if prediction is correct
    if(prediction == actual):
      correct += 1
    
    total += 1

print("Total number of images:", total)
print("\nAccuracy: {}%".format(correct/total))