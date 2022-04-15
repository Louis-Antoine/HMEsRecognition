import numpy as np
import torch
from sklearn import metrics
from TestingDataset import TestingDataset

def TestMetrics(m = 'model.pt'):
  print('Testing metrics on:', m)
  
  #load testing data set
  testingSet = TestingDataset()
  testingLoader = torch.utils.data.DataLoader(testingSet, batch_size=64, shuffle=True)

  #load model
  model = torch.load(m)

  #keep track of accuracy
  correct = 0
  total = 0
  pred = []
  truth = []
  answers = [0,1,2,3,4,5,6,7,8,9,'-','+','=','x','÷']

  for images,labels in testingLoader:
    for i in range(len(labels)):
      img = images[i].view(1, 45**2)
      with torch.no_grad():
          logps = model(img)

      
      ps = torch.exp(logps)
      probab = list(ps.numpy()[0])
      prediction = probab.index(max(probab))
      actual = labels.numpy()[i]

      pred.append(str(answers[prediction]))
      truth.append(str(answers[actual]))

      #check if prediction is correct
      if(prediction == actual):
        correct += 1
      
      total += 1

  #print("Total number of images:", total)
  #print("\nAccuracy: {}%".format(100*(correct/total)))

  # Print the confusion matrix
  #print(metrics.confusion_matrix(truth, pred))

  # Print the precision and recall, among other metrics
  print(metrics.classification_report(truth, pred, digits=3))