from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def adjust_height(img):
    upper = 0
    lower = len(img[0]) -1

    #find upper boundary
    while not any(i < 0.15 for i in img[0][upper]):
        upper+=1

    #find lower boundary
    while not any(i < 0.15 for i in img[0][lower]):
        lower-=1

    return torch.index_select(img, 1, torch.tensor(list(range(upper, lower+1))))

def extract_tokens(img):
    left = 0
    right =0
    isStarted = False
    tokens = []

    for col in range(len(img[0][0])):
        column = [i[col] for i in img[0]]

        if any(i < 0.15 for i in column) and not isStarted:
            left = col
            isStarted = True
        elif not any(i < 0.15 for i in column) and isStarted:
            right = col-1
            isStarted = False
            token = torch.index_select(img, 2, torch.tensor(list(range(left, right+1))))
            print(left, right)
            print(token)
            tokens.append(adjust_height(token))

    return tokens


image = Image.open('1.jpg')

#transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.RandomInvert(1) ,transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1) ,transforms.ToTensor()])

img_tensor = transform(image)
torch.set_printoptions(threshold=10_000)
print([float(i[0]) for i in img_tensor[0]])

img_tensor = adjust_height(img_tensor)


#img = img_tensor.view(1,len(img_tensor[0]),len(img_tensor[0][0]))

tokens = extract_tokens(img_tensor)



fig, axs = plt.subplots(figsize=(6,9), ncols=len(tokens))
index = 0
for t in tokens:
    img = t.view(1,len(t[0]),len(t[0][0]))
    axs[index].imshow(img.numpy().squeeze(), cmap='gray')
    index+=1


plt.show()