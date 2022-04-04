""" We don't really need that code """


from matplotlib import pyplot as plt
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms

def view_classify(img, ps):
    ''' Function for viewing an image and it's predicted classes.
    '''
    ps = ps.data.numpy().squeeze()
    print(np.where(ps == max(ps))[0])
    c = ['blue'] * 15
    


    c[np.where(ps == max(ps))[0][0]] = 'red'
    print(max(ps))
    print(c)

    fig, (ax1, ax2) = plt.subplots(figsize=(6,9), ncols=2)
    ax1.imshow(img.resize_(1, 45, 45).numpy().squeeze(), cmap='gray_r')
    ax1.axis('off')
    ax2.bar(np.arange(15), ps, color=c)
    #ax2.set_aspect(max(ps))
    ax2.set_xticks(np.arange(15))
    ax2.set_xticklabels([0,1,2,3,4,5,6,7,8,9,'-','+','=','x','÷'])
    ax2.set_title('Class Probability')
    #ax2.set_ylim(0, max(ps) + 0.1)
    #plt.tight_layout()
    plt.show()

# Read a PIL image
image = Image.open('original_data/7/7_465.jpg')
  
# Define a transform to convert PIL 
# image to a Torch tensor
transform = transforms.Compose([transforms.Grayscale(num_output_channels=1), transforms.RandomInvert(1) ,transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
  
# transform = transforms.PILToTensor()
# Convert the PIL image to Torch tensor
img_tensor = transform(image)
  
# print the converted Torch tensor
print(img_tensor)


img = img_tensor.view(1, 45**2)
model = torch.load('model1.pt')

with torch.no_grad():
    logps = model(img)

# Output of the network are log-probabilities, need to take exponential for probabilities
ps = torch.exp(logps)
probab = list(ps.numpy()[0])
print("Predicted Digit =", probab.index(max(probab)))
view_classify(img.view(1, 45, 45), ps)