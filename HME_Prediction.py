from tkinter import *
from matplotlib import pyplot as plt 
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, 
NavigationToolbar2Tk)
import torch
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog as fd

global image
global window
global model_select


# Function to remove white space above and below an HME
def adjust_height(img):
    upper = 0 #upper boundary
    lower = len(img[0]) -1 #lower boundary

    #find upper boundary
    while not any(i < 0.15 for i in img[0][upper]):
        upper+=1

    #find lower boundary
    while not any(i < 0.15 for i in img[0][lower]):
        lower-=1

    #return 'compressed' tensor
    return torch.index_select(img, 1, torch.tensor(list(range(upper, lower+1))))

def extract_tokens(img):
    left = 0 #beginning of token
    right = 0 #end of token
    isStarted = False #If beginning of token has been found
    tokens = [] #array of tokens

    #iterate through all columns of the image
    for col in range(len(img[0][0])):
        column = [i[col] for i in img[0]] #get array of first index of all tensors

        #check if at least on black pixel is present in column
        if any(i < 0.15 for i in column) and not isStarted:
            left = col
            isStarted = True
        #check for entirely white column
        elif not any(i < 0.15 for i in column) and isStarted:
            right = col-1
            isStarted = False
            #create subset of image which is just one token
            token = torch.index_select(img, 2, torch.tensor(list(range(left, right+1))))
            token = adjust_height(token)

            #make image square
            if len(token[0]) < len(token[0][0]):
                pad = int((len(token[0][0]) - len(token[0])) /2)
                token = F.pad(token, (0,0, pad, pad), "constant", 1)
            elif len(token[0]) > len(token[0][0]):
                pad = int((len(token[0]) - len(token[0][0])) /2)
                token = F.pad(token, (pad, pad, 0, 0), "constant", 1)

            tokens.append(token)

    return tokens

def PredictHME(dir = 'Test_HMEs/2.jpg'):

    global image
    global model_select
    global window

    image = Image.open(dir) #load image

    #tkinter window
    window = Tk() 
    window.protocol("WM_DELETE_WINDOW", quit)
    window.title('HME Recognition')
    window.geometry("1000x1000")

    model_options = [
    "Trained_models/model_NLL.pt",
    "Trained_models/model_NLL_No-Distortion.pt",
    "Trained_models/model_cross-entropy.pt"
    ]

    model_select = StringVar(window)
    model_select.set(model_options[0]) # default value

    model_dropdown = OptionMenu(window, model_select, *model_options)
    model_dropdown.pack()

    #Predict button
    plot_button = Button(master = window, command = plot, height = 2,  width = 10, text = "Predict")
    plot_button.pack()

    def select_file():
        filetypes = ([("Image File",'.jpg')])

        filename = fd.askopenfilename(title='Open a jpg image', initialdir='/', filetypes=filetypes)

        global image
        image = Image.open(filename)


    #open button
    open_button = ttk.Button(window, text='Open a jpg image', command=select_file)
    open_button.pack(expand=False)
    
    # run the gui
    window.mainloop()
  
#plot interface
def plot():
    global image
    global model_select
    global window

    #convert image to grayscale and transform to a Tensor 
    transform = transforms.Compose([transforms.Grayscale(num_output_channels=1) ,transforms.ToTensor()])
    img_tensor = transform(image)
    torch.set_printoptions(threshold=10_000)

    #remove vertical white space
    img_tensor = adjust_height(img_tensor)

    #extract all tokens
    tokens = extract_tokens(img_tensor)
    index = 0

    model = torch.load(model_select.get())

    answers = [0,1,2,3,4,5,6,7,8,9,'-','+','=','x','÷']
    height = 2* len(tokens) +3 #number of vertical subplots
  
    fig = plt.figure(figsize=(6,9))
    
    title = plt.subplot2grid((height,2), (0, 0), rowspan=2, colspan=2, xticklabels=[], yticklabels=[], xticks=[], yticks=[], fc="red",)
    title.imshow(image)
    title.set_title('Original Image')

    prediction = ''

    index = 2
    #generate a subplot for each token
    for t in tokens:

        #apply transformations
        transform = transforms.Compose([transforms.Resize((45,45)),transforms.ToPILImage(), transforms.Grayscale(num_output_channels=1), transforms.RandomInvert(1) ,transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
        t = transform(t)
        #add graph and image
        img = t.view(1,len(t[0]),len(t[0][0]))
        aximg = plt.subplot2grid((height, 2), (index, 0), rowspan=2, colspan=1, xticklabels=[], yticklabels=[], xticks=[], yticks=[])
        axgraph = plt.subplot2grid((height, 2), (index, 1), rowspan=2, colspan=1)
        aximg.imshow(img.numpy().squeeze(), cmap='gray_r')

        #predict digit
        digit = t.view(1, 45**2)
        with torch.no_grad():
            logps = model(digit)
        ps = torch.exp(logps)
        probab = list(ps.numpy()[0])

        #get top 3 predictions
        top3 = (np.argsort(probab)[-3:])
        ps = ps.data.numpy().squeeze()

        #create prediction graph
        c = ['blue'] * 3
        bars = [ps[top3[2]], ps[top3[1]], ps[top3[0]]]
        c[0] = 'red'
        axgraph.barh(np.arange(3), bars, color=c)
        axgraph.set_xticks([0,0.5,1])
        symbols = [0,1,2,3,4,5,6,7,8,9,'-','+','=','x','÷']
        axgraph.set_yticks(np.arange(3),[symbols[top3[2]], symbols[top3[1]], symbols[top3[0]]] )

        #add predicted digit to HME prediction string
        prediction += str(symbols[top3[2]])

        index+=2

    #add HME prediction at the bottom of the graph
    axpred = plt.subplot2grid((height, 2), (index, 0), rowspan=1, colspan=2, xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    axpred.text(0.5, 0.5, prediction, horizontalalignment='center', verticalalignment='center')
    axpred.set_title('Prediction')
    title.set_label(prediction)

    #add some white space between subplots
    plt.subplots_adjust(wspace=1, hspace=1)

    #add plot to tkinter window
    canvas = FigureCanvasTkAgg(fig, master = window) 
    canvas.draw()
    canvas.get_tk_widget().pack()
    toolbar = NavigationToolbar2Tk(canvas, window)
    toolbar.update()
    canvas.get_tk_widget().pack()
  
def quit():
    global window
    window.quit()
    window.destroy()