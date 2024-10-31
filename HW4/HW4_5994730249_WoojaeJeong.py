# Import library
import h5py
import numpy as np
import matplotlib.pyplot as plt 
import json

# Load pre-trained network

# File name
Data_fName = './mnist_network_params.hdf5'

# Read data
with h5py.File(Data_fName, 'r+') as df:
    W1 = df['W1'][:]
    W2 = df['W2'][:]
    W3 = df['W3'][:]
    b1 = df['b1'][:]
    b2 = df['b2'][:]
    b3 = df['b3'][:]

# Verify the dimension of each parameters

print('Dimension of W1 =', W1.shape)
print('Dimension of b1 =', b1.shape)
print('Dimension of W2 =', W2.shape)
print('Dimension of b2 =', b2.shape)
print('Dimension of W3 =', W3.shape)
print('Dimension of b3 =', b3.shape)

    # File name
Data_fName = './mnist_testdata.hdf5'

# Read data
with h5py.File(Data_fName, 'r+') as df:
    xdata = df['xdata'][:]
    ydata = df['ydata'][:]

# softmax function
def softmax(data):
    z = np.exp(data)
    return z / z.sum(axis = 0)

# ReLU function
def ReLU(data):
    f = np.maximum(0,data)
    return f

output = np.zeros((len(xdata),10))

for i in range(len(xdata)):
    # First layer
    a1 = ReLU(np.dot(W1,xdata[i])+b1)

    # Second layer
    a2 = ReLU(np.dot(W2,a1)+b2)

    # Output layer
    output[i,:] = softmax(np.dot(W3,a2)+b3)

# data
data = []

for i in range(10000):
    data += [{"index": int(i), "activations": output[i,:].tolist(), "classification": int(np.argmax(output[i,:]))}]

# Write to .json
with open("result.json", "w") as f:
    f.write(json.dumps(data))

correct = (np.argmax(output, axis=1) == np.argmax(ydata, axis=1)).sum()
print("Number of correctly classified images:", correct)