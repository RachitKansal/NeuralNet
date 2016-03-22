# NeuralNet
This is a prototype Add-On for Orange3 which provides a basic Feedforward Neural Network.

# Installation
Use ``python setup.py install`` to install the plugin.

# Requirements
1. Orange3
2. Numpy
3. theano
4. keras
5. cython
6. h5py

for installation use ``pip``, but if you get error related to the version use ``easy_install``

# Tutorial
I have uploaded the mnist dataset as csv file here along with some snapshots (for a small tutorial). This dataset is huge and may not work in windows machine as orange crashes. In ubuntu machine, in your development version be patient while the dataset loads. 

Because this is just a protoype, it is not complete but rather used to show how the complete neural network add-on would look like and it shows the usage of QThreads and keras callbacks.

Dataset Links:

test data: https://drive.google.com/file/d/0B-NVQRp0HfUrQXJVeGtydXg0QXc/view?usp=sharing

train data: https://drive.google.com/file/d/0B-NVQRp0HfUrMEJQM3JUbEdBYzA/view?usp=sharing

# Snapshots

![Layout](/Snapshots/snap1.png?raw=true "snap1")

![Trainer](/Snapshots/snap2.png?raw=true "snap2")

![Predictor](/Snapshots/snap3.png?raw=true "snap3")

![Saving/Loading](/Snapshots/snap4.png?raw=true "snap4")
