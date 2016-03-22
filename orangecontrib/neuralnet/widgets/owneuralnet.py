import os
import sys
import numpy as np
from PyQt4 import QtCore, QtGui
from Orange.widgets import gui, widget, settings
import Orange.data

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense
from keras.optimizers import SGD
from keras import callbacks

from orangecontrib.neuralnet.utils import normalize
from orangecontrib.neuralnet.learner import NetLearner

class MyThread(QtCore.QThread, callbacks.Callback):
    """ A seperate class to provide Multi-threading support during the
    training time so as to not let the tool freeze and also it helps to 
    update the progress bar which can only be done using QThread and not
    the ususal python threads"""
    trigger = QtCore.pyqtSignal(int, float)

    def __init__(self, model, progress_bar, train_data, train_target, \
        iteration_val, validation, batch, parent = None):
        # Just Initializing the parameters
        self.model = model
        self.progress_bar = progress_bar
        self.train_data = train_data
        self.train_target = train_target
        self.iteration_val = iteration_val
        self.validation = validation
        self.batch = batch
        self.value = 0
        super(MyThread, self).__init__(parent)

    def on_epoch_end(self, a, b):
        # At the end of every iteration, this function is called which
        # updates the progress bar
        self.value = self.value + 1
        self.trigger.emit(self.value, b["val_acc"])

    def on_train_end(self, a):
        # Called at the end of training to reinitialize the 
        # progress bar to value 0
        self.value = self.value + 1
        self.trigger.emit(0, -1)
    
    def run(self):
        # The actual method which performs the fitting
        self.model.fit(self.train_data, self.train_target, \
            nb_epoch = self.iteration_val, batch_size = self.batch, \
            verbose = 0, callbacks=[self], validation_split = self.validation, \
            show_accuracy = True)

class OWNeuralNet(widget.OWWidget):
    # Widget needs a name, or it is considered an abstract widget
    # and not shown in the menu.
    name = "MLP Multi-Class"
    description = "Generates a MLP model from data."
    icon = "icons/mywidget.svg"

    # Takes the Orange data Table as input and produces a learner output
    # which is sccepted by the predictor widget
    inputs = [("Data", Orange.data.Table, "set_data")]
    outputs = [("Predictor", NetLearner)]

    # setting the values for various combo-boxes and the checkbox
    # stores the number of layers
    num_layers = settings.Setting(0)
    
    # these values store the no. of neurons in each layer
    neuron_lyr1 = settings.Setting(0)
    neuron_lyr2 = settings.Setting(0)
    neuron_lyr3 = settings.Setting(0)
    neuron_lyr4 = settings.Setting(0)
    # These store the index of selected activation function in comboboxes
    activation_lyr1 = settings.Setting(0)
    activation_lyr2 = settings.Setting(0)
    activation_lyr3 = settings.Setting(0)
    activation_lyr4 = settings.Setting(0)
    
    # The different activation functions available
    activations = ['sigmoid', 'tanh', 'linear']
    # The different kind of loss functions available to the user
    lossfunctions = ['mean_squared_error', 'mean_squared_logarithmic_error', \
    'categorical_crossentropy', 'poisson']
    activation_out = settings.Setting(0)
    loss_function = settings.Setting(0)
    # Stores the selected learning rate
    learning_rate = settings.Setting(0)
    # stores the selected number of iterations to perform
    iteration_val = settings.Setting(0)
    # stores whether to perform Normalization on data or not
    checkbox_val = settings.Setting(0)
    # the percentage of data to be used for cross validation
    validation = settings.Setting(0)
    # the batchsize selected for stochastic gradient descent
    batchsize = settings.Setting(0)

    save_dir = settings.Setting("")

    def __init__(self):
        super().__init__()

        self.dataset = None

        # The Information Box
        self.infoBox = gui.widgetBox(self.controlArea, "Info", spacing=1)
        self.infoa = gui.widgetLabel(self.infoBox, 'No data on input yet, waiting to get something.')
        self.infob = gui.widgetLabel(self.infoBox, '')
        self.infoc = gui.widgetLabel(self.infoBox, '')
        gui.radioButtons(self.infoBox, self, 'num_layers', btnLabels=('1', '2', '3', '4'), \
            label = 'Number of Layers: ', orientation='horizontal', callback = self.updateLayer)
        
        # The Layer Box: provides all the options for all 4 layers
        self.layerBox = gui.widgetBox(self.controlArea, "Layer Options", spacing=1, \
            orientation='horizontal')
        
        self.layerBox1 = gui.widgetBox(self.layerBox, spacing=1)
        self.layerBox2 = gui.widgetBox(self.layerBox, spacing=1)

        self.box = []
        for i in range(4):
            if(i<2):
                self.box.append(gui.widgetBox(self.layerBox1, 'Layer ' + str(i+1)))
            else:
                self.box.append(gui.widgetBox(self.layerBox2, 'Layer ' + str(i+1)))
            gui.spin(self.box[i], self, 'neuron_lyr' + str(i+1), minv=1, maxv=50, step=1, \
                label='Neurons in hidden Layer ' + str(i+1) + ': ')
            combo = gui.comboBox(self.box[i], self, 'activation_lyr' + str(i+1), \
                label = 'Activation of Layer ' + str(i+1) + ':', \
                items=('sigmoid', 'tanh', 'linear'))
            self.box[i].setDisabled(True)
        self.layerBox.setDisabled(True)
        
        # The Options Box: gives some general options to configure the network
        self.optionsBox = gui.widgetBox(self.controlArea, "General Options", \
            orientation='vertical', spacing=1)
        self.generalBox = gui.widgetBox(self.optionsBox , orientation='horizontal', spacing=1)
        self.optionsBox1 = gui.widgetBox(self.generalBox, spacing=1)
        self.optionsBox2 = gui.widgetBox(self.generalBox, spacing=1)
        
        gui.spin(self.optionsBox1, self, 'learning_rate', minv=0.01, maxv=1.0, \
            step=0.01, label='Learning Rate: ', spinType=float)
        combo = gui.comboBox(self.optionsBox1, self, 'activation_out', \
            label = 'Ouput Layer Activation: ', orientation='horizontal', items=('sigmoid', 'tanh', 'linear'))
        gui.spin(self.optionsBox1, self, 'validation', minv=0.01, maxv=0.2, \
            step=0.01, label='Validation Split: ', spinType=float)
        gui.spin(self.optionsBox2, self, 'iteration_val', minv=1, maxv=100, \
            step=1, label='Iterations: ')
        combo = gui.comboBox(self.optionsBox2, self, 'loss_function', label = 'Loss Function: ', \
            orientation='horizontal', items=('Mean Squared Error', 'Mean Squared Log Error', \
                'Categorical Cross-Entropy', 'Poisson'))
        self.batch_spin = gui.spin(self.optionsBox2, self, 'batchsize', minv=1, maxv=300, \
            step=1, label='Batch Size: ')

        # The GUI for the Progress Bar
        self.progress = QtGui.QProgressBar(self.optionsBox)
        gui.miscellanea(self.progress, None, self.optionsBox)

        # The GUI for checkbox of normalizing the data
        gui.checkBox(self.optionsBox, self, 'checkbox_val',label='Normalize Data')
        
        self.buttonBox = gui.widgetBox(self.optionsBox, spacing=1)
        
        # The GUI for the apply button
        self.apply_button = gui.button(self.buttonBox, self, "Apply", callback = self.apply, \
            default=False, autoDefault=False)

        # The GUI for the save button
        self.save_button = gui.button(self.buttonBox, self, "Save", callback = self.save, \
            default=False, autoDefault=False)
        self.optionsBox.setDisabled(True)

    def updateLayer(self):
        """ This function is used to control the GUI of the widget, 
        in particular the layerBox for the widget"""
        if self.num_layers == 0:
            self.box[0].setDisabled(False)
            for i in range(1,4):
                self.box[i].setDisabled(True)
        elif self.num_layers == 1:
            self.box[0].setDisabled(False)
            self.box[1].setDisabled(False)
            for i in range(2,4):
                self.box[i].setDisabled(True)
        elif self.num_layers == 2:
            self.box[0].setDisabled(False)
            self.box[1].setDisabled(False)
            self.box[2].setDisabled(False)
            self.box[3].setDisabled(True)
        else:
            self.box[0].setDisabled(False)
            self.box[1].setDisabled(False)
            self.box[2].setDisabled(False)
            self.box[3].setDisabled(False)

    def set_data(self, dataset):
        """ This function is called whevever the data is input into the widget,
        It also controls some GUI aspects of the widget`"""
        if dataset is not None:
            self.infoa.setText('%d instances in input data set' % len(dataset))
            self.infob.setText('%d attributes in input data set' % len(dataset.domain.attributes))
            # Limited the batch size between 0.005 to 0.025, in
            # order tk=o make training fats and also accurate
            if(len(dataset) >= 200):
                self.batchsize = int(0.005 * len(dataset))
                self.batch_spin.setMinimum(int(0.005 * len(dataset)))
                self.batch_spin.setMaximum(int(0.025 * len(dataset)))
            else:
                # here the dataset is to small, hence fixed the
                # batch size programmatically
                self.batchsize = 1
                self.batch_spin.setMinimum(1)
                self.batch_spin.setMaximum(10)
            self.optionsBox.setDisabled(False)
            self.layerBox.setDisabled(False)
            self.updateLayer()
            self.dataset = dataset
            self.save_button.setDisabled(True)

        else:
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.infob.setText('')
            self.optionsBox.setDisabled(True)
            self.layerBox.setDisabled(True)
            self.dataset = None

    def apply(self):
        # The action for apply button
        self.buttonBox.setDisabled(True)
        self.save_button.setDisabled(False)

        # Taking care of normalizatio of data
        if(self.checkbox_val == True):
            dataX = normalize(self.dataset.X.astype(float))
        else:
            dataX = self.dataset.X.astype(float)
        
        # Setting the range of progress bar to number of iteration
        # and hence makes iot accurate
        self.progress.setRange(0, self.iteration_val)

        train_data = dataX.astype(float)
        
        # Preparing the target for multilabel classification
        target = self.dataset.Y.astype(int)
        train_target = []
        unique_target = np.unique(target)
        for i in range(0, len(target)):
            dummy = []
            k = 0
            for j in unique_target:
                if j == target[i]:
                    dummy.insert(k, 1)
                else:
                    dummy.insert(k, 0)
                k = k + 1
            train_target.insert(i, dummy)
        train_target = np.array(train_target)

        # Building the model for a feedforard network
        self.model = Sequential()
        self.model.add(Dense(input_dim = len(train_data[0]), output_dim = self.neuron_lyr1, \
            init = 'uniform', activation = self.activations[self.activation_lyr1]))
        if self.num_layers == 0:
            self.model.add(Dense(output_dim = len(train_target[0]), init = 'uniform', \
                activation = self.activations[self.activation_out]))
        elif self.num_layers == 1:
            self.model.add(Dense(output_dim = self.neuron_lyr2, init = 'uniform', \
                activation = self.activations[self.activation_lyr2]))
            self.model.add(Dense(output_dim = len(train_target[0]), init = 'uniform', \
                activation = self.activations[self.activation_out]))
        else:
            self.model.add(Dense(output_dim = self.neuron_lyr2, init = 'uniform', \
                activation = self.activations[self.activation_lyr2]))
            self.model.add(Dense(output_dim = self.neuron_lyr3, init = 'uniform', \
                activation = self.activations[self.activation_lyr3]))
            self.model.add(Dense(output_dim = len(train_target[0]), init = 'uniform', \
                activation = self.activations[self.activation_out]))

        # the stochastic gradient descent optimizer
        sgd = SGD(lr = self.learning_rate)

        self.model.compile(loss=self.lossfunctions[self.loss_function], optimizer=sgd)

        # Using multi-threading to train the model
        self.thread = MyThread(self.model, self.progress, train_data, train_target, \
            self.iteration_val, self.validation, self.batchsize)
        self.thread.trigger.connect(self.update_progress)
        self.thread.start()

        # Sending the output of the model
        self.send("Predictor", self.model)

    def save(self):
        # The functionality of the save button for the model
        filename = QtGui.QFileDialog.getSaveFileName(
            self, "Save Model", self.save_dir)
        self.save_dir = os.path.dirname(filename)
        json_string = self.model.to_json()
        open(filename + '.json', 'w').write(json_string)
        self.model.save_weights( filename + 'h5')

    def update_progress(self, value, accuracy):
        # Updates the progress bar for every iteration during training
        self.progress.setValue(value)
        if(value == 0):
            self.buttonBox.setDisabled(False)
        if(accuracy != -1):
            self.infoc.setText("Validation Accuracy: " + str(round(accuracy, 3)))