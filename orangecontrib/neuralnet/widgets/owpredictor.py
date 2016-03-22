import os
import sys
import numpy as np
from PyQt4 import QtCore, QtGui
from Orange.widgets import gui, widget, settings
import Orange.data

from keras.models import model_from_json

from orangecontrib.neuralnet.learner import NetLearner

class OWPredictor(widget.OWWidget):
    name = "Predictor"
    description = "Predicts on test data"
    icon = "icons/category.svg"

    # Takes the input as data and the learner from MLP MultiClass
    inputs = [("Data", Orange.data.Table, "set_data"),
              ("Neural Net", NetLearner, "set_learner")]
    outputs = [("TargetData", Orange.data.Table)]

    load_dir = settings.Setting("")

    def __init__(self):
        super().__init__()

        self.dataset = None
        self.model = None

        # tab_bar = QtGui.QTabBar(tabs)
        
        # the information box
        # self.tab = gui.tabWidget(self)
        self.infoBox = gui.widgetBox(self.controlArea, "Info", spacing=1)
        self.infoa = gui.widgetLabel(self.infoBox, 'No data on input yet, waiting to get something.')
        self.infob = gui.widgetLabel(self.infoBox, '')
        self.infoc = gui.widgetLabel(self.infoBox, 'No Learner Model selected')
        # self.tab.addTab(self,  )
        tabs = gui.tabWidget(self.controlArea)
        
        self.buttonBox = gui.widgetBox(self.controlArea, spacing=1)
        # Apply button
        self.apply_button = gui.button(self.buttonBox, self, "Apply", \
            callback = self.apply, default=False, autoDefault=False)
        # Load button: for loading the saved model
        self.load_button = gui.button(self.buttonBox, self, "Load", \
            callback = self.load, default=False, autoDefault=False)
        self.buttonBox.setDisabled(True)
        tab_1 = tabs.addTab(self.load_button, "Main")
        tab_2 = tabs.addTab(self.apply_button, "Description")

    def set_data(self, dataset):
        # the function to set the data locally and control some GUI aspects
        if dataset is not None:
            self.infoa.setText('%d instances in input data set' % len(dataset))
            self.infob.setText('%d attributes in input data set' % len(dataset.domain.attributes))
            self.dataset = dataset
            self.buttonBox.setDisabled(False)
            if(self.model == None):
                self.apply_button.setDisabled(True)
            else:
                self.apply_button.setDisabled(False)

        else:
            self.infoa.setText('No data on input yet, waiting to get something.')
            self.infob.setText('')
            self.buttonBox.setDisabled(True)

    def set_learner(self, model):
        # Setting the learner in case using the output from the 
        # MLP classifier
        self.infoc.setText('Using Output Learner Model')
        self.model = model
        self.apply_button.setDisabled(False)

    def apply(self):
        # The applt button functionality
        test_data = self.dataset.X.astype(float)
        # performing the fitting/predictions
        y_pred = self.model.predict_classes(test_data, verbose = 0)
        print(y_pred)
        print(len(y_pred.shape))
        y_pred = y_pred.reshape(len(y_pred), 1)
        # domain = Orange.data.Domain([Orange.data.Variable('Target')])
        # self.send("TargetData", Orange.data.Table.from_numpy(np.atleast_2d(y_pred.astype(int)), domain))
        self.send("TargetData", Orange.data.Table(np.atleast_2d(y_pred.astype(int))))

    def load(self):
        # responsible for loading the correct model
        self.infoc.setText('Using Loaded Learner Model')
        filename = QtGui.QFileDialog.getOpenFileName(
            self, "Open Model", self.load_dir, "*.json")
        self.model = model_from_json(open(filename).read())
        self.model.load_weights(filename[: -5] + 'h5')
        self.apply_button.setDisabled(False)