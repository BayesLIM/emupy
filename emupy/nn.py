"""
Neural Network Emulator
"""
import numpy as np
import torch
from collections.abc import Iterable

from .emulator import Emulator


class NNEmulator(Emulator, torch.nn.Module):
    """A subclass of the Emulator class for
    neural network based emulation, built
    on pytorch.nn
    """
    def __init__(self):
        """
        A subclass of the Emulator class for
        neural network based emulation, built
        on pytorch.nn
        """
        super(NNEmulator, self).__init__()
        super(Emulator, self).__init__()

    def set_layers(self, layers):
        """
        Set neural network layers

        Sets self.layers and self.layer{:d} for each layer

        Parameters
        ----------
        layers : list of torch.nn layers
            A list of instantiated torch.nn layers
            Ex. [torch.nn.Linear(2, 50), torch.nn.Linear(50, 3)]
       """
        self.layers = layers
        for i, layer in enumerate(layers):
            setattr(self, "layer{}".format(i + 1), layer)

    def set_activations(self, activations):
        """
        Set network layer activation functions

        Sets self.activations

        Parameters
        ----------
        activations : list of torch.nn activation functions
            A list of activation functions for each layer in layers
            Length must match length of layers. A None value results
            in no activation function for the associated layer.
            Ex. [torch.nn.ReLU(), None]
            To provide multiple operations for a given layer, feed
            the activation as a list of operatations in the desired
            order.
            Ex. [[torch.nn.ReLU(), torch.nn.Dropout()], None]
        """
        self.activations = activations
 
    def set_inits(self, inits):
        """
        Set layer weight initializations

        Parameters
        ----------
        inits : list of weight init functions
            List must match length of layers. Each element
            in inits is the weight initialization of the
            associated layer in self.layers
        """
        if not isinstance(inits, Iterable):
            inits = [inits] * len(self.layers)
        for i, init in enumerate(inits):
            init(self.layers[i].weight)

    def forward(self, X):
        """
        Forward propagate network

        Parameters
        ----------
        X : torch.Tensor, (Nsamples, Nfeatures)
            Feature values of data vector

        Returns
        -------
        array_like
            Output of network after evaluating layers and activations
        """
        for i, (layer, activ) in enumerate(zip(self.layers, self.activations)):
            # pass through layer
            X = layer(X)
            if activ is not None:
                # if activ is not an iterable, shallow wrap it
                if not isinstance(activ, Iterable):
                    activ = [activ]
                # iterate over activations for this layer, e.g. dropout(ReLU(x))
                for act in activ:
                    X = act(X)

        return X

    def train(self, X, y, loss_fn=None, optim=None, Nepochs=100, **kwargs):
        """
        Train neural network via batch backpropagation

        Sets self.loss_fn, self.optim, self.loss

        Parameters
        ----------
        X : torch.Tensor, (Nsamples, Nfeatures)
            Feature values of data vector for training

        y : torch.Tensor, (Nsamples, Ntargets)
            Target values of data vector for training

        loss_fn : pytorch _Loss object
            A loss function callable from torch.nn
            Default: MSELoss(reduction='mean')

        optim : pytorch Optimizer object
            An optimization object from torch.optim
            Default: Rprop(..., lr=0.01)

        Nepochs : int
            Number of full epoch cycles to complete during training

        kwargs : additional keyword arguments for loss_fn
        """
        if loss_fn is None:
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        if optim is None:
            self.optim = torch.optim.Rprop(self.parameters(), lr=0.01)

        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X)
        if not isinstance(y, torch.Tensor):
            y = torch.as_tensor(y)

        def closure():
            self.optim.zero_grad()
            outputs = self.forward(X)
            loss = self.loss_fn(outputs, y, **kwargs)
            loss.backward()
            return loss

        self.loss = []
        for i in range(Nepochs):
            # forward pass
            self.optim.zero_grad()
            outputs = self.forward(X)

            # backprop
            loss = self.loss_fn(outputs, y, **kwargs)
            loss.backward()
            self.optim.step(closure)
            self.loss.append(loss)

    def predict(self, X, unscale=False, reproject=False):
        """
        Prediction method after training the network.
        Shallow wrap around self.forward()

        Parameters
        ----------
        X : torch.Tensor, (Nsamples, Nfeatures)
            Feature values of data vector for training

        Returns
        -------
        array_like
            Output of network after evaluating layers and activations
        """
        pred = self.forward(torch.as_tensor(X)).detach().numpy()
        # unscale data if scaled
        if unscale:
            pred = self.unscale_data(pred)

        # reproject if KL basis
        if reproject:
            pred = self.klt_reproject(pred)

        return pred

