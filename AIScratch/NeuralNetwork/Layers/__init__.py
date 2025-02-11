from enum import Enum
from AIScratch.NeuralNetwork.Layers.layer import Layer
from AIScratch.NeuralNetwork.Layers.spatial_layer import SpatialLayer
from AIScratch.NeuralNetwork.Layers.dense import DenseLayer
from AIScratch.NeuralNetwork.Layers.dropout import DropoutLayer
from AIScratch.NeuralNetwork.Layers.pooling import PoolingLayer, PoolingType


layer_set = {
    "dense": DenseLayer,
    "dropout": DropoutLayer
}

