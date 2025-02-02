import numpy as np
from AIScratch.NeuralNetwork.Perceptron import Perceptron
from AIScratch.NeuralNetwork.ActivationFunctions import ActivationFunction
from AIScratch.NeuralNetwork.ErrorFunctions import ErrorFunction
from AIScratch.NeuralNetwork.Layers import Layer

class MLP():
    def __init__(self, input_number : int, layers : list[Layer], error_function : ErrorFunction):
        self.input_number = input_number
        self.layers : list[Layer] = layers
        self.error_function = error_function
        self.__initialize()

    def __initialize(self):
        prev_size = self.input_number
        for layer in self.layers:
            layer._initialize(prev_size)
            prev_size = layer.n_out

    def forward(self, inputs):
        outputs = np.array(inputs)
        for layer in self.layers:
            outputs = layer.forward(outputs)
        return outputs
    
    def backward(self, inputs, expected_outputs):
        _inputs = np.array(inputs)
        outputs = self.forward(_inputs) # all neurons stores the inputs and all layers store activations
        errors = self.error_function.backward(expected_outputs, outputs)
        for p in reversed(range(len(self.layers))): # each layer should compute gradient for itself and error for next
            # current layer computation
            layer = self.layers[p] # layer p
            gradients = np.array([layer.activation_function.backward(z) for z in layer.last_sums]) # f'p(Sp,j)
            layer.learn(errors, gradients) # a(p - 1) already in each neuron
            # next layer computation
            previous_weights = np.array([neuron.weights for neuron in layer.neurons]) # create matrix of previus layer weights (for next layer its our current one)
            errors = np.dot(errors, previous_weights) # compute next layer errors

    def extract(self, file_path : str):
        with open(file_path, "w") as f:
            for layer in self.layers:
                f.write(f"*-*{layer.name}//{layer.n_out}//{layer.eta}//{layer.activation_function.name}\n")
                for neuron in layer.neurons:
                    f.write("|".join(map(str, neuron.weights)))
                    f.write("|" + str(neuron.bias) + "\n")

    def load(self, 
            file_path : str, 
            name_to_layer : dict[str,Layer], 
            name_to_function : dict[str, ActivationFunction]):
        with open(file_path, "r") as f:
            lines = f.readlines()
        layer_data = None
        prev_size = self.input_number
        list_of_weights : list[list[float]] = []
        for i in range(len(lines) + 1):
            if i == len(lines) or lines[i].startswith("*-*"):
                if layer_data != None:
                    self.layers.append(
                        name_to_layer[layer_data["name"]](
                            layer_data["n_out"], 
                            layer_data["eta"],
                            name_to_function[layer_data["activation_function"]]
                        )
                    )
                    self.layers[-1]._initialize(prev_size, list_of_weights)
                    prev_size = layer_data["n_out"]
                if i == len(lines):
                    break
                layer_data = {}
                list_of_weights = []
                layer_txt = lines[i][3:].strip()
                layer_txt = layer_txt.split("//")
                layer_data["name"] = layer_txt[0]
                layer_data["n_out"] = int(layer_txt[1])
                layer_data["eta"] = float(layer_txt[2])
                layer_data["activation_function"] = layer_txt[3]
                continue
            if lines[i][0] == "N":
                continue
            list_of_weights.append([])
            weights_txt = lines[i].strip().split("|")
            for weight in weights_txt:
                list_of_weights[-1].append(float(weight))
