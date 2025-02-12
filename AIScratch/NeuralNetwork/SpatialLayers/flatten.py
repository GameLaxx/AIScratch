from AIScratch.NeuralNetwork.SpatialLayers import SpatialLayer

class FlattenLayer(SpatialLayer):
    def __init__(self):
        super().__init__(0, 0, 0, 0, None, "flatten")

    def _initialize(self, size_in, optimizer, list_of_filters=None):
        self.size_in = size_in
        self.size_out = (size_in[0]*size_in[1]*size_in[2],)

    def forward(self, inputs, is_training=False):
        return inputs.flatten()
    
    def propagation(self, grad_L_z):
        return grad_L_z.reshape(self.size_in)
            
    def store(self, grad_L_z):
        pass
    
    def backward(self):
        pass