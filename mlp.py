import numpy as np

"""
network = model.createNetwork([
    layers.DenseLayer(input_shape, activation='sigmoid'),
    layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
    layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
    layers.DenseLayer(24, activation='sigmoid', weights_initializer='heUniform'),
    layers.DenseLayer(output_shape, activation='softmax', weights_initializer='heUniform')
])
model.fit(network, data_train, data_valid, loss='binaryCrossentropy', learning_rate=0.0314, batch_size=8, epochs=84)
"""

SEED_GENERATOR = 42


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z: np.ndarray) -> np.ndarray:
    zeros = np.zeros_like(z)
    return np.maximum(zeros, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return 1 if z > 0 else 0


def binary_cross_entropy(Y_H: np.ndarray, Y: np.ndarray) -> np.ndarray:
    m = Y_H.shape[0]
    print("Y_h: ", Y_H, " Y: ", Y, "Y.T: ", Y.T)
    # l = -np.matmul(Y.T, np.log(Y_H))
    # r = -np.matmul((1 - Y).T, np.log((1 - Y_H)))
    l = -Y.T * np.log(Y_H)
    r = -(1 - Y).T * np.log((1 - Y_H))
    return (1 / m) * (l + r)


def xavier_init(layer_shape):
    in_dim, out_dim = layer_shape
    xavier_distribution_lim = np.sqrt(6.0, dtype=np.float32) / np.sqrt(
        in_dim + out_dim, dtype=np.float32
    )
    rng = np.random.default_rng(SEED_GENERATOR)
    return rng.uniform(
        low=-xavier_distribution_lim, high=xavier_distribution_lim, size=layer_shape
    )


# Building phase : create each layer of the network in a ordered way with its number of neurons and an activation function and initialize weights


class DenseLayer:
    def __init__(self, units, activation=None, input_shape=None):
        self.units = units
        self.activation_name = activation
        self.activation = relu
        self.input_shape = input_shape
        self.is_input_layer = False if self.input_shape is None else True
        self.bias = 1
        self.output = 0
        self.layer_shape = None

    def __init_weights(self, method=None):
        # The size of the matrix representing the weights
        # is equal to no. units * input_shape
        # print("SHAPE: ", self.layer_shape)
        # self.weights = np.zeros(shape=self.layer_shape, dtype=np.float32)
        self.weights = xavier_init(self.layer_shape)
        self.bias = np.zeros(shape=(self.layer_shape[-1], 1), dtype=np.float32)
        # print("INIT: ", self.weights)

    def get_input_shape(self):
        return self.input_shape

    def get_weights(self):
        return self.weights

    def get_bias(self):
        return self.bias

    def get_units(self):
        return self.units

    def get_output(self):
        return self.output

    def build(self, layer_shape, activation_mapping):
        if isinstance(layer_shape, tuple) and len(layer_shape) == 2:
            self.layer_shape = layer_shape
        else:
            raise ValueError(f"{type(layer_shape)}")
        self.__init_weights()
        self.activation = activation_mapping.get(self.activation_name)

    def layer_forward_pass(self, inputs):
        print("w: ", self.weights.T, "\ni: ", inputs)
        # print(self.weights.T.shape, inputs.shape)
        z = np.matmul(self.weights.T, inputs) + self.bias
        print("z: ", z)
        print("a: ", self.activation)
        output = self.activation(z)
        print("o: ", output)
        return output


class Model:
    def __init__(self, layers):
        self.__check_layers(layers)
        self.layers = []
        self.activation_mapping = self.__build_activation_mapping()
        previous_layer_n_units = 0
        for i, l in enumerate(layers):
            layer_shape = (
                (l.get_input_shape(), l.get_units())
                if i == 0
                else (previous_layer_n_units, l.get_units())
            )
            previous_layer_n_units = l.get_units()
            print(layer_shape)
            l.build(layer_shape, self.activation_mapping)
            self.layers.append(l)

    def __build_activation_mapping(self):
        activation_mapping = {"relu": relu, "sigmoid": sigmoid}
        return activation_mapping

    def __check_layers(self, layers):
        if not isinstance(layers, list):
            raise TypeError(f"Layers should be of type list, received {type(layers)}")
        # if len(layers) < 4:
        #     raise ValueError(f"Minimal number of layers is 4, received {len(layers)}")
        if layers[0].get_input_shape() is None:
            raise ValueError(f"Input shape required for first DenseLayer")
        # TODO: check if isinstance of DenseLayer

    def forward_pass(self, features, y):
        input = features
        for l in self.layers:
            output = l.layer_forward_pass(input)
            input = output
        loss = binary_cross_entropy(output, y)
        return loss

    def get_all_weights(self):
        return [l.get_weights() for l in self.layers]

    def get_all_biases(self):
        return [l.get_bias() for l in self.layers]


# Training phase :
# Forward pass:
# - run the input through the network to compute a prediction
# - calculate the loss of that prediction with the loss function
# Backward prop
# - calculate gradient for each layer and store the value of the new weights
# - update all weights

# Example usage:
if __name__ == "__main__":
    mlp = Model(
        [
            DenseLayer(units=4, activation="relu", input_shape=4),
            # DenseLayer(units=3, activation="relu"),
            # DenseLayer(units=3, activation="relu"),
            DenseLayer(units=1, activation="sigmoid"),
        ]
    )
    features = np.ones((4, 1))
    # print(mlp.get_all_weights())
    # print(mlp.get_all_biases())
    y = np.ones((1, 1))
    print(mlp.forward_pass(features, y))
