import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class Utility:
    train_file = "./trainNN.txt"
    test_file = "./testNN.txt"

    @staticmethod
    def get_feature_vectors_and_actual_class_values(option):
        data_frame = Utility.get_data_frame(option)

        feature_vectors = data_frame[data_frame.columns[:-1]].values
        actual_class_values = data_frame[data_frame.columns[-1]].values

        # normalization to avoid vanishing gradient problem
        modified_feature_vectors = (feature_vectors - feature_vectors.mean(axis = 0)) / feature_vectors.std(axis = 0)
    
        if option == "Train":
            number_of_features, number_of_classes = Utility.get_number_of_features_and_classes(data_frame)
            encoded_class_values = Utility.encode_class_values(actual_class_values) 
            return number_of_features, number_of_classes, modified_feature_vectors, encoded_class_values

        return modified_feature_vectors, actual_class_values

    @staticmethod 
    def get_data_frame(option):
        if option == "Train":
            file = Utility.train_file
        else:
            file = Utility.test_file
        
        data_frame = pd.read_csv(file, delim_whitespace = True, header = None)
        
        return data_frame

    @staticmethod
    def get_number_of_features_and_classes(data_frame):
        number_of_features = data_frame.shape[1] - 1  # excluding the class (last column)
        number_of_classes = len(set(data_frame[data_frame.columns[-1]].values))
        return number_of_features, number_of_classes

    @staticmethod
    def encode_class_values(actual_class_values):
        encoder = OneHotEncoder(sparse = False)
        encoded_class_values = encoder.fit_transform(actual_class_values.reshape(-1, 1))
        return encoded_class_values

    @staticmethod
    def activation_output(V, activator = "Sigmoid"):
        if activator == "Sigmoid":
            return 1.0 / (1.0 + np.exp(-V))

    @staticmethod
    def activation_derivate(V, activator = "Sigmoid"):
        ac_output = Utility.activation_output(V, activator)
        return ac_output * (1 - ac_output)


class Layer:
    def __init__(self, number_of_neurons, number_of_features):
        np.random.seed(1)
        self.transposed_weight_vectors = np.random.randn(number_of_neurons, number_of_features)
        np.random.seed(0)
        self.biases = np.random.randn(number_of_neurons, 1)
        self.pre_activation_output = None
        self.activation_output = None
        self.input = None


class NeuralNetwork:
    def __init__(self, number_of_features, number_of_classes, hidden_layer_sizes):
        self.number_of_features = number_of_features
        self.number_of_classes = number_of_classes
        self.hidden_layer_sizes = hidden_layer_sizes  
        self.layers = []  # hidden and output layers
        self.number_of_layers = None
        self.deltas = []  # hidden and output layers
        
    
    def add_layers(self):
        layer_sizes = [self.number_of_features] + self.hidden_layer_sizes + [self.number_of_classes]
        self.number_of_layers = len(layer_sizes) - 1;  # excluding the input layer
        for i in range(1, self.number_of_layers + 1):
            self.layers.append(Layer(layer_sizes[i], layer_sizes[i - 1]))


    def feedforward(self, feature_vectors, activation_function):
        previous_layer_output = feature_vectors.copy().T
        
        for layer in self.layers:
            layer.input = previous_layer_output.copy()

            layer.pre_activation_output = np.matmul(layer.transposed_weight_vectors, layer.input) + layer.biases
            layer.activation_output = Utility.activation_output(layer.pre_activation_output, activation_function)

            previous_layer_output = layer.activation_output
        
        return previous_layer_output


    def compute_cost(self, train_actual_class_values):
        cost = np.sum((self.layers[-1].activation_output - train_actual_class_values.T) ** 2)
        return cost / 2.0


    def backward_propagation(self, train_actual_class_values, activation_function):
        self.deltas = [0] * self.number_of_layers
        self.deltas[-1] = (self.layers[-1].activation_output - train_actual_class_values.T) * Utility.activation_derivate(self.layers[-1].pre_activation_output, activation_function)
        # delta dimension -> k_r * number_of_samples

        for i in reversed(range(self.number_of_layers - 1)):
            self.deltas[i] = np.matmul(self.layers[i + 1].transposed_weight_vectors.T, self.deltas[i + 1]) * Utility.activation_derivate(self.layers[i].pre_activation_output)

    
    def update_weights(self, train_feature_vectors, learning_rate):
        for i in range(self.number_of_layers):
            if i == 0:
                previous_layer_output = train_feature_vectors.T
            else:
                previous_layer_output = self.layers[i - 1].activation_output
    
            delta_w = -learning_rate * np.matmul(self.deltas[i], previous_layer_output.T)
            self.layers[i].transposed_weight_vectors += delta_w
            
            delta_b = -learning_rate * np.sum(self.deltas[i], axis=1, keepdims=True)
            self.layers[i].biases += delta_b

    def train_neural_network(self, train_feature_vectors, train_actual_class_values, activation_function = "Sigmoid", learning_rate = 0.001, max_iterations = 10000):
        self.add_layers()
        previous_cost = -1
        # i = 0
        for i in range(max_iterations):
            self.feedforward(train_feature_vectors, activation_function)
            current_cost = self.compute_cost(train_actual_class_values)
            
            if i == 0 or (current_cost < previous_cost and previous_cost - current_cost > 1e-20):
                # print(current_cost)
                previous_cost = current_cost
            else:
                break
            
            self.backward_propagation(train_actual_class_values, activation_function)
            self.update_weights(train_feature_vectors, learning_rate)

            # i += 1
            
    def predict_class(self, test_feature_vectors, activation_function = "Sigmoid"):
        last_layer_output = self.feedforward(test_feature_vectors, activation_function)
        
        prediced_classes = []
        for row in last_layer_output.T:
            prediced_classes.append(np.argmax(row) + 1)  # +1 for 1-based indexing

        return np.asarray(prediced_classes)

    def test_neural_network(self, test_feature_vectors, test_actual_class_values, activation_function = "Sigmoid"):
        predicted_classes = self.predict_class(test_feature_vectors, activation_function)
        # matches = np.count_nonzero(predicted_classes == test_actual_class_values)
        # accuracy = matches / len(test_actual_class_values) * 100
        match_count = 0

        for i in range(len(test_actual_class_values)):
            if predicted_classes[i] == test_actual_class_values[i]:
                match_count += 1
            else:
                print(test_feature_vectors[i])
        
        accuracy = match_count / len(test_feature_vectors) * 100
        return accuracy


def main():
    number_of_features, number_of_classes, train_feature_vectors, train_actual_class_values = Utility.get_feature_vectors_and_actual_class_values("Train")
    test_feature_vectors, test_actual_class_values = Utility.get_feature_vectors_and_actual_class_values("Test")

    nn = NeuralNetwork(number_of_features, number_of_classes, [3, 2, 5, 4, 10, 7])
    nn.train_neural_network(train_feature_vectors, train_actual_class_values)
    accuracy = nn.test_neural_network(test_feature_vectors, test_actual_class_values)
    print(accuracy)
    

if __name__ == "__main__":
    main()