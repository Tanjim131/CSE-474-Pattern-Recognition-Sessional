import pandas as pd
import numpy as np

class Perceptron:
    linearly_separable_train_file_path = "./trainLinearlySeparable.txt"
    linearly_separable_test_file_path = "./testLinearlySeparable.txt"

    linearly_non_separable_train_file_path = "./trainLinearlyNonSeparable.txt" 
    linearly_non_separable_test_file_path = "./testLinearlyNonSeparable.txt"

    def __init__(self) : pass

    def generate_column_names(self, total_features):
        column_names = []
        for i in range(total_features):
            column_names.append("F" + str(i))
        column_names.append("Class")
        return column_names

    def compute_accuracy(self, option):
        if option == "Basic":
            total_features, weight_vector = self.basic_algorithm()
            file_path = self.linearly_separable_test_file_path
        elif option == "RP":
            total_features, weight_vector = self.rp_algorithm()
            file_path = self.linearly_separable_test_file_path
        else:
            total_features, weight_vector = self.pocket_algorithm()
            file_path = self.linearly_non_separable_test_file_path

        data_frame = self.get_data_frame(file_path, total_features, 0)
        unique_classes = sorted(set(data_frame["Class"]))
        match_count = 0
  
        for row in data_frame.to_numpy():
            actual_class = row[-1]
            feature_vector = row[ : -1].reshape(-1, 1)
            
            product = np.matmul(np.transpose(weight_vector), feature_vector)[0,0]
            
            if product > 0:
                predicted_class = unique_classes[0]
            else:
                predicted_class = unique_classes[1]
            
            if predicted_class == actual_class:
                match_count += 1
        
        accuracy = match_count / len(data_frame) * 100.0

        return accuracy

    def get_paramaters(self, file_path):
        with open(file_path) as file:
            parameters = file.readline()
        return [int(parameter) for parameter in parameters.split()]

    def get_data_frame(self, file_path, total_features, skip_rows = 1):
        column_names = self.generate_column_names(total_features)
        data_frame = pd.read_csv(file_path, skiprows = skip_rows, delim_whitespace = True, header = None, skipinitialspace = True, names = column_names)
        # extended dimension
        dummy_feature = "F" + str(total_features)
        data_frame.insert(total_features, dummy_feature, 1)
        return data_frame

    def basic_algorithm(self):
        total_features, total_classes, total_samples = self.get_paramaters(self.linearly_separable_train_file_path)
        data_frame = self.get_data_frame(self.linearly_separable_train_file_path, total_features)
    
        weight_vector = np.zeros((total_features + 1, 1))
        learning_rate = 0.1

        unique_classes = sorted(set(data_frame["Class"]))
        iterations = 0
        while True:
            misclassified_samples = []

            for row in data_frame.to_numpy():
                actual_class = row[-1]
                feature_vector = row[ : -1].reshape(-1, 1)
                term = np.matmul(np.transpose(weight_vector), feature_vector)[0,0]
                delta = 1
                if actual_class == unique_classes[0]:
                    delta = -1
                term *= delta
                if term >= 0:
                    misclassified_samples.append((feature_vector, delta))

            if not misclassified_samples:
                break
            
            vector_to_subtract = np.zeros((total_features + 1, 1))
            for feature_vector, delta in misclassified_samples:
                feature_vector *= learning_rate * delta
                vector_to_subtract = np.add(vector_to_subtract, feature_vector)
            
            weight_vector = np.subtract(weight_vector, vector_to_subtract)
            iterations += 1
            learning_rate = 1 / (iterations)

        print("Training Basic Perceptron classifier took", iterations, "iterations to terminate.")
            
        return total_features, weight_vector

    def rp_algorithm(self):
        total_features, total_classes, total_samples = self.get_paramaters(self.linearly_separable_train_file_path)
        data_frame = self.get_data_frame(self.linearly_separable_train_file_path, total_features)

        weight_vector = np.zeros((total_features + 1, 1))
        learning_rate = 1

        unique_classes = sorted(set(data_frame["Class"]))
        iterations = 0
        correct_classification_count = 0
        while True:
            for row in data_frame.to_numpy():
                actual_class = row[-1]
                feature_vector = row[ : -1].reshape(-1, 1)
                term = np.matmul(np.transpose(weight_vector), feature_vector)[0,0]
                
                if actual_class == unique_classes[0] and term <= 0:
                    correct_classification_count = 0
                    weight_vector = np.add(weight_vector, learning_rate * feature_vector)
                elif actual_class == unique_classes[1] and term >= 0:
                    correct_classification_count = 0
                    weight_vector = np.subtract(weight_vector, learning_rate * feature_vector)
                else:
                    correct_classification_count += 1

                iterations += 1            
                learning_rate = 1 / iterations

                if correct_classification_count == len(data_frame):
                    break

            if correct_classification_count == len(data_frame):
                break
        
        print("Training Reward & Punishment classifier took", iterations, "iterations to terminate.") 
        
        return total_features, weight_vector

    def pocket_algorithm(self):
        MAX_ITERATIONS = 40

        total_features, total_classes, total_samples = self.get_paramaters(self.linearly_non_separable_train_file_path)
        data_frame = self.get_data_frame(self.linearly_non_separable_train_file_path, total_features)

        weight_vector = np.zeros((total_features + 1, 1))
        learning_rate = 1

        stored_vector = np.zeros((total_features + 1, 1))
        history_counter = 0

        unique_classes = sorted(set(data_frame["Class"]))
        iterations = 0
        for iterations in range(1, MAX_ITERATIONS + 1):
            misclassified_samples = []

            for row in data_frame.to_numpy():
                actual_class = row[-1]
                feature_vector = row[ : -1].reshape(-1, 1)
                term = np.matmul(np.transpose(weight_vector), feature_vector)[0,0]
                delta = 1
                if actual_class == unique_classes[0]:
                    delta = -1
                term *= delta
                if term >= 0:
                    misclassified_samples.append((feature_vector, delta))

            if not misclassified_samples:
                break
            
            vector_to_subtract = np.zeros((total_features + 1, 1))
            for feature_vector, delta in misclassified_samples:
                feature_vector *= learning_rate * delta
                vector_to_subtract = np.add(vector_to_subtract, feature_vector)
            
            weight_vector = np.subtract(weight_vector, vector_to_subtract)

            match_count = 0
            for row in data_frame.to_numpy():
                actual_class = row[-1]
                feature_vector = row[ : -1].reshape(-1, 1)
        
                product = np.matmul(np.transpose(weight_vector), feature_vector)[0,0]
                
                if product > 0:
                    predicted_class = unique_classes[0]
                else:
                    predicted_class = unique_classes[1]
                
                if predicted_class == actual_class:
                    match_count += 1

            if match_count > history_counter:
                history_counter = match_count
                stored_vector = weight_vector

            learning_rate = 1 / iterations

        print("Pocket Perceptron classifier took", MAX_ITERATIONS, "iterations to terminate.")
            
        return total_features, stored_vector

def main():
    solver = Perceptron()
    algorithms = ["Basic", "RP", "Pocket"]
    for algorithm in algorithms:
        print("Accuracy of", algorithm, "=", solver.compute_accuracy(algorithm))

if __name__ == "__main__":
    main() 