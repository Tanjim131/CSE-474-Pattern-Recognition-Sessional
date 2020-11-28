import numpy as np
from scipy.stats import multivariate_normal

class Utility:
    CONFIGURATION_FILE_PATH = './parameter.txt'
    TRAIN_FILE_PATH = './train.txt'
    TEST_FILE_PATH = './test.txt'

    EPS = np.finfo(float).eps

    @staticmethod
    def extract_configuration_parameters():
        with open(Utility.CONFIGURATION_FILE_PATH, 'r') as configuration_file:
            n, l = map(int, configuration_file.readline().split())
            channel_coefficients = np.asarray(list(map(float, configuration_file.readline().split()))).reshape(-1, 1).T  
            assert(channel_coefficients.shape == (1, n)) # Dimension = 1 x n
            noise_variance = float(configuration_file.readline())
        return n, l, channel_coefficients, noise_variance

    @staticmethod
    def extract_train_bits():
        with open(Utility.TRAIN_FILE_PATH, 'r') as train_file:
            train_bits = np.asarray(list(map(int, train_file.readline())))
        return train_bits

    @staticmethod
    def extract_test_bits():
        with open(Utility.TEST_FILE_PATH, 'r') as test_file:
            test_bits = np.asarray(list(map(int, test_file.readline())))
        return test_bits

    @staticmethod
    def read_file(option):
        if option == 'Configuration':
            return Utility.extract_configuration_parameters()
        elif option == 'Train':
            return Utility.extract_train_bits()
        else:
            return Utility.extract_test_bits()

    @staticmethod
    def binary_string_to_decimal(binary_string):
        binary_array = np.asarray(list(map(int, binary_string)))
        return Utility.binary_array_to_decimal(binary_array)

    @staticmethod
    def binary_array_to_decimal(binary_array):
        # converts a binary numpy array to its decimal representation
        # E.g.: Given [1 1 0] this function will return 3 (rightmost bit is MSB, leftmost bit is transmitted earlier than the rest)
        return binary_array.dot(1 << np.arange(binary_array.shape[-1]))

    @staticmethod
    def bit_array_to_bit_string(binary_array):
        binary_string = ''
        for bit in binary_array:
            binary_string += str(bit)
        return binary_string

    @staticmethod
    def write_output(method, binary_string):
        file_name = 'Out1.txt' if method == 'Bayesian' else 'Out2.txt'
        with open(file_name, 'w') as out_file:
            out_file.write(binary_string)


class Channel:
    def __init__(self, channel_coefficients, noise_variance):
        self.channel_coefficients = channel_coefficients
        self.noise_variance = noise_variance

    def compute_inter_symbol_interference(self, transmitted_sequence):
        channels_impulse_response = np.sum(transmitted_sequence * self.channel_coefficients, axis = 1).reshape(-1, 1)
        assert(channels_impulse_response.shape == (transmitted_sequence.shape[0], 1))
        return channels_impulse_response

    def compute_channel_noises(self, number_of_samples):
        channel_noises = np.random.normal(loc = 0.0, scale = np.sqrt(self.noise_variance), size = number_of_samples).reshape(-1, 1)
        return channel_noises


class Equalizer:
    def __init__(self, channel_equalization):
        self.channel_equalization = channel_equalization

    def generate_observations(self):
        number_of_received_bits = len(self.channel_equalization.test_received_sequence)
        observations = []
        for i in range(number_of_received_bits - self.channel_equalization.l + 1):
            x_k = self.channel_equalization.test_received_sequence[i : i + self.channel_equalization.l].flatten().tolist()  # x_k vector
            observations.append(x_k)
        self.observations = np.asarray(observations, dtype = 'object')

    def calculate_observation_mean(self, observation, cluster):
        return multivariate_normal.pdf(observation, self.channel_equalization.cluster_centroids[cluster], self.channel_equalization.cluster_covariance_matrices[cluster])

    def compute_bayesian_transition_cost(self, current_observation, currently_considered_cluster, previous_cluster, first_observation):
        observation_mean = self.calculate_observation_mean(current_observation, currently_considered_cluster)
        if first_observation:
            transition_probability = self.channel_equalization.prior_probabilities[currently_considered_cluster]
        else:
            transition_probability = self.channel_equalization.transition_probabilities[currently_considered_cluster][previous_cluster]
        return np.log(transition_probability * observation_mean + Utility.EPS)

    def compute_euclidean_cost(self, current_observation, currently_considered_cluster):
        return np.linalg.norm(current_observation - self.channel_equalization.cluster_centroids[currently_considered_cluster])

    # Viterbi algorithm
    def reconstruct_tranmission_bits(self, method):
        predicted_tranmission_bits = []
        predicted_clusters = []
        if method == 'Bayesian':
            previous_best_values = np.zeros(self.channel_equalization.number_of_clusters)
            for i in range(len(self.observations)):
                current_best_values = np.zeros(self.channel_equalization.number_of_clusters)
                for j in range(self.channel_equalization.number_of_clusters):
                    cluster_probabilities = []
                    for k in range(self.channel_equalization.number_of_clusters):
                        first_observation_flag = True if i == 0 else False
                        cluster_probability = previous_best_values[k] + self.compute_bayesian_transition_cost(self.observations[i], currently_considered_cluster = j, previous_cluster = k, first_observation = first_observation_flag)
                        cluster_probabilities.append(cluster_probability)
                    current_best_values[j] = np.amax(np.asarray(cluster_probabilities))
                predicted_cluster = np.argmax(current_best_values)
                predicted_clusters.append(predicted_cluster)
                previous_best_values = current_best_values
        else:
            for i in range(len(self.observations)):
                costs = []
                for j in range(self.channel_equalization.number_of_clusters):
                    euclidean_cost = self.compute_euclidean_cost(self.observations[i], currently_considered_cluster = j)             
                    costs.append(euclidean_cost)
                costs = np.asarray(costs)
                predicted_cluster = np.argmin(costs)
                predicted_clusters.append(predicted_cluster)

        for predicted_cluster in predicted_clusters:
            if predicted_cluster < (self.channel_equalization.number_of_clusters / 2):  # for the first half I_k = 0, and for the rest I_k = 1
                predicted_tranmission_bits.append(0)
            else:
                predicted_tranmission_bits.append(1)
        predicted_tranmission_bits = np.asarray(predicted_tranmission_bits, dtype = 'object')
        
        correctly_predicted = np.count_nonzero(predicted_tranmission_bits == self.channel_equalization.test_bits[self.channel_equalization.l:])
        accuracy = correctly_predicted / len(predicted_tranmission_bits) * 100
        return predicted_tranmission_bits, accuracy

class ChannelEqualization:
    def __init__(self, n, l, channel_coefficients, noise_variance, train_bits, test_bits):
        self.n = n
        self.l = l
        self.channel = Channel(channel_coefficients, noise_variance)
        self.train_bits = train_bits
        self.test_bits = test_bits

    def generate_transmitted_bits_sequence(self, option):
        number_of_bits = len(self.train_bits) if option == 'Train' else len(self.test_bits)
        sequence = []
        for i in range(number_of_bits - self.n + 1):
            if option == 'Train':
                transmitted_bits = self.train_bits[i : i + self.n]
            else:
                transmitted_bits = self.test_bits[i : i + self.n]
            sequence.append(transmitted_bits)
        
        if option == 'Train':
            self.train_transmitted_sequence = np.asarray(sequence)
        else:
            self.test_transmitted_sequence = np.asarray(sequence)

    def generate_received_bits_sequence(self, option):
        channel_contribution = self.channel.compute_inter_symbol_interference(self.train_transmitted_sequence if option == 'Train' else self.test_transmitted_sequence) # channel contribution to overall corruption
        noise = self.channel.compute_channel_noises(len(self.train_transmitted_sequence) if option == 'Train' else len(self.test_transmitted_sequence))
        if option == 'Train':
            self.train_received_sequence = channel_contribution + noise
        else:
            self.test_received_sequence = channel_contribution + noise

    def generate_clusters_received_samples_mapping(self):
        number_of_train_bits = len(self.train_bits)
        self.cluster_bits_length = self.n + self.l - 1
        clusters = [[] for _ in range (2 ** self.cluster_bits_length)]
        
        for i in range(number_of_train_bits - self.cluster_bits_length + 1):
            cluster_binary = self.train_bits[i : i + self.cluster_bits_length]  # binary array representation
            cluster_id = Utility.binary_array_to_decimal(cluster_binary)  # decimal representation
            x_k = self.train_received_sequence[i : i + self.l].flatten().tolist()  # x_k vector
            clusters[cluster_id].append(x_k)
        
        clusters = np.asarray([np.asarray(cluster, dtype = 'object') for cluster in clusters], dtype = 'object')
        self.clusters = clusters
    
    def compute_cluster_centroids(self):
        cluster_means = []
        for cluster in self.clusters:
            cluster_mean = np.average(cluster, axis = 0)
            cluster_means.append(cluster_mean)
        self.cluster_centroids = np.asarray(cluster_means, dtype = 'object')

    def compute_cluster_covariance_matrices(self):
        cluster_covariance_matrices = []
        for i in range(len(self.clusters)):
            difference = self.clusters[i] - self.cluster_centroids[i]
            cluster_covariance_matrix = np.matmul(difference.T, difference) / len(self.clusters[i])  # Dimesion = l x l
            assert(cluster_covariance_matrix.shape == (self.l, self.l))
            cluster_covariance_matrices.append(cluster_covariance_matrix)
        self.cluster_covariance_matrices = np.asarray(cluster_covariance_matrices, dtype = 'object')

    def compute_prior_probabilities(self):
        prior_probabilities = []
        total_cluster_size = np.sum(np.asarray([len(cluster) for cluster in self.clusters]))
        for cluster in self.clusters:
            prior_probabilities.append(len(cluster) / total_cluster_size)
        self.prior_probabilities = np.asarray(prior_probabilities)

    def calculate_cluster_transition_probability(self, current_cluster_id, next_possible_cluster_ids):
        total_cluster_size = np.sum(np.asarray([len(self.clusters[cluster_id]) for cluster_id in next_possible_cluster_ids], dtype = 'object'))
        for next_possible_cluster_id in next_possible_cluster_ids:
            self.transition_probabilities[current_cluster_id][next_possible_cluster_id] = len(self.clusters[next_possible_cluster_id]) / total_cluster_size

    def compute_transition_probabilities(self):
        self.number_of_clusters = len(self.clusters)
        self.transition_probabilities = np.zeros((self.number_of_clusters, self.number_of_clusters))
        for i in range(self.number_of_clusters):
            current_cluster_binary_string = np.binary_repr(i, width = self.cluster_bits_length)  # binary_string representation
            current_cluster_id = Utility.binary_string_to_decimal(current_cluster_binary_string)
            next_possbile_cluster_ids = []
            for j in range(2):  # next possible bits are 0 or 1
                next_possible_cluster_binary_string = (current_cluster_binary_string + str(j))[1:]  # binary string representation
                next_possible_cluster_id = Utility.binary_string_to_decimal(next_possible_cluster_binary_string)
                next_possbile_cluster_ids.append(next_possible_cluster_id)
            self.calculate_cluster_transition_probability(current_cluster_id, next_possbile_cluster_ids)

def main():
    n, l, channel_coefficients, noise_variance = Utility.read_file('Configuration')
    train_bits = Utility.read_file('Train')
    test_bits = Utility.read_file('Test')
    channel_equalization = ChannelEqualization(n, l, channel_coefficients, noise_variance, train_bits, test_bits)
    channel_equalization.generate_transmitted_bits_sequence('Train')
    channel_equalization.generate_received_bits_sequence('Train')
    channel_equalization.generate_clusters_received_samples_mapping()
    channel_equalization.compute_cluster_centroids()
    channel_equalization.compute_cluster_covariance_matrices()
    channel_equalization.compute_prior_probabilities()
    channel_equalization.compute_transition_probabilities()
    channel_equalization.generate_transmitted_bits_sequence('Test')
    channel_equalization.generate_received_bits_sequence('Test')
    equalizer = Equalizer(channel_equalization)
    equalizer.generate_observations()
    bayesian_predicted_transmission_bits, bayesian_accuracy = equalizer.reconstruct_tranmission_bits('Bayesian')
    Utility.write_output('Bayesian', Utility.bit_array_to_bit_string(bayesian_predicted_transmission_bits))

    print('Bayesian Accuracy:', bayesian_accuracy)
    euclidean_predicted_transmission_bits, euclidean_accuracy = equalizer.reconstruct_tranmission_bits('Euclidean')
    Utility.write_output('Euclidean', Utility.bit_array_to_bit_string(euclidean_predicted_transmission_bits))
    print('Euclidean Accuracy:', euclidean_accuracy)

if __name__ == '__main__':
    main()