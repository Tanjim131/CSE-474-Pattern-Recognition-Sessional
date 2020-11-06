import cv2
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable

class Utility:
    INPUT_REFERENCE_IMAGE = './reference.jpg'
    INPUT_VIDEO_FILE = './input.mov'

    @staticmethod
    def extract_video_frames():
        original_frame_matrices = []
        grayscale_frame_matrices = [] 
        vidcap = cv2.VideoCapture(Utility.INPUT_VIDEO_FILE)
        while True:
            success, image = vidcap.read()
            if not success:
                break
            original_frame_matrices.append(image)
            grayscale_frame_matrices.append(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))  # converted to grayscale
        return vidcap, np.asarray(original_frame_matrices), np.asarray(grayscale_frame_matrices)
    
    @staticmethod
    def get_reference_image():
        return cv2.cvtColor(cv2.imread(Utility.INPUT_REFERENCE_IMAGE), cv2.COLOR_BGR2GRAY)

    @staticmethod
    def generate_video(vidcap, frame_matrices, method):
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        height, width, layers = frame_matrices[0].shape
        outputV = cv2.VideoWriter('output_' + method + '.mov', fourcc, fps, (width, height))
        for frame in frame_matrices:
            outputV.write(frame)
        outputV.release()

class Methods:
    def __init__(self, vidcap, original_frame_matrices, grayscale_frame_matrices, reference_image_matrix):
        self.vidcap = vidcap
        self.original_frame_matrices = original_frame_matrices
        self.grayscale_frame_matrices = grayscale_frame_matrices
        self.reference_image_matrix = reference_image_matrix
        self.reference_image_matrix_height, self.reference_image_matrix_width =  self.reference_image_matrix.shape
        self.frame_matrix_height, self.frame_matrix_width = self.grayscale_frame_matrices[0].shape

    def is_valid_location(self, i, j, reference_image_matrix, frame_matrix):
        reference_image_matrix_height, reference_image_matrix_width = reference_image_matrix.shape
        frame_matrix_height, frame_matrix_width = frame_matrix.shape
        return i >= 0 and i + reference_image_matrix_height < frame_matrix_height and j >= 0 and j + reference_image_matrix_width < frame_matrix_width

    def calculate_cost(self, reference_image_matrix, block_matrix):
        return np.sum((reference_image_matrix / 255.0 - block_matrix / 255.0) ** 2)

    def draw_frame_rectangle(self, best_location_top_left, frame_matrix):
        best_location_bottom_right = best_location_top_left[0] + self.reference_image_matrix_width, best_location_top_left[1] + self.reference_image_matrix_height 
        color = (0, 0, 255)
        thickness = 2
        cv2.rectangle(frame_matrix, best_location_top_left, best_location_bottom_right, color, thickness)
        return frame_matrix

    def exhaustive_search_on_submatrix(self, reference_image_matrix, grayscale_frame_matrix, previous_best_location, p):
        previous_best_height, previous_best_width = previous_best_location
        reference_image_matrix_height, reference_image_matrix_width =  reference_image_matrix.shape
        minimum_value = np.inf
        new_best_location = -1, -1

        search_counter = 0

        for i in range(previous_best_height - p, previous_best_height + p + 1):
            for j in range(previous_best_width - p, previous_best_width + p + 1):
                if not self.is_valid_location(i, j, reference_image_matrix, grayscale_frame_matrix):
                    continue
                search_counter += 1
                block_matrix = grayscale_frame_matrix[i : i + reference_image_matrix_height, j : j + reference_image_matrix_width]
                value = self.calculate_cost(reference_image_matrix, block_matrix)
                if value < minimum_value:
                    minimum_value = value
                    new_best_location = i , j

        return new_best_location, search_counter

    def exhaustive_search_on_entire_matrix(self, reference_image_matrix, grayscale_frame_matrix):
        reference_image_matrix_height, reference_image_matrix_width =  reference_image_matrix.shape
        frame_matrix_height, frame_matrix_width = grayscale_frame_matrix.shape
        best_location = -1, -1
        minimum_value = np.inf

        search_counter = 0
        for i in range(frame_matrix_height - reference_image_matrix_height + 1):
            for j in range(frame_matrix_width - reference_image_matrix_width + 1):
                search_counter += 1
                block_matrix = grayscale_frame_matrix[i : i + reference_image_matrix_height, j : j + reference_image_matrix_width]
                value = self.calculate_cost(reference_image_matrix, block_matrix)
                if value < minimum_value:
                    minimum_value = value
                    best_location = i , j

        return best_location, search_counter

    def logarithmic_2d_search(self, reference_image_matrix, grayscale_frame_matrix, previous_best_location, p):
        previous_best_height, previous_best_width = previous_best_location
        reference_image_matrix_height, reference_image_matrix_width =  reference_image_matrix.shape
        minimum_value = np.inf
        new_best_location = -1, -1

        k = np.int(np.ceil(np.log2(p)))
        d = 2 ** (k - 1)
        p //= 2

        search_counter = 0

        while d > 1:
            for i in range(previous_best_height - d, previous_best_height + d + 1, d):
                for j in range(previous_best_width - d, previous_best_width + d + 1, d):
                    if not self.is_valid_location(i, j, reference_image_matrix, grayscale_frame_matrix):
                        continue
                    search_counter += 1
                    block_matrix = grayscale_frame_matrix[i : i + reference_image_matrix_height, j : j + reference_image_matrix_width]
                    value = self.calculate_cost(reference_image_matrix, block_matrix)
                    if value < minimum_value:
                        minimum_value = value
                        new_best_location = i , j
            k = np.int(np.ceil(np.log2(p)))
            d = 2 ** (k - 1)
            p //= 2

        return new_best_location, search_counter        

    def hierarchical_search(self, reference_image_matrix, grayscale_frame_matrix, previous_best_location, p):
        level_wise_reference_image_matrices = [reference_image_matrix]
        level_wise_reference_image_matrices.append(cv2.pyrDown(level_wise_reference_image_matrices[0]))
        level_wise_reference_image_matrices.append(cv2.pyrDown(level_wise_reference_image_matrices[1]))

        level_wise_grayscale_frame_matrices = [grayscale_frame_matrix]
        level_wise_grayscale_frame_matrices.append(cv2.pyrDown(level_wise_grayscale_frame_matrices[0]))
        level_wise_grayscale_frame_matrices.append(cv2.pyrDown(level_wise_grayscale_frame_matrices[1]))

        x, y = previous_best_location
        (x1, y1), counter1 = self.exhaustive_search_on_submatrix(level_wise_reference_image_matrices[2], level_wise_grayscale_frame_matrices[2], (x // 4, y // 4), p // 4)
        (x2, y2), counter2 = self.exhaustive_search_on_submatrix(level_wise_reference_image_matrices[1], level_wise_grayscale_frame_matrices[1], (2 * x1, 2 * y1), p // 2)
        best_location, counter3 = self.exhaustive_search_on_submatrix(level_wise_reference_image_matrices[0], level_wise_grayscale_frame_matrices[0], (2 * x2, 2 * y2), p)

        search_counter = counter1 + counter2 + counter3

        return best_location, search_counter

    def execute_search(self, method, p):
        frame_matrices = self.original_frame_matrices.copy()
        best_location, search_counter = self.exhaustive_search_on_entire_matrix(self.reference_image_matrix, self.grayscale_frame_matrices[0])
        total_search_counter = search_counter
        for i in range(1, len(frame_matrices)):
            if method == "exhaustive":
                best_location_top_left, search_counter = self.exhaustive_search_on_submatrix(self.reference_image_matrix, self.grayscale_frame_matrices[i], best_location, p)
            elif method == "2d_logarithmic":
                best_location_top_left, search_counter = self.logarithmic_2d_search(self.reference_image_matrix, self.grayscale_frame_matrices[i], best_location, p)
            else:
                best_location_top_left, search_counter = self.hierarchical_search(self.reference_image_matrix.copy(), self.grayscale_frame_matrices[i], best_location, p)
            best_location_top_left = best_location_top_left[::-1]
            frame_matrices[i] = self.draw_frame_rectangle(best_location_top_left, frame_matrices[i])
            best_location = best_location_top_left[::-1]
            total_search_counter += search_counter
        Utility.generate_video(self.vidcap, frame_matrices, method)

        return total_search_counter

def main():
    vidcap, original_frame_matrices, grayscale_frame_matrices = Utility.extract_video_frames()
    reference_image_matrix = Utility.get_reference_image()
    solver = Methods(vidcap, original_frame_matrices, grayscale_frame_matrices, reference_image_matrix)
    number_of_frames = len(original_frame_matrices)

    exhaustive_list = []
    logarithmic_list = []
    hierarchical_list = []

    p_start = 5
    p_end = 20
    for p in range(p_start, p_end + 1):
        exhaustive_search_counter = solver.execute_search(method = "exhaustive", p = p)
        logarithmic_search_counter = solver.execute_search(method = "2d_logarithmic", p = p)
        hierarchical_search_counter = solver.execute_search(method  = "hierarchical", p = p)
        exhaustive_list.append((p, exhaustive_search_counter / number_of_frames))
        logarithmic_list.append((p, logarithmic_search_counter / number_of_frames))
        hierarchical_list.append((p, hierarchical_search_counter / number_of_frames))

    exhaustive_array = np.asarray(exhaustive_list)        
    logarithmic_array = np.asarray(logarithmic_list)
    hierarchical_array = np.asarray(hierarchical_list) 

    plt.plot(exhaustive_array[:,0], exhaustive_array[:,1], "-b", label = "Exhaustive")
    plt.plot(logarithmic_array[:,0], logarithmic_array[:,1], "-r", label = "2D Log")
    plt.plot(hierarchical_array[:,0], hierarchical_array[:,1], "-g", label = "Hierarchical")

    plt.xlabel('p', fontsize = 18)
    plt.ylabel('frame search count', fontsize = 18)
    plt.legend(loc = "upper left")

    plt.show()

    pretty_table = PrettyTable(['p', 'Exhaustive', '2D Log', 'Hierarchical'])
    for i in range(len(exhaustive_array)):
        pretty_table.add_row([exhaustive_array[i][0], exhaustive_array[i][1] , logarithmic_array[i][1], hierarchical_array[i][1]])
        
    with open('1505082_report.txt', 'w') as f:
        f.write(pretty_table.get_string())

if __name__ == "__main__":
    main()