import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.interpolate import splprep, splev
from scipy.ndimage import sobel
from scipy.optimize import minimize
import time

class LaneDetection:
    '''
    Lane detection module using edge detection and b-spline fitting

    args: 
        cut_size (cut_size=68) cut the image at the front of the car
        spline_smoothness (default=10)
        gradient_threshold (default=14)
        distance_maxima_gradient (default=3)
    '''
    def __init__(self, cut_size=65, spline_smoothness=50, gradient_threshold=200, distance_maxima_gradient=6):
        self.car_position = np.array([48, 0])
        self.spline_smoothness = spline_smoothness
        self.cut_size = cut_size
        self.gradient_threshold = gradient_threshold
        self.distance_maxima_gradient = distance_maxima_gradient
        self.lane_boundary1_old = 0
        self.lane_boundary2_old = 0

    def cut_gray(self, state_image_full: np.ndarray) -> np.ndarray:
        '''
        ##### TODO #####
        This function should cut the imagen at the front end of the car (e.g. pixel row 68) 
        and translate to grey scale

        input:
            state_image_full 96x96x3

        output:
            gray_state_image 68x96x1
        '''
        return np.dot(state_image_full[:self.cut_size], [0.299, 0.587, 0.114])[::-1]

    def edge_detection(self, gray_image):
        '''
        ##### TODO #####
        In order to find edges in the gray state image, 
        this function should derive the absolute gradients of the gray state image.
        Derive the absolute gradients using numpy for each pixel. 
        To ignore small gradients, set all gradients below a threshold (self.gradient_threshold) to zero. 

        input:
            gray_state_image 68x96x1

        output:
            gradient_sum 68x96x1

        '''
        # Compute the magnitude of the gradients
        # Display the original grayscale image and the gradient magnitude
        gradient_x = sobel(gray_image, axis=0, mode='nearest')
        gradient_y = sobel(gray_image, axis=1, mode='nearest')
        gradient_magnitude = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        gradient_sum = np.where(gradient_magnitude > self.gradient_threshold, gradient_magnitude, 0)
        return gradient_sum

    def find_maxima_gradient_rowwise(self, gradient_sum):
        '''
        ##### TODO #####
        This function should output arguments of local maxima for each row of the gradient image.
        You can use scipy.signal.find_peaks to detect maxima.
        Hint: Use distance argument for a better robustness.

        input:
            gradient_sum 68x96x1

        output:
            maxima (np.array) 2x Number_maxima

        '''
        return [find_peaks(row, distance=self.distance_maxima_gradient)[0] for row in gradient_sum]


    def find_first_lane_point(self, gradient_sum):
        '''
        Find the first lane_boundaries points above the car.
        Special cases like just detecting one lane_boundary or more than two are considered. 
        Even though there is space for improvement ;) 

        input:
            gradient_sum 68x96x1

        output: 
            lane_boundary1_startpoint
            lane_boundary2_startpoint
            lanes_found  true if lane_boundaries were found
        '''

        # Variable if lanes were found or not
        lanes_found = False
        row = 0

        # loop through the rows
        while not lanes_found:

            # Find peaks with min distance of at least 3 pixel 
            argmaxima = find_peaks(gradient_sum[row], distance=3)[0]

            # if one lane_boundary is found
            if argmaxima.shape[0] == 1:
                lane_boundary1_startpoint = np.array([[argmaxima[0], row]])

                if argmaxima[0] < 48:
                    lane_boundary2_startpoint = np.array([[0, row]])
                else:
                    lane_boundary2_startpoint = np.array([[96, row]])

                lanes_found = True

            # if 2 lane_boundaries are found
            elif argmaxima.shape[0] == 2:
                lane_boundary1_startpoint = np.array([[argmaxima[0], row]])
                lane_boundary2_startpoint = np.array([[argmaxima[1], row]])
                lanes_found = True

            # if more than 2 lane_boundaries are found
            elif argmaxima.shape[0] > 2:
                # if more than two maxima then take the two lanes next to the car, regarding least square
                A = np.argsort((argmaxima - self.car_position[0]) ** 2)
                lane_boundary1_startpoint = np.array([[argmaxima[A[0]], 0]])
                lane_boundary2_startpoint = np.array([[argmaxima[A[1]], 0]])
                lanes_found = True

            row += 1

            # if no lane_boundaries are found
            if row == self.cut_size:
                lane_boundary1_startpoint = np.array([[0, 0]])
                lane_boundary2_startpoint = np.array([[0, 0]])
                break

        return lane_boundary1_startpoint, lane_boundary2_startpoint, lanes_found

    def lane_detection(self, state_image_full):
        '''
        ##### TODO #####
        This function should perform the road detection 

        args:
            state_image_full [96, 96, 3]

        out:
            lane_boundary1 spline
            lane_boundary2 spline
        '''

        # to gray
        global lane_boundary2, lane_boundary1
        gray_state = self.cut_gray(state_image_full)

        # edge detection via gradient sum and thresholding
        gradient_sum = self.edge_detection(gray_state)
        maxima = self.find_maxima_gradient_rowwise(gradient_sum)

        # first lane_boundary points
        lane_boundary1_points, lane_boundary2_points, lane_found = self.find_first_lane_point(gradient_sum)

        # if no lane was found,use lane_boundaries of the preceding step
        if lane_found:

            ##### TODO #####
            #  in every iteration: 
            # 1- find maximum/edge with the lowest distance to the last lane boundary point 
            # 2- append maxium to lane_boundary1_points or lane_boundary2_points
            # 3- delete maximum from maxima
            # 4- stop loop if there is no maximum left 
            #    or if the distance to the next one is too big (>=100)
            def get_deleted(x: np.ndarray, value: int):
                """ 이미 넣은 점은 다시 넣지 않기 위해서 지워줌 """
                return np.delete(x, np.where(np.isin(x, value)))

            def get_appended(x: np.ndarray, point: np.array) -> np.ndarray:
                """ 점을 x에 넣기 위한 함수 """
                return np.append(x, [point], axis=0)

            def get_neighborhoods(arg_maxima: list, point: np.array, rad: int) -> list:
                """ 현재 위치에서 y축 기준으로 rad 반경 안에 있는 모든 arg_maxima(이웃)의 좌표를 구함"""
                neighborhoods = []
                x, y = point[0], point[1]
                for r in range(-rad, rad+1):
                    idx = y + r
                    if self.cut_size > idx >= 0:
                        for v in arg_maxima[idx]:
                            if (x, y) != (v, idx):  # 자기 자신 제외
                                neighborhoods.append(np.array([v, idx]))
                return neighborhoods

            def get_closest(neighborhoods: list, point: np.array) -> (np.array, float):
                """ 가장 가까운 이웃의 좌표를 구함 """
                closest = neighborhoods[0]
                min_dist = np.linalg.norm(closest - point)
                for nbd in neighborhoods:
                    distance = np.linalg.norm(nbd - point)
                    if distance < min_dist:
                        min_dist = distance
                        closest = nbd
                return closest, min_dist

            # lane 1에 대한 과정
            while True:
                curr_point: np.array = lane_boundary1_points[-1]
                nbds = get_neighborhoods(maxima, curr_point, 2)
                if len(nbds) == 0:
                    break
                next_point, dist = get_closest(nbds, curr_point)
                if dist > 20:
                    break
                maxima[next_point[1]] = get_deleted(maxima[next_point[1]], next_point[0])
                lane_boundary1_points = get_appended(lane_boundary1_points, next_point)

            # lane 2에 대한 과정
            while True:
                curr_point: np.array = lane_boundary2_points[-1]
                nbds = get_neighborhoods(maxima, curr_point, 2)
                if len(nbds) == 0:
                    break
                next_point, dist = get_closest(nbds, curr_point)
                if dist > 20:
                    break
                maxima[next_point[1]] = get_deleted(maxima[next_point[1]], next_point[0])
                lane_boundary2_points = get_appended(lane_boundary2_points, next_point)

            ################

            ##### TODO #####
            # spline fitting using scipy.interpolate.splprep 
            # and the arguments self.spline_smoothness
            # 
            # if there are more lane_boundary points points than spline parameters 
            # else use perceding spline
            if lane_boundary1_points.shape[0] > 4 and lane_boundary2_points.shape[0] > 4:
                # Pay attention: the first lane_boundary point might occur twice
                # lane_boundary 1
                lane_boundary1 = splprep(
                    [lane_boundary1_points[1:, 0], lane_boundary1_points[1:, 1]],
                    s=self.spline_smoothness, k=2
                )[0]
                # lane_boundary 2
                lane_boundary2 = splprep(
                    [lane_boundary2_points[1:, 0], lane_boundary2_points[1:, 1]],
                    s=self.spline_smoothness, k=2
                )[0]
                pass
            else:
                lane_boundary1 = self.lane_boundary1_old
                lane_boundary2 = self.lane_boundary2_old
            ################

        else:
            lane_boundary1 = self.lane_boundary1_old
            lane_boundary2 = self.lane_boundary2_old

        self.lane_boundary1_old = lane_boundary1
        self.lane_boundary2_old = lane_boundary2

        # output the spline
        return lane_boundary1, lane_boundary2

    def plot_state_lane(self, state_image_full, steps, fig, waypoints=[]):
        '''
        Plot lanes and way points
        '''
        # evaluate spline for 6 different spline parameters.
        t = np.linspace(0, 1, 6)
        lane_boundary1_points_points = np.array(splev(t, self.lane_boundary1_old))
        lane_boundary2_points_points = np.array(splev(t, self.lane_boundary2_old))

        plt.gcf().clear()
        plt.imshow(state_image_full[::-1])
        plt.plot(lane_boundary1_points_points[0], lane_boundary1_points_points[1] + 96 - self.cut_size, linewidth=5,
                 color='orange')
        plt.plot(lane_boundary2_points_points[0], lane_boundary2_points_points[1] + 96 - self.cut_size, linewidth=5,
                 color='orange')
        if len(waypoints):
            plt.scatter(waypoints[0], waypoints[1] + 96 - self.cut_size, color='white')

        plt.axis('off')
        plt.xlim((-0.5, 95.5))
        plt.ylim((-0.5, 95.5))
        plt.gca().axes.get_xaxis().set_visible(False)
        plt.gca().axes.get_yaxis().set_visible(False)
        fig.canvas.flush_events()
