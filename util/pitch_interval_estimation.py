import numpy as np
from matplotlib import pyplot as plt
def compute_cost(a, b, x, y, E_line=0.1):
    # a and b are indexes into array x and y
    # x is the horizontal axis, y is the vertical
    cost = E_line
    slope = (y[b] - y[a]) / (x[b] - x[a])
    y_int = y[b] - slope * x[b]
    real_y = y[a:b]
    vertical_difference = real_y - (y_int + slope * x[a:b])
    vertical_difference = np.linalg.norm(vertical_difference, ord=1)
    cost = cost + vertical_difference
    return cost
def traverse_solution(back_track):


    queue = [[0, back_track.shape[0] - 1]]
    if back_track.shape[0] == 1:
        back_track[0, 0] = -1
    sol = []
    while len(queue) > 0:
        current = queue.pop(0)
        current_pointer = int(back_track[current[0], current[1]])
        if current_pointer == -1:
            sol.append([current[0], current[1]])
        else:
            queue = [[current[0], current_pointer], [current_pointer, current[1]]] + queue
    return sol
def piece_wise_linear_intervals(x, y, E_line=400):
    M = np.zeros((y.shape[0], y.shape[0]))
    back_track = np.zeros((y.shape[0], y.shape[0]))
    for i in range(1, y.shape[0]):
        # for each diagonal
        diagona_size = y.shape[0] - i
        diag_i = np.zeros((diagona_size,))
        for a in range(0, diagona_size):
            # for each element in the diagonal
            diag_i[a] = compute_cost(a, i + a, x, y, E_line)
            back_track[a, a + i] = -1
            # iterate through the precomputed matrix
            for k in range(1, i):
                combined_cost = M[a, a + i - k] + M[a + (i - k), a + i]
                if combined_cost < diag_i[a]:
                    diag_i[a] = combined_cost
                    back_track[a, a + i] = a + i - k
        M = M + np.diag(diag_i, i)

    # M is the value matrix, backtrack is the matrix that contains the solution
    return traverse_solution(back_track)
def efficient_piece_wise_linear_intervals(x, y):
    L = 200
    # compute E_line
    E_line = (y.max() - y.min()) / 2


    # divide the input to shorter subarrays
    sub_x_lists = []
    sub_y_lists = []
    for i in range(0, int(np.ceil(x.shape[0] / L))):
        sub_x_lists.append(x[int(i * L):int(min((i + 1) * L, x.shape[0]))])
        sub_y_lists.append(y[int(i * L):int(min((i + 1) * L, x.shape[0]))])

    # use dynamic programming to get linear intervals
    pitch_intervals_index = []
    for i in range(0, len(sub_x_lists)):
        sol_index = piece_wise_linear_intervals(sub_x_lists[i], sub_y_lists[i], E_line)
        sol_index = [[val[0] + int(i * L), val[1] + int(i * L)] for val in sol_index]
        sol = [[x[val[0]], x[val[1]]] for val in sol_index]
        pitch_intervals_index = pitch_intervals_index + sol_index
    for i in range(0, len(pitch_intervals_index) - 1):
        if pitch_intervals_index[i][1] < pitch_intervals_index[i + 1][0]:
            pitch_intervals_index[i][1] = pitch_intervals_index[i + 1][0]

    # obtain the slope of these intervals
    pitch_intervals_slopes = []
    for i in range(0, len(pitch_intervals_index)):
        interval_i_index = pitch_intervals_index[i]
        slope = (y[interval_i_index[1]] - y[interval_i_index[0]]) / (x[interval_i_index[1]] - x[interval_i_index[0]])
        pitch_intervals_slopes.append(slope)

    if len(pitch_intervals_slopes) == 1:
        return pitch_intervals_slopes, pitch_intervals_index
    # merge nearby intervals
    pitch_intervals = []
    pitch_slope = []
    prev_begin = 0
    prev_slope = pitch_intervals_slopes[0]
    counting = 1
    for i in range(1, len(pitch_intervals_slopes)):
        current_slope = pitch_intervals_slopes[i]
        if abs(current_slope - prev_slope) <= 30:
            prev_slope = ((prev_slope * counting) + current_slope) / (counting + 1)
            counting = counting + 1
            if i == len(pitch_intervals_slopes) - 1:
                pitch_intervals.append([prev_begin, pitch_intervals_index[i][1]])
                pitch_slope.append(prev_slope)
        else:
            if counting > 1:
                pitch_intervals.append([prev_begin, pitch_intervals_index[i - 1][1]])
                pitch_slope.append(prev_slope)
            else:
                pitch_intervals.append([pitch_intervals_index[i - 1][0], pitch_intervals_index[i - 1][1]])
                pitch_slope.append(prev_slope)
            prev_slope = current_slope
            counting = 1
            prev_begin = pitch_intervals_index[i][0]
            if i == len(pitch_intervals_slopes) - 1:
                pitch_intervals.append([prev_begin, pitch_intervals_index[i][1]])
                pitch_slope.append(prev_slope)
    # pitch_slope = [0 if abs(val) < 25 else val for val in pitch_slope]
    return pitch_slope, pitch_intervals
def plot_piece_wise_lienar_intervals(xs, original_curve, intervals, slope):
    x, y = get_key_points(xs, original_curve, intervals, slope)
    plt.plot(x, y, "o")
    plt.plot(xs, original_curve)
    plt.show()
    return x, y
def get_key_points(xs, original_curve, intervals, slope):
    x = [xs[intervals[0][0]]]
    y = [original_curve[intervals[0][0]]]
    for i in range(0, len(intervals)):
        x.append(xs[intervals[i][1]])
        y.append(original_curve[intervals[i][1]])
    x = np.array(x)
    y = np.array(y)
    return x, y
