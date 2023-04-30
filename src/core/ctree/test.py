import time
import math
import numpy as np
from functools import wraps
import ctree


def time_wrapper(func):
    @wraps(func)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = func(*args, **kwargs)
        t2 = time.time()
        print(f"{func.__name__} took {t2 - t1} seconds")
        return result
    return measure_time


def ucb_score(p_visit, c_visit, c_prior, c_norm_value, pb_c_base, pb_c_init) -> float:
    pb_c = math.log(
        (p_visit + pb_c_base + 1) / pb_c_base) + pb_c_init
    pb_c *= math.sqrt(p_visit) / (c_visit + 1)

    prior_score = pb_c * c_prior
    value_score = c_norm_value

    return prior_score + value_score


@time_wrapper
def test_ucb_c(p_visit, c_visit, c_prior, c_norm_value, pb_c_base, pb_c_init):
    return ctree.ucb_score(p_visit, c_visit, c_prior, c_norm_value, pb_c_base, pb_c_init)


@time_wrapper
def test_ucb_py(n_times, p_visit, c_visit, c_prior, c_norm_value, pb_c_base, pb_c_init):
    return ucb_score(n_times, p_visit, c_visit, c_prior, c_norm_value, pb_c_base, pb_c_init)


@time_wrapper
def test_select_child_c(p_visit_arr, c_visit_arr, c_prior_arr, c_norm_value_arr, pb_c_base, pb_c_init):
    return ctree.select_child(p_visit_arr, c_visit_arr, c_prior_arr, c_norm_value_arr, pb_c_base, pb_c_init)


@time_wrapper
def test_select_child_py(p_visit_arr, c_visit_arr, c_prior_arr, c_norm_value_arr, pb_c_base, pb_c_init):
    index = 0
    max_score = -10000
    for i in range(len(p_visit_arr)):
        score = ucb_score(p_visit_arr[i], c_visit_arr[i], c_prior_arr[i], c_norm_value_arr[i], pb_c_base, pb_c_init)
        if score > max_score:
            score = max_score
            index = i
    return index


if __name__=='__main__':
    # n_times = 10000
    # p_visit = 6
    # c_visit = 3
    # c_prior = 0.3
    # c_norm_value = 0.25
    # pb_c_base = 19652
    # pb_c_init = 1.25
    #
    # res_c = test_ucb_c(p_visit, c_visit, c_prior, c_norm_value, pb_c_base, pb_c_init)
    # res_py = test_ucb_py(n_times, p_visit, c_visit, c_prior, c_norm_value, pb_c_base, pb_c_init)
    #
    # print('c: ', res_c)
    # print('py: ', res_py)

    p_visit = np.array([6, 5, 10, 2, 4, 5])
    c_visit = np.array([3, 2, 5, 1, 2, 3])
    c_prior = np.array([0.3, 0.2, 0.1, 0.5, 0.7, 0.4]).astype(np.float32)
    c_norm_value = np.array([0.25, 0.51, 0.24, 0.53, 0.61, 0.15]).astype(np.float32)
    pb_c_base = 19652
    pb_c_init = 1.25

    res_c = test_select_child_c(p_visit, c_visit, c_prior, c_norm_value, pb_c_base, pb_c_init)
    res_py = test_select_child_py(p_visit, c_visit, c_prior, c_norm_value, pb_c_base, pb_c_init)

    print('c: ', res_c)
    print('py: ', res_py)


