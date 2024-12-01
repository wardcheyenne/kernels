import numpy as np
import random

feature_map = [lambda x: x**2, lambda x: x, lambda x: 1]
data_space = [np.random.randint(1, 20) for _ in range(5)]
weights = [1, -1, 1]

#The function space defines a unique function in the RKHS. 
function_space = [
    lambda x, f=f, c=c: c * f(x) 
    for f, c in zip(feature_map, weights)
]

def eval_RKHS(points, function_space):
    eval_points = []
    for i in points:
        eval_points.append([f(i) for f in function_space])
    return np.dot(eval_points[0], eval_points[1]), eval_points, points

print(eval_RKHS(random.sample(data_space, 2), function_space))
