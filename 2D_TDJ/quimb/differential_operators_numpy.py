# differential operators using finite differences in numpy
import numpy as np

def central_difference_1_2_x(f, dx):
    diff = np.zeros_like(f)
    diff = (np.roll(f, -1, axis=0) - np.roll(f, 1, axis=0)) / (2 * dx)
    return diff

def central_difference_1_2_y(f, dx):
    diff = np.zeros_like(f)
    diff = (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * dx)
    return diff

def central_difference_1_8_x(f, dx):
    coeffs = np.array([4/5, -1/5, 4/105, -1/280]) / dx
    diff = np.zeros_like(f)
    for i, coeff in enumerate(coeffs):
        diff += coeff * (np.roll(f, -i-1, axis=0) - np.roll(f, i+1, axis=0)) 
    return diff

def central_difference_1_8_y(f, dx):
    coeffs = np.array([4/5, -1/5, 4/105, -1/280]) / dx
    diff = np.zeros_like(f)
    for i, coeff in enumerate(coeffs):
        diff += coeff * (np.roll(f, -i-1, axis=1) - np.roll(f, i+1, axis=1)) 
    return diff

def central_difference_2_8_x(f, dx):
    coeffs = np.array([8/5, -1/5, 8/315, -1/560]) / dx**2
    diff = np.zeros_like(f)
    for i, coeff in enumerate(coeffs):
        diff += coeff * (np.roll(f, -i-1, axis=0) + np.roll(f, i+1, axis=0)) 
    diff += -205/72*f/dx**2
    return diff

def central_difference_2_2_x(f, dx):
    coeffs = np.array([1]) / dx**2
    diff = np.zeros_like(f)
    for i, coeff in enumerate(coeffs):
        diff += coeff * (np.roll(f, -i-1, axis=0) + np.roll(f, i+1, axis=0)) 
    diff += -2*f/dx**2
    return diff

def central_difference_2_2_y(f, dx):
    coeffs = np.array([1]) / dx**2
    diff = np.zeros_like(f)
    for i, coeff in enumerate(coeffs):
        diff += coeff * (np.roll(f, -i-1, axis=1) + np.roll(f, i+1, axis=1)) 
    diff += -2*f/dx**2
    return diff

def central_difference_2_8_y(f, dx):
    coeffs = np.array([8/5, -1/5, 8/315, -1/560]) / dx**2
    diff = np.zeros_like(f)
    for i, coeff in enumerate(coeffs):
        diff += coeff * (np.roll(f, -i-1, axis=1) + np.roll(f, i+1, axis=1)) 
    diff += -205/72*f/dx**2
    return diff

def laplace(f, dx):
    diff = np.zeros_like(f)
    diff = central_difference_2_8_x(f, dx) + central_difference_2_8_y(f, dx)
    return diff

def Dx(f, dx):
    return central_difference_1_8_x(f, dx)

def Dy(f, dx):
    return central_difference_1_8_y(f, dx)

def Dx2(f, dx):
    return central_difference_2_8_x(f, dx)

def Dy2(f, dx):
    return central_difference_2_8_y(f, dx)