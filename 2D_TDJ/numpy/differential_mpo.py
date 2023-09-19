# Differential operators in form of MPOs
import numpy as np


def Diff_1_2_y_MPO(n, dx):
    # first order derivative with second order precision in y direction
    # tensor stucture:
    #                          |
    # left_terminator - (central_node)^n - right_terminator
    #                          |
    left_terminator = np.array([1, 1, 1], dtype='float64').reshape(1, 3)
    central_node = np.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    central_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    central_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    central_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: substract 1
    central_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for substraction (2)
    central_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    central_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for substraction (2)
    central_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry(0)
    right_terminator = 1/dx*np.array([0, -1/2, 1/2], dtype='float64').reshape(3, 1) # coefficients

    # Define arrays for MPO
    left_node = np.einsum('ab, burd->aurd', left_terminator, central_node)
    right_node = np.einsum('lurd, re->lued', central_node, right_terminator)
    arrays = [left_node] + [central_node]*(n-2) + [right_node]

    return arrays


def Diff_1_2_x_MPO(n, dx):
    # first order derivative with second order precision in x direction
    # tensor stucture:
    #                          |
    # left_terminator - (central_node)^n - right_terminator
    #                          |
    left_terminator = np.array([1, 1, 1], dtype='float64').reshape(1, 3)
    central_node = np.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    central_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    central_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    central_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: substract 1
    central_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for substraction (2)
    central_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for substraction (2)
    central_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    central_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    right_terminator = 1/dx*np.array([0, -1/2, 1/2], dtype='float64').reshape(3, 1) # coefficients

    # Define arrays for MPO
    left_node = np.einsum('ab, burd->aurd', left_terminator, central_node)
    right_node = np.einsum('lurd, re->lued', central_node, right_terminator)
    arrays = [left_node] + [central_node]*(n-2) + [right_node]

    return arrays


def Diff_2_2_y_MPO(n, dx):
    # second order derivative with second order precision in y direction
    # tensor stucture:
    #                          |
    # left_terminator - (central_node)^n - right_terminator
    #                          |
    left_terminator = np.array([1, 1, 1], dtype='float64').reshape(1, 3)
    central_node = np.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    central_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    central_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    central_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: substract 1
    central_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for substraction (2)
    central_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    central_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for substraction (2)
    central_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry(0)
    right_terminator = 1/dx**2*np.array([-2, 1, 1], dtype='float64').reshape(3, 1) # coefficients

    # Define arrays for MPO
    left_node = np.einsum('ab, burd->aurd', left_terminator, central_node)
    right_node = np.einsum('lurd, re->lued', central_node, right_terminator)
    arrays = [left_node] + [central_node]*(n-2) + [right_node]

    return arrays


def Diff_2_2_x_MPO(n, dx):
    # second order derivative with second order precision in x direction
    # tensor stucture:
    #                          |
    # left_terminator - (central_node)^n - right_terminator
    #                          |
    left_terminator = np.array([1, 1, 1], dtype='float64').reshape(1, 3)
    central_node = np.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    central_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    central_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    central_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: substract 1
    central_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for substraction (2)
    central_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for substraction (2)
    central_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    central_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    right_terminator = 1/dx**2*np.array([-2, 1, 1], dtype='float64').reshape(3, 1) # coefficients

    # Define arrays for MPO
    left_node = np.einsum('ab, burd->aurd', left_terminator, central_node)
    right_node = np.einsum('lurd, re->lued', central_node, right_terminator)
    arrays = [left_node] + [central_node]*(n-2) + [right_node]

    return arrays


def Diff_1_8_y_MPO(n, dx):
    # first order derivative with eigth order precision in y direction
    # tensor stucture:
    #                          |                         |               |
    # left_terminator - (central_node)^(n-2) - second to last node - last node - right_terminator
    #                          |                         |               |
    # left terminator
    left_terminator = np.array([1, 1, 1], dtype='float64').reshape(1, 3)
    # central node
    central_node = np.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    central_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    central_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    central_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: substract 1
    central_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for substraction (2)
    central_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    central_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for substraction (2)
    central_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry(0)
    # second to last node
    sec_last_node = np.zeros((3, 4, 5, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    sec_last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    sec_last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    sec_last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    sec_last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    sec_last_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    sec_last_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    sec_last_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    sec_last_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: substract 1
    sec_last_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for substraction (2)
    sec_last_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    sec_last_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for substraction (2)
    sec_last_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry (0)
    # 3 from right: identity with carry for addition
    sec_last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    sec_last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    sec_last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    sec_last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    # 4 from right: identity with carry for substraction
    sec_last_node[2, 0, 4, 0] =  1   # 00 -> 00 carry for substraction (2)
    sec_last_node[2, 1, 4, 1] =  1   # 01 -> 01 carry for substraction (2)
    sec_last_node[2, 2, 4, 2] =  1   # 10 -> 10 carry for substraction (2)
    sec_last_node[2, 3, 4, 3] =  1   # 11 -> 11 carry for substraction (2)
    # last node
    last_node = np.zeros((5, 4, 6, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    last_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    last_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    last_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    last_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: substract 1
    last_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for substraction (2)
    last_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    last_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for substraction (2)
    last_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry (0)
    # 3 from right: add and substract 2
    last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    last_node[2, 0, 3, 0] = -1   # 00 -> 00 carry for substraction (2)
    last_node[2, 1, 3, 1] = -1   # 01 -> 01 carry for substraction (2)
    last_node[2, 2, 3, 2] = -1   # 10 -> 10 carry for substraction (2)
    last_node[2, 3, 3, 3] = -1   # 11 -> 11 carry for substraction (2)
    # 4 from right: add and substract 3
    last_node[1, 0, 4, 1] = 1    # 00 -> 01 carry for addition (1)
    last_node[3, 1, 4, 0] = 1    # 01 -> 00 carry for addition one further (3)
    last_node[1, 2, 4, 3] = 1    # 10 -> 11 carry for addition (1)
    last_node[3, 3, 4, 2] = 1    # 11 -> 10 carry for addition one further (3)
    last_node[4, 0, 4, 1] = -1   # 00 -> 01 carry for substraction one further (4)
    last_node[2, 1, 4, 0] = -1   # 01 -> 00 carry for substraction (2)
    last_node[4, 2, 4, 3] = -1   # 10 -> 11 carry for substraction one further (4)
    last_node[2, 3, 4, 2] = -1   # 11 -> 10 carry for substraction (2)
    # 5 from right: add and substract 4
    last_node[3, 0, 5, 0] = 1    # 00 -> 00 carry for addition one further (3)
    last_node[3, 1, 5, 1] = 1    # 01 -> 01 carry for addition one further (3)
    last_node[3, 2, 5, 2] = 1    # 10 -> 10 carry for addition one further (3)
    last_node[3, 3, 5, 3] = 1    # 11 -> 11 carry for addition one further (3)
    last_node[4, 0, 5, 0] = -1   # 00 -> 00 carry for substraction one further (4)
    last_node[4, 1, 5, 1] = -1   # 01 -> 01 carry for substraction one further (4)
    last_node[4, 2, 5, 2] = -1   # 10 -> 10 carry for substraction one further (4)
    last_node[4, 3, 5, 3] = -1   # 11 -> 11 carry for substraction one further (4)
    # right terminator
    right_terminator = 1/dx*np.array([0, -4/5, 4/5, 1/5, -4/105, 1/280], dtype='float64').reshape(6, 1) # coefficients

    # Define arrays for MPO
    left_node = np.einsum('ab, burd->aurd', left_terminator, central_node)
    right_node = np.einsum('lurd, re->lued', last_node, right_terminator)
    arrays = [left_node] + [central_node]*(n-3) + [sec_last_node] + [right_node]

    return arrays


def Diff_1_8_x_MPO(n, dx):
    # first order derivative with eighth order precision in x direction
    # tensor stucture:
    #                          |                         |               |
    # left_terminator - (central_node)^(n-2) - second to last node - last node - right_terminator
    #                          |                         |               |
    # left terminator
    left_terminator = np.array([1, 1, 1], dtype='float64').reshape(1, 3)
    # central node
    central_node = np.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    central_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    central_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    central_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: substract 1
    central_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for substraction (2)
    central_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for substraction (2)
    central_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    central_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # second to last node
    sec_last_node = np.zeros((3, 4, 5, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    sec_last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    sec_last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    sec_last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    sec_last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    sec_last_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    sec_last_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    sec_last_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    sec_last_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: substract 1
    sec_last_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for substraction (2)
    sec_last_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for substraction (2)
    sec_last_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    sec_last_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # 3 from right: identity with carry for addition
    sec_last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    sec_last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    sec_last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    sec_last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    # 4 from right: identity with carry for substraction
    sec_last_node[2, 0, 4, 0] =  1   # 00 -> 00 carry for substraction (2)
    sec_last_node[2, 1, 4, 1] =  1   # 01 -> 01 carry for substraction (2)
    sec_last_node[2, 2, 4, 2] =  1   # 10 -> 10 carry for substraction (2)
    sec_last_node[2, 3, 4, 3] =  1   # 11 -> 11 carry for substraction (2)
    # last node
    last_node = np.zeros((5, 4, 6, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    last_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    last_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    last_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    last_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: substract 1
    last_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for substraction (2)
    last_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for substraction (2)
    last_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    last_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # 3 from right: add and substract 2
    last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    last_node[2, 0, 3, 0] = -1   # 00 -> 00 carry for substraction (2)
    last_node[2, 1, 3, 1] = -1   # 01 -> 01 carry for substraction (2)
    last_node[2, 2, 3, 2] = -1   # 10 -> 10 carry for substraction (2)
    last_node[2, 3, 3, 3] = -1   # 11 -> 11 carry for substraction (2)
    # 4 from right: add and substract 3
    last_node[1, 0, 4, 2] = 1    # 00 -> 10 carry for addition (1)
    last_node[1, 1, 4, 3] = 1    # 01 -> 11 carry for addition (1)
    last_node[3, 2, 4, 0] = 1    # 10 -> 00 carry for addition one further (3)
    last_node[3, 3, 4, 1] = 1    # 11 -> 01 carry for addition one further (3)
    last_node[4, 0, 4, 2] = -1   # 00 -> 10 carry for substraction one further (4)
    last_node[4, 1, 4, 3] = -1   # 01 -> 11 carry for substraction one further (4)
    last_node[2, 2, 4, 0] = -1   # 10 -> 00 carry for substraction (2)
    last_node[2, 3, 4, 1] = -1   # 11 -> 01 carry for substraction (2)
    # 5 from right: add and substract 4
    last_node[3, 0, 5, 0] = 1    # 00 -> 00 carry for addition one further (3)
    last_node[3, 1, 5, 1] = 1    # 01 -> 01 carry for addition one further (3)
    last_node[3, 2, 5, 2] = 1    # 10 -> 10 carry for addition one further (3)
    last_node[3, 3, 5, 3] = 1    # 11 -> 11 carry for addition one further (3)
    last_node[4, 0, 5, 0] = -1   # 00 -> 00 carry for substraction one further (4)
    last_node[4, 1, 5, 1] = -1   # 01 -> 01 carry for substraction one further (4)
    last_node[4, 2, 5, 2] = -1   # 10 -> 10 carry for substraction one further (4)
    last_node[4, 3, 5, 3] = -1   # 11 -> 11 carry for substraction one further (4)
    # right terminator
    right_terminator = 1/dx*np.array([0, -4/5, 4/5, 1/5, -4/105, 1/280], dtype='float64').reshape(6, 1) # coefficients

    # Define arrays for MPO
    left_node = np.einsum('ab, burd->aurd', left_terminator, central_node)
    right_node = np.einsum('lurd, re->lued', last_node, right_terminator)
    arrays = [left_node] + [central_node]*(n-3) + [sec_last_node] + [right_node]

    return arrays


def Diff_2_8_y_MPO(n, dx):
    # second order derivative with eigth order precision in y direction
    # tensor stucture:
    #                          |                         |               |
    # left_terminator - (central_node)^(n-2) - second to last node - last node - right_terminator
    #                          |                         |               |
    # left terminator
    left_terminator = np.array([1, 1, 1], dtype='float64').reshape(1, 3)
    # central node
    central_node = np.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    central_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    central_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    central_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: substract 1
    central_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for substraction (2)
    central_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    central_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for substraction (2)
    central_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry(0)
    # second to last node
    sec_last_node = np.zeros((3, 4, 5, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    sec_last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    sec_last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    sec_last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    sec_last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    sec_last_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    sec_last_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    sec_last_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    sec_last_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: substract 1
    sec_last_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for substraction (2)
    sec_last_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    sec_last_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for substraction (2)
    sec_last_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry (0)
    # 3 from right: identity with carry for addition
    sec_last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    sec_last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    sec_last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    sec_last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    # 4 from right: identity with carry for substraction
    sec_last_node[2, 0, 4, 0] =  1   # 00 -> 00 carry for substraction (2)
    sec_last_node[2, 1, 4, 1] =  1   # 01 -> 01 carry for substraction (2)
    sec_last_node[2, 2, 4, 2] =  1   # 10 -> 10 carry for substraction (2)
    sec_last_node[2, 3, 4, 3] =  1   # 11 -> 11 carry for substraction (2)
    # last node
    last_node = np.zeros((5, 4, 6, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    last_node[0, 0, 1, 1] = 1    # 00 -> 01 no carry (0)
    last_node[1, 1, 1, 0] = 1    # 01 -> 00 carry for addition (1)
    last_node[0, 2, 1, 3] = 1    # 10 -> 11 no carry (0)
    last_node[1, 3, 1, 2] = 1    # 11 -> 10 carry for addition (1)
    # 2 from right: substract 1
    last_node[2, 0, 2, 1] = 1    # 00 -> 01 carry for substraction (2)
    last_node[0, 1, 2, 0] = 1    # 01 -> 00 no carry (0)
    last_node[2, 2, 2, 3] = 1    # 10 -> 11 carry for substraction (2)
    last_node[0, 3, 2, 2] = 1    # 11 -> 10 no carry (0)
    # 3 from right: add and substract 2
    last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    last_node[2, 0, 3, 0] = 1    # 00 -> 00 carry for substraction (2)
    last_node[2, 1, 3, 1] = 1    # 01 -> 01 carry for substraction (2)
    last_node[2, 2, 3, 2] = 1    # 10 -> 10 carry for substraction (2)
    last_node[2, 3, 3, 3] = 1    # 11 -> 11 carry for substraction (2)
    # 4 from right: add and substract 3
    last_node[1, 0, 4, 1] = 1    # 00 -> 01 carry for addition (1)
    last_node[3, 1, 4, 0] = 1    # 01 -> 00 carry for addition one further (3)
    last_node[1, 2, 4, 3] = 1    # 10 -> 11 carry for addition (1)
    last_node[3, 3, 4, 2] = 1    # 11 -> 10 carry for addition one further (3)
    last_node[4, 0, 4, 1] = 1    # 00 -> 01 carry for substraction one further (4)
    last_node[2, 1, 4, 0] = 1    # 01 -> 00 carry for substraction (2)
    last_node[4, 2, 4, 3] = 1    # 10 -> 11 carry for substraction one further (4)
    last_node[2, 3, 4, 2] = 1    # 11 -> 10 carry for substraction (2)
    # 5 from right: add and substract 4
    last_node[3, 0, 5, 0] = 1    # 00 -> 00 carry for addition one further (3)
    last_node[3, 1, 5, 1] = 1    # 01 -> 01 carry for addition one further (3)
    last_node[3, 2, 5, 2] = 1    # 10 -> 10 carry for addition one further (3)
    last_node[3, 3, 5, 3] = 1    # 11 -> 11 carry for addition one further (3)
    last_node[4, 0, 5, 0] = 1    # 00 -> 00 carry for substraction one further (4)
    last_node[4, 1, 5, 1] = 1    # 01 -> 01 carry for substraction one further (4)
    last_node[4, 2, 5, 2] = 1    # 10 -> 10 carry for substraction one further (4)
    last_node[4, 3, 5, 3] = 1    # 11 -> 11 carry for substraction one further (4)
    # right terminator
    right_terminator = 1/dx**2*np.array([-205/72, 8/5, 8/5, -1/5, 8/315, -1/560], dtype='float64').reshape(6, 1) # coefficients

    # Define arrays for MPO
    left_node = np.einsum('ab, burd->aurd', left_terminator, central_node)
    right_node = np.einsum('lurd, re->lued', last_node, right_terminator)
    arrays = [left_node] + [central_node]*(n-3) + [sec_last_node] + [right_node]

    return arrays


def Diff_2_8_x_MPO(n, dx):
    # second order derivative with eighth order precision in x direction
    # tensor stucture:
    #                          |                         |               |
    # left_terminator - (central_node)^(n-2) - second to last node - last node - right_terminator
    #                          |                         |               |
    # left terminator
    left_terminator = np.array([1, 1, 1], dtype='float64').reshape(1, 3)
    # central node
    central_node = np.zeros((3, 4, 3, 4), dtype='float64')   # left, up, right, down
    # 0 from right: identity
    central_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    central_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    central_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    central_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    central_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    central_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    central_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    central_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: substract 1
    central_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for substraction (2)
    central_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for substraction (2)
    central_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    central_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # second to last node
    sec_last_node = np.zeros((3, 4, 5, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    sec_last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    sec_last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    sec_last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    sec_last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    sec_last_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    sec_last_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    sec_last_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    sec_last_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: substract 1
    sec_last_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for substraction (2)
    sec_last_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for substraction (2)
    sec_last_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    sec_last_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # 3 from right: identity with carry for addition
    sec_last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    sec_last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    sec_last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    sec_last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    # 4 from right: identity with carry for substraction
    sec_last_node[2, 0, 4, 0] =  1   # 00 -> 00 carry for substraction (2)
    sec_last_node[2, 1, 4, 1] =  1   # 01 -> 01 carry for substraction (2)
    sec_last_node[2, 2, 4, 2] =  1   # 10 -> 10 carry for substraction (2)
    sec_last_node[2, 3, 4, 3] =  1   # 11 -> 11 carry for substraction (2)
    # last node
    last_node = np.zeros((5, 4, 6, 4), dtype='float64')  # left, up, right, down
    # 0 from right: identity
    last_node[0, 0, 0, 0] = 1    # 00 -> 00 no carry (0)
    last_node[0, 1, 0, 1] = 1    # 01 -> 01 no carry (0)
    last_node[0, 2, 0, 2] = 1    # 10 -> 10 no carry (0)
    last_node[0, 3, 0, 3] = 1    # 11 -> 11 no carry (0)
    # 1 from right: add 1
    last_node[0, 0, 1, 2] = 1    # 00 -> 10 no carry (0)
    last_node[0, 1, 1, 3] = 1    # 01 -> 11 no carry (0)
    last_node[1, 2, 1, 0] = 1    # 10 -> 00 carry for addition (1)
    last_node[1, 3, 1, 1] = 1    # 11 -> 01 carry for addition (1)
    # 2 from right: substract 1
    last_node[2, 0, 2, 2] = 1    # 00 -> 10 carry for substraction (2)
    last_node[2, 1, 2, 3] = 1    # 01 -> 11 carry for substraction (2)
    last_node[0, 2, 2, 0] = 1    # 10 -> 00 no carry (0)
    last_node[0, 3, 2, 1] = 1    # 11 -> 01 no carry (0)
    # 3 from right: add and substract 2
    last_node[1, 0, 3, 0] = 1    # 00 -> 00 carry for addition (1)
    last_node[1, 1, 3, 1] = 1    # 01 -> 01 carry for addition (1)
    last_node[1, 2, 3, 2] = 1    # 10 -> 10 carry for addition (1)
    last_node[1, 3, 3, 3] = 1    # 11 -> 11 carry for addition (1)
    last_node[2, 0, 3, 0] = 1    # 00 -> 00 carry for substraction (2)
    last_node[2, 1, 3, 1] = 1    # 01 -> 01 carry for substraction (2)
    last_node[2, 2, 3, 2] = 1    # 10 -> 10 carry for substraction (2)
    last_node[2, 3, 3, 3] = 1    # 11 -> 11 carry for substraction (2)
    # 4 from right: add and substract 3
    last_node[1, 0, 4, 2] = 1    # 00 -> 10 carry for addition (1)
    last_node[1, 1, 4, 3] = 1    # 01 -> 11 carry for addition (1)
    last_node[3, 2, 4, 0] = 1    # 10 -> 00 carry for addition one further (3)
    last_node[3, 3, 4, 1] = 1    # 11 -> 01 carry for addition one further (3)
    last_node[4, 0, 4, 2] = 1    # 00 -> 10 carry for substraction one further (4)
    last_node[4, 1, 4, 3] = 1    # 01 -> 11 carry for substraction one further (4)
    last_node[2, 2, 4, 0] = 1    # 10 -> 00 carry for substraction (2)
    last_node[2, 3, 4, 1] = 1    # 11 -> 01 carry for substraction (2)
    # 5 from right: add and substract 4
    last_node[3, 0, 5, 0] = 1    # 00 -> 00 carry for addition one further (3)
    last_node[3, 1, 5, 1] = 1    # 01 -> 01 carry for addition one further (3)
    last_node[3, 2, 5, 2] = 1    # 10 -> 10 carry for addition one further (3)
    last_node[3, 3, 5, 3] = 1    # 11 -> 11 carry for addition one further (3)
    last_node[4, 0, 5, 0] = 1    # 00 -> 00 carry for substraction one further (4)
    last_node[4, 1, 5, 1] = 1    # 01 -> 01 carry for substraction one further (4)
    last_node[4, 2, 5, 2] = 1    # 10 -> 10 carry for substraction one further (4)
    last_node[4, 3, 5, 3] = 1    # 11 -> 11 carry for substraction one further (4)
    # right terminator
    right_terminator = 1/dx**2*np.array([-205/72, 8/5, 8/5, -1/5, 8/315, -1/560], dtype='float64').reshape(6, 1) # coefficients

    # Define arrays for MPO
    left_node = np.einsum('ab, burd->aurd', left_terminator, central_node)
    right_node = np.einsum('lurd, re->lued', last_node, right_terminator)
    arrays = [left_node] + [central_node]*(n-3) + [sec_last_node] + [right_node]

    return arrays