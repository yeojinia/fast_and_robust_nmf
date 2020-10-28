from scipy.sparse import spdiags
from numpy import linalg as LA
import numpy as np
from scipy.optimize import nnls
import scipy.io as sio
import cv2
from sklearn.metrics import mean_squared_error
from math import sqrt,floor
import matplotlib.pyplot as plt
import gc

def two_norm(M):
    return np.sum(M ** 2, axis=0)

def p_norm(M, p):
    return (np.sum(M ** p, axis=0)) ** (2. / p)

def fastSepNMF(M, r, func_type='a', p = 1, normalize = 0):

    """ This function is implementation of the algorithm 1, "N. Gillis and S.A. Vavasis, Fast and Robust Recursive Algorithms
    % for Separable Nonnegative Matrix Factorization"

        Args:
            M (float numpy.array): Original Matrix
            r (int): low rank 'r'
            func_type (string): 'a'= l_2 norm, 'b' = l_p norm
            p (float): p-value of l_p norm
            normalize: normalize option
        Returns:
            J : Column indices of matrix M
            normM: norm of t he columns of the last residual matrix
            U: normalized extracted columns of the residual.
    """
    m, n =  M.shape
    if normalize == 1:
        D = spdiags( np.transpose(sum(M)**(-1)), 0, n, n)
        M = np.dot(M,D)

    normM = []
    if func_type == 'a':
        normM = two_norm(M)
    elif func_type == 'b':
        normM = p_norm(M, float(p))

    nM = np.max(normM)

    U = np.zeros((m,r))
    J = []
    i = 0
    while i < r and max(normM)/nM > 1e-9:
        a = max(normM) # a: max value
        b = np.amax(normM) # find max index

        normM1 = []
        if i == 1 :
            normM1 = normM
        b = np.where((a-normM)/a <= 1e-6)

        if len(b) > 1:
            c = max(normM1[b]) # value
            d = np.argmax(normM1[b])
            b = b[d]

        J.append(b)
        U[:, i] = np.ravel(M[:, b])

        for j in range(i-2):
            U[:, i] = U[:, i] - U[:, j]*(np.transpose(U[:, j])*U[:,i])

        U[:, i] = U[:, i] / LA.norm(U[:, i])

        normM = normM - (np.dot(np.transpose(U[:, i]),M))**2
        i = i + 1

    return(np.squeeze(J), normM, U)

if __name__ == '__main__':

    print('Select a Function for Fast & Robust Recursive Algorithm: \'a\' (l_2-norm) or \'b\' (l_p-norm)')
    func_type = input()
    p = 2
    if func_type == 'a':
        print('You selected, ' + func_type)
    elif func_type == 'b':
        print('You selected, ' + func_type + '. Please give an input \"p\" (1.25 < p < 2.1) \n')
        p = input()
        if float(p) < 1.25 or float(p) > 2.1:
            print('The range of p is not allowed. We will use p = 2 by default ')
            p = 2.
        else:
            print('You selected,  ' + p )
            p = float(p)
    else:
        print('You gave a wrong input. We will use l_2-norm by default.' + func_type)
        func_type = 'a'

    r = [16, 25, 36, 49, 64]
    colors = ['green','red','blue','magenta','cyan']
    test_list = ['test0', 'test1', 'test2', 'test3', 'test4', 'test5', 'test6', 'test7', 'test8', 'test9']
    labels = ['0', '1', '2', '3', '4', '5','6', '7', '8', '9']
    pos_x = np.linspace(0, 9, 10)

    bar_width = 0.25

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.set_title(' normalized root mean square error')
    ax.set_xlabel(' test number(#) ')
    ax.set_ylabel(' NRMSE ')

    index = 0
    for r_ in r:
        nrmse = []
        for element in test_list:
            mat_contents = sio.loadmat('mnist_all.mat')
            np_arr0 = np.array(mat_contents[element].astype(float))
            [m, n ] = np_arr0.shape
            cv2.imwrite('original_'+element+'.jpg', np_arr0[1, :].reshape(28, 28))

            [J_, norm_, U_] = fastSepNMF(np_arr0, r_, func_type , p, normalize=0)

            C = (np_arr0[:, J_])
            d = np_arr0
            x = []
            for i in range(0, n):
                x.append(nnls(C, d[:,i])[0])

            X = np.array(x)
            M_ = np.matmul(C, np.transpose(X))
            print(element," M shape ", M_.shape)

            M = M_[1, :].reshape(28,28)

            cv2.imwrite('reconstructed_'+element + "_"+ str(r_) + '.jpg',M)

            # RMSE normalised by mean
            nrmse.append(sqrt(mean_squared_error(np_arr0, M_)) / (np.max(np_arr0) - np.min(np_arr0)))
        ax.errorbar(pos_x, np.array(nrmse), marker='o', ms = 8, linestyle='dotted', color=colors[index], label = "r = "+str(r_))
        index = index +1

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    plt.show()

