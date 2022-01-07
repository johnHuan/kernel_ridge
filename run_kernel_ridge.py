import csv, os, sys
import numpy as np
from kernel_ridge import KernelRidge
from sklearn import linear_model
from sklearn.kernel_ridge import KernelRidge as KR2

filepath = os.path.dirname(os.path.abspath(__file__))
filepath = filepath.replace('\\', '/')


def calc_mse(y, y_hat):
    return np.nanmean(((y - y_hat) ** 2))


def test_main(filename, C=1.0, kernel_type='linear'):
    # Load data
    data = np.loadtxt('%s/%s' % (filepath, filename), delimiter=',', dtype=np.float32)
    # Split data
    X, y = data[:, 0:-1], data[:, -1].astype(int)
    y = y[np.newaxis, :]
    print(X.shape)
    print(y.shape)

    # fit our model
    model = KernelRidge(kernel_type='gaussian', C=0.1, gamma=5.0)
    model.fit(X, y)
    y_hat = model.predict(x_train=X, x_test=X)
    mse = calc_mse(y, y_hat)  # Calculate accuracy
    print("mse of KRR:\t%.3f" % (mse))

    # fit linear model for test

    ls = linear_model.LinearRegression()
    ls.fit(X, y[0, :])
    y_ls = ls.predict(X)
    mse = calc_mse(y, y_ls)
    print("mse of LS (from sklearn):\t%.3f" % (mse))

    # fit KRR from sklearn for test

    kr2 = KR2(kernel='rbf', gamma=5, alpha=10)
    kr2.fit(X, y[0, :])
    y_krr = kr2.predict(X)
    mse = calc_mse(y, y_krr)
    print("mse of KRR (from sklearn):\t%.3f" % (mse))


if __name__ == '__main__':
    # test_main(filename='small_data/iris-slwc.txt')
    # test_main(filename='small_data/iris-virginica.txt')
    test_main(filename='small_data/iris-virginica.txt')
