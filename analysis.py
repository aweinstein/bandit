import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import tree
from sklearn.cluster import KMeans, DBSCAN
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

Fig_Dir = 'figs'
DF_Dir = 'df'

def tree_classifier():
    """Create an HPS classifier using the alpha-beta."""
    fn_fit = os.path.join(DF_Dir, 'fit_constant_step_size_01_bounded.pkl')
    fit = pd.read_pickle(fn_fit)
    print('Using data from', fn_fit)
    X = fit[['0_alpha', '0_beta', '1_alpha', '1_beta']].values
    y = fit['HPS_level'].values

    clf = tree.DecisionTreeClassifier(max_depth=4)
    clf.fit(X, y)
    from sklearn.externals.six import StringIO
    import pydotplus as pydot
    dot_data = StringIO()
    feature_names = ['a0', 'b0', 'a1', 'b1']
    target_names = ['low', 'medium', 'high']
    tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=feature_names,
                         class_names=target_names,
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())
    fn = os.path.join(Fig_Dir, 'hpf_tree_classifier.pdf')
    graph.write_pdf(fn)
    print('Tree saved as', fn)


def clustering():
    fn_fit = os.path.join(DF_Dir, 'fit_constant_step_size_01_bounded.pkl')
    fit = pd.read_pickle(fn_fit)
    print('Using data from', fn_fit)
    X = fit[['0_alpha', '0_beta', '1_alpha', '1_beta']].values
    y = fit['HPS_level'].values


    # SSD = []
    # ns = np.arange(2, 40)
    # for n in ns:
    #     estimator = KMeans(n_clusters=n)
    #     estimator.fit(X)
    #     SSD.append(estimator.inertia_)

    # plt.close('all')
    # plt.plot(ns, SSD, 'o-')
    # plt.show()

    #db = DBSCAN(eps=0.3, min_samples=10).fit(X)
    db = DBSCAN().fit(X)
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    print(n_clusters_)


# Probar con RANSAC
# Probar con ridge regression, lasso regression
# Theil-Sen
# Polinomial features plus lasso

def regression():

    fn_fit = os.path.join(DF_Dir, 'fit_constant_step_size_01_bounded.pkl')
    fit = pd.read_pickle(fn_fit)
    print('Using data from', fn_fit)
    X = fit[['0_alpha', '0_beta', '1_alpha', '1_beta']].values
    y = fit['HPS'].values

    reg = linear_model.LinearRegression()
    reg.fit(X, y)
    y_hat = reg.predict(X)

    model_ransac = linear_model.RANSACRegressor(linear_model.LinearRegression())
    model_ransac.fit(X, y)
    y_hat_ransac = model_ransac.predict(X)

    model_theilsen = linear_model.TheilSenRegressor()
    model_theilsen.fit(X,y)
    y_hat_theilsen = model_theilsen.predict(X)

    mse_lr = mean_squared_error(y, y_hat)
    mse_ransac = mean_squared_error(y, y_hat_ransac)
    mse_theil_ransac = mean_squared_error(y, y_hat_theilsen)

    i_sort = np.argsort(y)
    plt.close('all')
    plt.plot(y[i_sort], 'o', label='y')
    plt.plot(y_hat[i_sort], 'x', label='y_hat')
    plt.plot(y_hat_ransac[i_sort], '+', label='y ransac')
    plt.plot(y_hat_theilsen[i_sort], '<', label='y thiel')
    plt.legend(loc='best')
    plt.show()

    globals().update(locals())



if __name__ == '__main__':
    regression()
