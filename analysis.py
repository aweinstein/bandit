import os
import numpy as np
import pandas as pd

from sklearn import tree
from sklearn.cluster import KMeans, DBSCAN

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


if __name__ == '__main__':
    pass
