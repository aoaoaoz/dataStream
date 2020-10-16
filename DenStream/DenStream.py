import sys
import numpy as np
from sklearn.utils import check_array
from copy import copy
from MicroCluster import MicroCluster
from math import ceil
from sklearn.cluster import DBSCAN
import time
import logging

online, offline, dbscan_time = 0, 0, 0

def ConfigConsoleLogging():
    fmt = '[%(levelname)1.1s %(asctime)s %(process)d %(name)s %(funcName)s:%(lineno)d] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    logging.basicConfig(stream=sys.stdout, format=fmt, datefmt=datefmt, level=logging.DEBUG)

class DenStream:

    def __init__(self, lambd=1, eps=1, beta=2, mu=2):
        """
        DenStream - Density-Based Clustering over an Evolving Data Stream with
        Noise.

        Parameters
        ----------
        lambd: float, optional
            The forgetting factor. The higher the value of lambda, the lower
            importance of the historical data compared to more recent data.
        eps : float, optional
            The maximum distance between two samples for them to be considered
            as in the same neighborhood.

        Attributes
        ----------
        labels_ : array, shape = [n_samples]
            Cluster labels for each point in the dataset given to fit().
            Noisy samples are given the label -1.

        Notes
        -----


        References
        ----------
        Feng Cao, Martin Estert, Weining Qian, and Aoying Zhou. Density-Based
        Clustering over an Evolving Data Stream with Noise.
        """
        self.lambd = lambd
        self.eps = eps
        self.beta = beta
        self.mu = mu
        self.t = 0
        self.p_micro_clusters = []
        self.o_micro_clusters = []
        if lambd > 0:
            self.tp = ceil((1 / lambd) * np.log((beta * mu) / (beta * mu - 1)))
        else:
            self.tp = sys.maxsize

    def partial_fit(self, X, y=None, sample_weight=None):
        """
        Online learning.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        self : returns an instance of self.
        """

        X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # if not hasattr(self, "potential_micro_clusters"):

        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "data %d." % (n_features, self.coef_.shape[-1]))

        for sample, weight in zip(X, sample_weight):
            self._partial_fit(sample, weight)
        return self

    def fit_predict(self, X, y=None, sample_weight=None):
        """
        Lorem ipsum dolor sit amet

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            Subset of training data

        y : Ignored

        sample_weight : array-like, shape (n_samples,), optional
            Weights applied to individual samples.
            If not provided, uniform weights are assumed.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            Cluster labels
        """

        global online, offline, dbscan_time

        start = time.process_time()

        X = check_array(X, dtype=np.float64, order="C")

        n_samples, _ = X.shape

        sample_weight = self._validate_sample_weight(sample_weight, n_samples)

        # if not hasattr(self, "potential_micro_clusters"):

        # if n_features != :
        # raise ValueError("Number of features %d does not match previous "
        # "data %d." % (n_features, self.coef_.shape[-1]))

        for sample, weight in zip(X, sample_weight):
            self._partial_fit(sample, weight)
        
        online += time.process_time() - start
        onlineEnd = time.process_time()
        
        p_micro_cluster_centers = np.array([p_micro_cluster.center() for
                                            p_micro_cluster in
                                            self.p_micro_clusters])
        if len(p_micro_cluster_centers) == 0:
            offline += time.process_time() - onlineEnd
            return []
        p_micro_cluster_weights = [p_micro_cluster.weight() for p_micro_cluster in
                                   self.p_micro_clusters]

        dbscan_before_time = time.process_time()
        dbscan = DBSCAN(eps=0.8, min_samples=self.mu, algorithm='brute')
        dbscan.fit(p_micro_cluster_centers,
                   sample_weight=p_micro_cluster_weights)
        dbscan_after_time = time.process_time()
        dbscan_time += dbscan_after_time - dbscan_before_time
        
        y = []
        for sample in X:
            index, _ = self._get_nearest_micro_cluster(sample,
                                                       self.p_micro_clusters)
            y.append(dbscan.labels_[index])

        offline += time.process_time() - onlineEnd
        SSQ = 0
        for l in range(1+max(dbscan.labels_)):
            ps = p_micro_cluster_centers[dbscan.labels_[:] == l]
            m = np.mean(ps)
            for p in ps:
                SSQ += sum((p-m) ** 2)
                    
        return y, SSQ

    def _get_nearest_micro_cluster(self, sample, micro_clusters):
        smallest_distance = sys.float_info.max
        nearest_micro_cluster = None
        nearest_micro_cluster_index = -1
        for i, micro_cluster in enumerate(micro_clusters):
            current_distance = np.linalg.norm(micro_cluster.center() - sample)
            if current_distance < smallest_distance:
                smallest_distance = current_distance
                nearest_micro_cluster = micro_cluster
                nearest_micro_cluster_index = i
        return nearest_micro_cluster_index, nearest_micro_cluster

    def _try_merge(self, sample, weight, micro_cluster):
        if micro_cluster is not None:
            micro_cluster_copy = copy(micro_cluster)
            micro_cluster_copy.insert_sample(sample, weight)
            if micro_cluster_copy.radius() <= self.eps:
                micro_cluster.insert_sample(sample, weight)
                return True
        return False

    def _merging(self, sample, weight):
        # Try to merge the sample with its nearest p_micro_cluster
        _, nearest_p_micro_cluster = \
            self._get_nearest_micro_cluster(sample, self.p_micro_clusters)
        success = self._try_merge(sample, weight, nearest_p_micro_cluster)
        if not success:
            # Try to merge the sample into its nearest o_micro_cluster
            index, nearest_o_micro_cluster = \
                self._get_nearest_micro_cluster(sample, self.o_micro_clusters)
            success = self._try_merge(sample, weight, nearest_o_micro_cluster)
            if success:
                if nearest_o_micro_cluster.weight() > self.beta * self.mu:
                    del self.o_micro_clusters[index]
                    self.p_micro_clusters.append(nearest_o_micro_cluster)
            else:
                # Create new o_micro_cluster
                micro_cluster = MicroCluster(self.lambd, self.t)
                micro_cluster.insert_sample(sample, weight)
                self.o_micro_clusters.append(micro_cluster)

    def _decay_function(self, t):
        return 2 ** ((-self.lambd) * (t))

    def _partial_fit(self, sample, weight):
        self._merging(sample, weight)
        if self.t % self.tp == 0:
            self.p_micro_clusters = [p_micro_cluster for p_micro_cluster
                                     in self.p_micro_clusters if
                                     p_micro_cluster.weight() >= self.beta *
                                     self.mu]
            Xis = [((self._decay_function(self.t - o_micro_cluster.creation_time
                                          + self.tp) - 1) /
                    (self._decay_function(self.tp) - 1)) for o_micro_cluster in
                   self.o_micro_clusters]
            self.o_micro_clusters = [o_micro_cluster for Xi, o_micro_cluster in
                                     zip(Xis, self.o_micro_clusters) if
                                     o_micro_cluster.weight() >= Xi]
        self.t += 1

    def _validate_sample_weight(self, sample_weight, n_samples):
        """Set the sample weight array."""
        if sample_weight is None:
            # uniform sample weights
            sample_weight = np.ones(n_samples, dtype=np.float64, order='C')
        else:
            # user-provided array
            sample_weight = np.asarray(sample_weight, dtype=np.float64,
                                       order="C")
        if sample_weight.shape[0] != n_samples:
            raise ValueError("Shapes of X and sample_weight do not match.")
        return sample_weight

def RunData(path=None):

    from sklearn import datasets
    import matplotlib.pyplot as plt
    import numpy

    global online, offline, dbscan_time

    # load data
    print(f"path: {path}")
    data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            now = list(map(float, line.strip().split(',') ))
            data.append(now)
    data = numpy.array(data)

    # ax = plt.subplot()
    # clusterer = DenStream(lambd=0.001, eps=0.4374, beta=0.0319, mu=14.6812)
    clusterer = DenStream(lambd=0.001, eps=0.4374, beta=0.2, mu=14.6812)
    iter = 0
    N = 20
    print(f"{N}; {len(data)//N}")
    for i in range(len(data)//N):
        iter += 1
        y, ssq = clusterer.fit_predict(data[i*N:(i+1)*N])
        # if iter % 100 == 0:
        print(f"{iter} label: {y}; SSQ: {ssq}; online: {online}s; offline: {offline}s; dbscan_time:{dbscan_time}s")
        print(f"{iter} Number of p_micro_clusters is {len(clusterer.p_micro_clusters)}")
        print(f"{iter} Number of o_micro_clusters is {len(clusterer.o_micro_clusters)}")
        print(f"aoaoao\t{(i+1)*N}\t{ssq}\t{online}\t{offline}\t{dbscan_time}")
        
    # for item in clusterer.p_micro_clusters:
    #     ax.scatter(item.center()[0], item.center()[1], marker='v', c='black')
        # print(item.center())
    # plt.show()

RunData(r'I:\数据流聚类算法实现\aaa_norm.csv')
# data = np.random.random([10000, 5]) * 1000
# clusterer = DenStream(lambd=0.1, eps=0.001, beta=0.5, mu=3)
# with open('bbb.csv', 'r') as f:
#     data = f.readlines()
# iter = 0
# for row in data[:100]:
#     iter += 1
#     row = row.strip().split(',')
#     # clusterer.partial_fit([row], 1)
#     print(clusterer.fit_predict([row]))
#     print(f"{iter} Number of p_micro_clusters is {len(clusterer.p_micro_clusters)}")
#     print(f"{iter} Number of o_micro_clusters is {len(clusterer.o_micro_clusters)}")
