import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_blobs


X, Y = make_blobs(n_samples=500, centers=5)
print(X.shape, Y.shape)


plt.figure(0)
plt.grid(True)
plt.scatter(X[:, 0], X[:, 1])
plt.show()


k = 5
colors = ['green', 'red', 'blue', 'yellow', 'orange', 'pink']
clusters = {}
for kx in range(k):
    centre = 10.0 * (2 * np.random.random((X.shape[1],)) - 1)
    points = []
    cluster = {
        'centre': centre,
        'points': points,
        'color': colors[kx]
    }
    clusters[kx] = cluster

print(clusters)



def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))


def plotting(i,clusters):
    plt.figure(i)
    plt.grid(True)
    for kx in range(k):
        pts = clusters[kx]['coords']
        try:
            plt.scatter(pts[:, 0], pts[:, 1], color=clusters[kx]['color'])
        except:
            pass

        center = clusters[kx]['centre']
        plt.scatter(center[0], center[1], color='black', s=100, marker="*")
    plt.show()

def KMeans(X,k,clusters,iterations=10):
    for i in range(iterations):

        #Assigning Points To The Cluster Centres
        for ix in range(X.shape[0]):
            dist = []
            curr_x = X[ix]

            for kx in range(k):
                d = distance(curr_x, clusters[kx]['centre'])
                dist.append(d)

            current_cluster = np.argmin(dist)
            clusters[current_cluster]['points'].append(curr_x)

        for kx in range(k):
            pts = np.array(clusters[kx]['points'])
            clusters[kx]['coords'] = pts

        plotting(i,clusters)

        #Assigning New Centre To Cluster
        for kx in range(k):
            if clusters[kx]['coords'].shape[0] > 0:
                new_center = clusters[kx]['coords'].mean(axis=0)
            else:
                new_center = clusters[kx]['centre']

            clusters[kx]['centre'] = new_center
            clusters[kx]['points'] = []

    return clusters

clusters=KMeans(X,k,clusters)
print(clusters)