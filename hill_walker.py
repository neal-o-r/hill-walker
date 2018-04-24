import rasterio
import numpy as np
import matplotlib.pyplot as plt

src = rasterio.open('./data/srtm_35_02.tif')
band = src.read(1)
pt = np.array((52.635449, -6.767769))

def elevation(pt):
        vals = src.index(pt[1], pt[0])
        return band[vals]

def random_pt():
        return (np.random.uniform(src.bounds.bottom, src.bounds.top),
                np.random.uniform(src.bounds.left, src.bounds.right))

def get_pt():
        pt = random_pt()
        if elevation(pt) < 0:
                return get_pt()
        return pt

def grad(pt, eps=1e-3):
        x, y = pt[0], pt[1]
        x_p = elevation((x+eps, y)) / elevation((x-eps, y)) - 1
        y_p = elevation((x, y+eps)) / elevation((x, y-eps)) - 1

        return np.asarray((x_p, y_p))


def descent(f, X=None, lr=0.1, maxiter=1000, eps=1e-5):

        if X is None:
                X = get_pt()

        i = 1
        delta = 1
        path = [list(X) + [elevation(X)]]
        while i < maxiter and delta > eps and elevation(X) >= 5:
                step = grad(X) * lr
                i += 1
                delta = np.abs(grad(X-step)).sum()
                X = X - step
                path.append(list(X) + [elevation(X)])

        return path


def momentum(f, X=None, lr=0.1, mom=0.1, maxiter=1000, eps=0.001):

        if X is None:
                X = get_pt()
        v = np.asarray((0, 0))

        i = 1
        delta = 1
        path = [list(X) + [elevation(X)]]
        while i < maxiter and delta > eps and elevation(X) >= 0:
                v = mom * v + grad(X) * lr
                i += 1
                delta = np.abs(grad(X-v)).sum()
                X = X - v
                path.append(list(X) + [elevation(X)])

        return path

def write(path):

        with open('output.csv', 'w') as f:
                for l in path:
                        f.write(f'{l[0]}, {l[1]}\n')


if __name__ == '__main__':

        path = momentum(elevation, mom=0.95, lr=0.1, X=[51.999447, -9.742744])
        write(path)
        h = [p[-1] if p[-1] > 0 else 0 for p in path]
#        plt.plot(h)
