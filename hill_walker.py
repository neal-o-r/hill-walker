import rasterio
import numpy as np
from scipy.optimize import approx_fprime
from functools import partial

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

def grad(pt, eps=1e-2):
        elev1 = elevation((pt[0] + eps, pt[1]))
        elev2 = elevation((pt[0] - eps, pt[1]))
        elev3 = elevation((pt[0], pt[1] + eps))
        elev4 = elevation((pt[0], pt[1] - eps))

        lat_slope = elev1 / elev2
        lon_slope = elev3 / elev4

        return np.asarray((lat_slope, lon_slope))

def descent(f, X=None, lr=0.1, maxiter=1000, eps=0.01):

        if X is None:
                X = get_pt()

        i = 1
        delta = 1
        path = [list(X) + [elevation(X)]]
        while i < maxiter and delta > eps and elevation(X) >= 0:
                step = grad(X) * lr
                i += 1
                delta = np.abs(grad(X-step)).sum()
                X = X - step
                path.append(list(X) + [elevation(X)])

        return path


def momentum(f, X=None, lr=0.1, mom=0.1, maxiter=1000, eps=0.01):

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

        path = momentum(elevation, X=pt, lr=0.001, mom=0.5)
        write(path)

