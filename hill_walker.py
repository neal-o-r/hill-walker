import rasterio
import numpy as np
import matplotlib.pyplot as plt

src = rasterio.open("./data/srtm_35_02.tif")
band = src.read(1)
pt = np.array((52.635449, -6.767769))


def elevation(pt):
    vals = src.index(pt[1], pt[0])
    return band[vals]



def random_pt():
    return (
        np.random.uniform(src.bounds.bottom, src.bounds.top),
        np.random.uniform(src.bounds.left, src.bounds.right),
    )


def get_pt():
    pt = random_pt()
    if elevation(pt) < 0:
        return get_pt()
    return pt


def grad(f, pt, eps=1e-3):
    x, y = pt[0], pt[1]
    x_p = (f((x + eps, y)) - f((x, y))) / eps
    y_p = (f((x, y + eps)) - f((x, y))) / eps

    return np.asarray((x_p, y_p))


def descent(f, X=None, lr=0.1, maxiter=1000, eps=1e-5, threshold=1e-5):

    if X is None:
        X = get_pt()

    i = 1
    delta = 1
    path = [list(X) + [f(X)]]
    while i < maxiter and delta > eps and f(X) >= threshold:
        step = grad(f, X) * lr
        i += 1
        delta = np.abs(grad(f, X - step)).sum()
        X = X - step
        path.append(list(X) + [f(X)])

    return path


def momentum(f, X=None, lr=0.1, mom=0.1, maxiter=1000, eps=0.001,
             threshold=1e-5):

    if X is None:
        X = get_pt()
    v = np.asarray((0, 0))

    i = 1
    delta = 1
    path = [list(X) + [f(X)]]
    while i < maxiter and delta > eps and f(X) >= threshold:
        v = mom * v + grad(f, X) * lr
        i += 1
        delta = np.abs(grad(f, X - v)).sum()
        X = X - v
        path.append(list(X) + [f(X)])

    return path


def write(path):

    with open("output.csv", "w") as f:
        for l in path:
            f.write(f"{l[0]}, {l[1]}\n")


if __name__ == "__main__":

    X = [52.617,-6.778]
    path = momentum(elevation, mom=0.95, threshold=5, lr=1e-7, X=X)
    write(path)
    h = [p[-1] if p[-1] > 0 else 0 for p in path]
#        plt.plot(h)
