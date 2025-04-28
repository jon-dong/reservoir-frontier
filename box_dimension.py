# %%
import numpy as np
import matplotlib.pylab as plt
import afbf
import os
import porespy as ps

import utils
def linear_regression(X, Y):
    ##    Computes least squares linear regression: Y = H*X + V

    if X.ndim != 1 or Y.ndim != 1:
        raise ValueError("Both X and Y must be 1D arrays.")
    if X.size != Y.size:
        raise ValueError("X and Y must be the same length.")
    
    # Construct design matrix
    A = np.vstack([X, np.ones_like(X)]).T

    # Solve least squares
    H, V = np.linalg.lstsq(A, Y, rcond=None)[0]
    
    return H, V

def count_boxes(field: np.ndarray, box_size: int):
    #Count how many box_size×box_size (or smaller at the edges) boxes contain at least one True in a 2D boolean array.

    if box_size < 1:
        raise ValueError("box_size must be at least 1")
    H, W = field.shape

    # how many boxes fit (round up to cover the whole image)
    n_rows = int(np.ceil(H / box_size))
    n_cols = int(np.ceil(W / box_size))

    N_boxes = 0
    for i in range(n_rows):
        # compute the y–slice for this row of boxes
        y0 = i * box_size
        y1 = min((i+1) * box_size, H)

        for j in range(n_cols):
            # compute the x–slice for this column of boxes
            x0 = j * box_size
            x1 = min((j+1) * box_size, W)

            # if any point in this sub-array is True, count the box
            if np.any(field[y0:y1, x0:x1]):
                N_boxes += 1

    return N_boxes, box_size  

def count_boxes_through_scales(fractal, scales):
    count_through_scales = []
    int_scales = []
    for scale in scales:
        count, int_scale = count_boxes(fractal, scale)
        count_through_scales.append(count)
        int_scales.append(int_scale)
    return np.array(count_through_scales), np.array(int_scales)

def log_count_for_j_scales(fractal, scales= list(range(1,50))):
    count_through_scales, integer_scales = count_boxes_through_scales(fractal, scales)
    log_count = np.log(count_through_scales)
    log_scales = np.log(integer_scales)
    return log_count, log_scales

def compute_dim(X, min_idx, max_idx):
    log_count, log_scales = log_count_for_j_scales(X)
    H, V = linear_regression(log_scales[min_idx:max_idx], log_count[min_idx:max_idx])

    return -H, log_count, log_scales

def sierpinski_triangle(size=512, n_points=200000):
    """
    Returns a (size × size) boolean array approximating the Sierpinski triangle.
    Uses the chaos game to scatter points.
    """
    # Triangle vertices in pixel coords
    v = np.array([
        [0, 0],
        [size - 1, 0],
        [size // 2, int((size - 1) * np.sqrt(3) / 2)]
    ], dtype=float)
    
    img = np.zeros((size, size), dtype=bool)
    # start at a random point
    x, y = np.random.uniform(0, size), np.random.uniform(0, size)
    
    for _ in range(n_points):
        # pick one of the three maps at random
        vx, vy = v[np.random.randint(0, 3)]
        x, y = (x + vx) / 2, (y + vy) / 2
        img[int(y), int(x)] = True
    
    return img

def sierpinski_carpet(order=4):
    """
    Returns a (3**order × 3**order) boolean array of the Sierpinski carpet.
    """
    size = 3**order
    img = np.ones((size, size), dtype=bool)

    def remove(x0, y0, s):
        # remove the central s/3 × s/3 block
        t = s // 3
        img[y0 + t : y0 + 2*t, x0 + t : x0 + 2*t] = False
        if t > 1:
            for dy in (0, t, 2*t):
                for dx in (0, t, 2*t):
                    if dx == t and dy == t:
                        continue
                    remove(x0 + dx, y0 + dy, t)

    remove(0, 0, size)
    return img

def cantor_1d(order=5):
    """
    Returns a length-(3**order) boolean array for the 1D Cantor set.
    """
    n = 3**order
    arr = np.ones(n, dtype=bool)

    def remove(i0, length):
        t = length // 3
        arr[i0 + t : i0 + 2*t] = False
        if t > 1:
            remove(i0, t)
            remove(i0 + 2*t, t)

    remove(0, n)
    return arr

def cantor_dust(order=5):
    """
    Returns a (3**order × 3**order) boolean array = Cantor set × Cantor set.
    """
    a = cantor_1d(order)
    return a[:, None] & a[None, :]

# %%
folder = '250130stability_frontier_data/'
hist_video= []
images = []
for file in os.listdir(folder):
    img = np.load(folder+file,allow_pickle=True)
    img[img<1e-3]=-1
    img[img>=1e-3]= 1
    images.append(img)
    hist_video.append(img)

fig, axs = plt.subplots(1,len(hist_video))
for idx, img in enumerate(hist_video):
    axs[idx].imshow(img)





# %%
level = 0.0
idx = 2


triangle = sierpinski_triangle(512)
carpet = sierpinski_carpet(6)
dust = cantor_dust(6)

fractals = [triangle, carpet, dust]
h_vals = [1.58, 1.89, 1.26]
fig, axs = plt.subplots(1,3)
for idx in range(len(fractals)):
    axs[idx].imshow(fractals[idx])

# %%


#for threshold in np.logspace(-3, -1, 5):
for idx in range(3):
    #field_of_zeros = hist_video[idx] >= 0

    field_of_zeros = fractals[idx]

    #threshold = 1e-1
    #field_of_zeros = (np.abs(fields[idx]) >= level) & (np.abs(fields[idx]) <= (level +threshold))
    
    signed_field = np.ones(field_of_zeros.shape)
    signed_field[field_of_zeros]*= -1
    edges= utils.extract_edges(signed_field)


    min_idx = 0
    max_idx = -1
    H, log_count, log_scales = compute_dim(field_of_zeros, min_idx, max_idx)
    H_edges, log_count_edges, log_scales_edges = compute_dim(edges, min_idx, max_idx)

    print(f"Slope (H): {H}, Slope edges (H): {H_edges}")#, Intercept (V): {V}")
    dim_spore = utils.estimate_fractal_dimension([signed_field])
    ret_spore_zero = ps.metrics.boxcount(field_of_zeros)
    ret_spore_edge = ps.metrics.boxcount(edges)
    print(f'dim utils : {dim_spore} \t dim  : {np.median(ret_spore_zero.slope)} \t dim edge : {np.median(ret_spore_edge.slope)}')


    fig, axs = plt.subplots(2,2, sharey='row')
    axs[0,0].imshow(field_of_zeros)
    axs[0,1].imshow(edges)
    axs[0,0].axis('off')
    axs[0,1].axis('off')

    axs[1,0].plot(log_scales[min_idx:max_idx], log_count[min_idx:max_idx])
    axs[1,0].plot(np.log(ret_spore_zero.size), np.log(ret_spore_zero.count))

    axs[1,0].set_title(f'dim {H:.3f}, edges {H_edges:.3f}')

    axs[1,1].plot(log_scales_edges[min_idx:max_idx], log_count_edges[min_idx:max_idx])
    axs[1,1].plot(np.log(ret_spore_edge.size), np.log(ret_spore_edge.count))

    axs[1,1].set_title(f'spore {np.median(ret_spore_zero.slope):.3f} edge {np.median(ret_spore_edge.slope):.3f}')





# %%
# %%
from afbf import tbfield
h_list = [0.25,0.5,0.75]
h_vals = [1.0 - h for h in h_list]
fields = []

for h in h_list:
    # Define the field.
    Z = tbfield('fbf')

    # Change the parameter of the Hurst function.
    Z.hurst.ChangeParameters(np.array([[h]]))
    Z.NormalizeModel()

    # Simulate the field.
    z = Z.Simulate(coord =afbf.coordinates(512))
    field = z.values.reshape(z.M) ## M contains shape of image
    field -= np.mean(field)
    fields.append(field)