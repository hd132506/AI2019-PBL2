from sklearn import metrics
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler

def flatten(images):
    return images.reshape((-1, 28*28))

def evaluation(prediction, goal):
    print("Classification report\n%s\n"
        % (metrics.classification_report(goal, prediction, digits=3)))
    print("Confusion matrix:\n%s" 
        % metrics.confusion_matrix(goal, prediction))

def scale(dataset):
    # ztransform = lambda arr : (arr - np.mean(arr))/np.std(arr)
    scaler = StandardScaler()
    # scaler = RobustScaler()
    # scaler = MinMaxScaler()

    scaler.fit(dataset)
    return scaler.transform(dataset)

def binarization(dataset):
    ret = []
    for data in dataset:
        threshold = np.mean(data)
        ret.append(np.where(data < threshold, 0, 255))
    return np.array(ret)



def hog(img, cell_size=4, block_size=2, orientations=8):
    """
    Histogram of Gradient operation
    Using square-shaped cell and block
    img: 2D numpy array
    cell_size: the number of pixels in one side of a cell
    block_size: the number of cells in one side of a block
    orientations: the number of unsigned orientations
    """
    # Get gradients
    y_gradient = np.zeros(img.shape)
    x_gradient = np.zeros(img.shape)
    y_gradient[1:-1, :] = img[2:, :] - img[:-2, :]
    x_gradient[:, 1:-1] = img[:, 2:] - img[:, :-2]

    # Compute magnitude and orientation matrice
    magnitude_mat = np.sqrt(x_gradient**2 + y_gradient**2)
    orientation_mat = np.rad2deg(np.arctan(y_gradient/(x_gradient + 10**(-8)))) % 180

    # Quantitize orientations
    ori_arr = np.array([(i*180/orientations) for i in range(orientations)])
    quant = lambda d: np.argmin(np.minimum(np.abs(d - ori_arr), 180 - np.abs(d - ori_arr)))

    # For transforming pixel position to cell position
    cell_pos = lambda k: k//cell_size

    # Make orientation histogram for each cell
    cell_hist = np.zeros((cell_pos(img.shape[0]), cell_pos(img.shape[1]), orientations))
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            cell_hist[cell_pos(y), cell_pos(x), quant(orientation_mat[y, x])] \
            += magnitude_mat[y, x]
    cell_hist /= cell_size**2

    n_cells_y = cell_pos(img.shape[0])
    n_cells_x = cell_pos(img.shape[1])

    # Block normalization
    n_blocks_y = (n_cells_y - block_size) + 1
    n_blocks_x = (n_cells_x - block_size) + 1

    
    block_normalize = lambda block: \
                    block / np.sqrt(np.sum(block ** 2) + 1e-12) # L2 Norm

    normalized_blocks = np.zeros((n_blocks_y*n_blocks_x, block_size, block_size, orientations))

    
    for y in range(n_blocks_y):
        for x in range(n_blocks_x):
            normalized_blocks[y*n_blocks_x + x, :] = block_normalize(cell_hist[y:y+block_size, x:x+block_size, :])
    
    return normalized_blocks.reshape(-1)

def hog_descriptors(images):
    new_desc = []

    for i in images:
        new_desc.append(hog(i))
    return new_desc
