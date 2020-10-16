import numpy as np

def addCoordLayers(x):
    new_dims = np.append(np.shape(x)[:-1],2)
    new = np.zeros(new_dims)

    dim1 = np.shape(x)[1]
    for i in range(dim1):
        new[:,i,:,0] = i/dim1
    dim2 = np.shape(x)[2]
    for j in range(dim2):
        new[:,:,j,1] = j/dim2

    return np.concatenate([x,new], axis=-1)

def catCoordLayers(img, x,y):
    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, -1)
    return np.concatenate([img, x, y], axis=-1)

