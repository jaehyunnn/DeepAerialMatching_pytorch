import numpy as np

def createCheckBoard(I1, I2, n=7):
    assert I1.shape == I2.shape
    if I2.dtype == 'uint8':
        I2 = np.float32(I2/255)
    height, width, channels = I1.shape
    hi, wi = int(height/n), int(width/n)
    outshape = (hi*n, wi*n, channels)

    out_image = np.zeros(outshape, dtype='float32')
    for i in range(n):
        h = hi * i
        h1 = h + hi
        for j in range(n):
            w = wi * j
            w1 = w + wi
            if (i-j)%2 == 0:
                out_image[h:h1, w:w1, :] = I1[h:h1, w:w1, :]
            else:
                out_image[h:h1, w:w1, :] = I2[h:h1, w:w1, :]

    return out_image