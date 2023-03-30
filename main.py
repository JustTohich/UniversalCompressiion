import cv2
import numpy as np
from numpy.linalg import inv


tile_size = 16
im_shape = []


def read_image(filename):
    return cv2.imread(filename)


def write_image(im, filename):
    cv2.imwrite(filename, im)


def rgb2ycrcb(im):
    return cv2.cvtColor(np.array([im]), cv2.COLOR_BGR2YCR_CB)[0]


def ycrcb2rgb(im):
    return cv2.cvtColor(np.array([im]), cv2.COLOR_YCR_CB2BGR)[0]


def enc_tiling(im):
    global im_shape
    im_shape = im.shape[:]

    print(im_shape, tile_size)

    tiles = []
    c = 0

    for i in range(0, im.shape[0] - tile_size, tile_size):
        for j in range(0, im.shape[1] - tile_size, tile_size):
            tiles.append(im[i:i + tile_size, j:j + tile_size, :].reshape((tile_size * tile_size, -1)))
            c += 1
    return tiles


def dec_tiling(tiles):
    im = np.zeros(im_shape)

    c = 0
    for i in range(0, im.shape[0] - tile_size, tile_size):
        for j in range(0, im.shape[1] - tile_size, tile_size):
            im[i:i + tile_size, j:j + tile_size, :] = tiles[c].reshape((tile_size, tile_size, -1))
            c += 1
    return im


A = np.eye(tile_size * tile_size)
A_1 = np.eye(tile_size * tile_size)


def teach_a(im):
    global A
    optim_vector = np.zeros_like(A)
    A += optim_vector


def teach_stop_a():
    global A_1
    A_1 = inv(A)


def enc_a(im):
    return im.T.dot(A).astype(dtype=im.dtype)


def dec_a(im):
    return (im.dot(A_1)).T.astype(dtype=im.dtype)


def enc_ac(im):
    return im


def dec_ac(im):
    return im


def teach(filename):
    im = read_image(filename)
    for tile in enc_tiling(im):
        teach_a(rgb2ycrcb(tile))
    teach_stop_a()


def test(filename):
    im = read_image(filename)
    print("Img size", im.shape[0] * im.shape[1] * im.shape[2])
    tiles = enc_tiling(im)
    enc_size = 0
    dec_tiles = []
    for tile in tiles:
        tile = rgb2ycrcb(tile)
        tile = enc_a(tile)
        tile = enc_ac(tile)
        enc_size += tile.shape[0] * tile.shape[1]
        tile = dec_ac(tile)
        tile = dec_a(tile)
        tile = ycrcb2rgb(tile)
        dec_tiles.append(tile)
    print("Enc size", enc_size)
    dec_im = dec_tiling(dec_tiles)
    write_image(dec_im, filename[:-4] + ".dec.bmp")


if __name__ == '__main__':
    teach("Lenna.png")
    test("Lenna.png")
