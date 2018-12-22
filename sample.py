import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import scipy.io
import time
from skimage.transform import resize
import numpy as np
import cv2
import random


def main():
    # Read the data:
    img_src = mpimg.imread('src_test.jpg')
    img_dst = mpimg.imread('dst_test.jpg')
    # matches = scipy.io.loadmat('matches') #matching points and some outliers
    matches = scipy.io.loadmat('matches_test') #loading perfect matches
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)

    # The parameters for subplot are: number of rows, number of columns, and which subplot you're currently on.
    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(img_src)
    axarr[0].set_title("SRC")
    axarr[1].imshow(img_dst)
    axarr[1].set_title("DST")
    plt.show()

    # Make images the same size, save a ratio to adjust points in original image to their location in new scaled image
    new_shape = img_dst.shape if img_dst.shape[0] < img_src.shape[0] else img_src.shape
    new_src_img = resize(img_src, new_shape)
    new_dst_img = resize(img_dst, new_shape)
    src_ratio = (new_shape[0]/img_src.shape[0], new_shape[1]/img_src.shape[1], 1)
    dst_ratio = (new_shape[0]/img_dst.shape[0], new_shape[1]/img_dst.shape[1], 1)
    # Combine the images along the X axis
    combined_img = np.array(np.hstack((new_src_img, new_dst_img)))

    plt.figure()
    plt.imshow(combined_img)
    plt.show()

    ######## Connect lines ########

    def generator_points(lst_a, lst_b, factor_a, factor_b):
        for k in range(len(lst_a[0])):
            pnt_a = lst_a[0][k]*factor_a[1], lst_a[1][k]*factor_a[0]
            pnt_b = lst_b[0][k]*factor_b[1], lst_b[1][k]*factor_b[0]

            yield (pnt_a, pnt_b)

    gen_pnts = generator_points(match_p_src, match_p_dst, src_ratio, dst_ratio)

    cv_img = combined_img[:, :, ::-1]   # I'm going to work with openCV, to do RGB -> BGR
    cv_img = cv_img.copy()

    for pt_a, pt_b in gen_pnts:
        # the points are x,y while the image is y,x
        # open CV line accepts (x,y)
        pt_a = int(pt_a[0]), int(pt_a[1])
        pt_b = int(pt_b[0]) + new_shape[1], int(pt_b[1])
#        print("{} -> {}".format(str(pt_a), str(pt_b)))
        lineColor = tuple([255 if r == random.randrange(0, 3) else 0 for r in range(3)])
        cv2.line(cv_img, pt_a, pt_b, lineColor, 2)

    combined_img2 = cv_img[:, :, ::-1]  # Restore to RGB
    plt.figure()
    plt.imshow(combined_img2)
    plt.show()

# End


if __name__ == "__main__":
    main()
