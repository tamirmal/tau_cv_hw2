import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as patches
import scipy.io
import time
import numpy as np
import cv2
import random
import PIL


# Compute transformation SRC -> DST
def compute_homography_naive(mp_src, mp_dst):
    # Generator that return the 2 lines in A for each pair
    def gen_pnt_pair_lines(src, dst):
        for k in range(src.shape[1]):
            line_pair = np.array(
                [[src[0][k], src[1][k], 1, 0, 0, 0, -src[0][k]*dst[0][k], -dst[0][k]*src[1][k], -dst[0][k]],
                [0, 0, 0, src[0][k], src[1][k], 1, -dst[1][k]*src[0][k], -dst[1][k]*src[1][k], -dst[1][k]]]
            )
            yield line_pair

    gen_A = gen_pnt_pair_lines(mp_src, mp_dst)
    A = next(gen_A)
    for lines in gen_A:
        A = np.vstack((A, lines))

    print("A :")
    print(A)

    AT_A = np.dot(A.T, A)
    U, s, U_T = np.linalg.svd(AT_A, full_matrices=True)
    H = U[:, 8].reshape(3, 3)

    print("H: ")
    print(H)

    return H


###############################################################################
# UNIT TEST
###############################################################################
def warpTwoImages(img_dst, img_src, H):
    """
    For understanding this warping process,
    Look on this example
    We have : src, dst, H
    We start by applying the H homography to src - and we get all of src coordinates in dst system
    Lets say that there are now negative points in src, the min point is (-Xs, -Ys)
    The geometric meaning is that src location in space is upper&left from dst :

  (-Xs,-Ys)
     __________
    |          |
    |   SRC    |_______
    |__________|      |
            |    DST  |
            |_________|

    From that we need that the combined image DIMs will be (Xmax-Xs, Ymax - Ys).

    For placing the SRC image, its coordinates now consist of negative values, so we need to do translation :
    (-Xs, -Ys) to (0,0) -> done by a translation matrix
    And then place the image in the combined image

    For placing the DST image : its original coordinates are no longer valid since we enlarged the image dimensions
    We can do translation from (0,0) -> (Xs, Ys)
    Or simply locate at the [Xs:DST_W, Ys:DST_H] indices

    In the following code we start by calculating the needed translation and combining it with the given homography
    and then perform the transformation

    """
    # warp img_src to img_dst with homograph H

    ##################
    # Steps are :
    # 1 - 4) Obtain new homography if translation of src image is needed
    # 5) Apply transformation, only src image is displayed
    # 6) Append dst image
    ##################

    # First need to obtain the combined image dimensions
    # 1) generate vectors that will represent the coordinates of the images corners
    h_dst,w_dst = img_dst.shape[:2]
    h_src,w_src = img_src.shape[:2]
    pts_dst = np.float32([[0,0],[0,h_dst],[w_dst,h_dst],[w_dst,0]]).reshape(-1,1,2)
    pts_src = np.float32([[0,0],[0,h_src],[w_src,h_src],[w_src,0]]).reshape(-1,1,2)

    # 2) Apply transformation on the src corners, we get new src image coordinates in the dest coordinate system.
    #    We can get negative coordinates here! as the src image might be located "to the left"/"above" of the dst image
    pts_src_in_dst = cv2.perspectiveTransform(pts_src, H)

    # 3) Check what will be the combined image boundaries
    #    Again - we can get negative values here
    pts = np.concatenate((pts_dst, pts_src_in_dst), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    # 4) If there were negative corner coordinates, we will need to shift the src img there
    #    Can be done by a translation homography, and combine it with the given homography
    #    If there are no negative coordinates then t = [0, 0].
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translate to (-xmin, -ymin)

    # 5) Apply final homography
    result = cv2.warpPerspective(img_src, Ht.dot(H), (xmax-xmin, ymax-ymin))

    if False:
        plt.figure()
        plt.imshow(result)
        plt.show()

    # 6) Append dst image
    result[t[1]:h_dst+t[1],t[0]:w_dst+t[0]] = img_dst

    return result


def main():
    img_src = mpimg.imread('src.jpg')
    img_dst = mpimg.imread('dst.jpg')
    matches = scipy.io.loadmat('matches_perfect') #loading perfect matches
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)

    H = compute_homography_naive(match_p_src, match_p_dst)
    warp_src = cv2.warpPerspective(img_src, H, (img_dst.shape[1], img_dst.shape[0]))
    f, axarr = plt.subplots(2, 2)
    axarr[0][0].imshow(img_src)
    axarr[0][0].set_title("src")
    axarr[0][1].imshow(img_dst)
    axarr[0][1].set_title("dst")
    axarr[1][0].imshow(warp_src)
    axarr[1][0].set_title("warp_src")
    axarr[1][1].imshow(img_dst)
    axarr[1][1].set_title("dast")
    plt.show()

    combined = warpTwoImages(img_dst, img_src, H)
    plt.figure()
    plt.imshow(combined)
    plt.show()

# End of main()


if __name__ == "__main__":
    main()
