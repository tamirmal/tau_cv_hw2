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

#    print("A :")
#    print(A)

    AT_A = np.dot(A.T, A)
    U, s, U_T = np.linalg.svd(AT_A, full_matrices=True)
    H = U[:, 8].reshape(3, 3)

#    print("H: ")
#    print(H)

    return H
# End of compute_homography_naive


def apply_homograpy(pnt, H):
    homog_pnt = np.column_stack((pnt, 1)).T
    dst_pnt = H.dot(homog_pnt)
    return np.array((dst_pnt[0]/dst_pnt[2], dst_pnt[1]/dst_pnt[2]))


def forward_mapping(img_src, img_dst, H):
    # src & dst image corners
    h_dst,w_dst = img_dst.shape[:2]
    h_src,w_src = img_src.shape[:2]
    pts_dst = np.float32([[0,0],[0,h_dst],[w_dst,h_dst],[w_dst,0]]).reshape(-1,1,2)
    pts_src = np.float32([[0,0],[0,h_src],[w_src,h_src],[w_src,0]]).reshape(-1,1,2)

    # 2) Apply transformation on the src corners, we get new src image coordinates in the dest coordinate system.
    #    We can get negative coordinates here! as the src image might be located "to the left"/"above" of the dst image
    pts_src_in_dst = [apply_homograpy(x, H) for x in pts_src]  # Add ,1 to move into homogenous coordinates
    pts_src_in_dst = np.array(pts_src_in_dst).reshape(-1, 1, 2)

    # 3) Check what will be the combined image boundaries
    #    Again - we can get negative values here
    pts = np.concatenate((pts_dst, pts_src_in_dst), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    new_img = np.ndarray(shape=(ymax-ymin, xmax-xmin, 3))
    for y in range(img_src.shape[0]):
        for x in range(img_src.shape[1]):
            dst_pnt = apply_homograpy(np.array((x, y)).reshape(1, 2), H)
            dst_x = int(dst_pnt[0] - xmin + 0.5)
            dst_y = int(dst_pnt[1] - ymin + 0.5)

            # Make sure not crossing boundaries
            dst_x = min(dst_x, new_img.shape[1] - 1)
            dst_y = min(dst_y, new_img.shape[0] - 1)
            dst_x = max(dst_x, 0)
            dst_y = max(dst_y, 0)

            new_img[dst_y, dst_x, :] = img_src[y, x, :]/255

    if False:
        plt.figure()
        plt.imshow(new_img)
        plt.show()

    return new_img
#    print("DONE")


def backward_mapping(img_src, img_dst, H):
    ##########################################
    # Bilinear interpolation
    #
    #   Q12----------------Q22
    #    |                  |
    #    | <---a-->p(x,y)   |
    #    |           |      |
    #    |           b      |
    #    |           |      |
    #   Q11----------------Q21
    #
    #   f(x,y) = (1-a)(1-b)Q11 +
    #                a(1-b)Q21 +
    #                  a(b)Q22 +
    #                (1-a)bQ12
    #
    ##########################################
    def interp_bilinear(x, y, src_img):
        # ensure we dont cross boundaries
        if any([x >= src_img.shape[1] - 1, y >= src_img.shape[0] - 1]):
            return np.array((0, 0, 0)).reshape(1, 1, 3)
        elif any([x < 0, y < 0]):
            return np.array((0, 0, 0)).reshape(1, 1, 3)

        x_q11, y_q11 = int(x), int(y)
        x_q22, y_q22 = int(x+1), int(y+1)
        x_q12, y_q12 = x_q11, y_q22
        x_q21, y_q21 = x_q22, y_q11
        a, b = x - int(x), y-int(y)

        val = (1-a)*(1-b)*src_img[y_q11, x_q11] + a*(1-b)*src_img[y_q21, x_q21] + a*b*src_img[y_q22, x_q22] + (1-a)*b*src_img[y_q12, x_q12]
        return val
    # End()

    # Calc new img dims - like in forward_mapping()
    h_dst,w_dst = img_dst.shape[:2]
    h_src,w_src = img_src.shape[:2]
    pts_dst = np.float32([[0,0],[0,h_dst],[w_dst,h_dst],[w_dst,0]]).reshape(-1,1,2)
    pts_src = np.float32([[0,0],[0,h_src],[w_src,h_src],[w_src,0]]).reshape(-1,1,2)
    pts_src_in_dst = [apply_homograpy(x, H) for x in pts_src]
    pts_src_in_dst = np.array(pts_src_in_dst).reshape(-1, 1, 2)
    pts = np.concatenate((pts_dst, pts_src_in_dst), axis=0)
    [xmin, ymin] = np.int32(pts.min(axis=0).ravel() - 0.5)
    [xmax, ymax] = np.int32(pts.max(axis=0).ravel() + 0.5)

    # If the image size is larger then img_dst,
    # there are negative corners, meaning we need to translate them to 0,0
    t = [-xmin,-ymin]
    Ht = np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]]) # translation : (-xmin,-ymin) -> (0,0)
    HtH = Ht.dot(H)
    H_inv = np.linalg.inv(HtH)
    # corners in new system
    pts_src_in_dst = [apply_homograpy(x, HtH) for x in pts_src]
    pts_src_in_dst = np.array(pts_src_in_dst).reshape(-1, 1, 2)
#    print(pts_src_in_dst)
    [src_x_min, src_y_min] = np.int32(pts_src_in_dst.min(axis = 0).ravel())
    [src_x_max, src_y_max] = np.int32(pts_src_in_dst.max(axis = 0).ravel())

    new_img = np.ndarray(shape=(ymax - ymin, xmax - xmin, 3))

    for y in range(src_y_min, src_y_max):
        for x in range(src_x_min, src_x_max):
            # backward mapping from dest to src
            dst_x, dst_y = apply_homograpy(np.array((x, y)).reshape(1, 2), H_inv)
            # If needed - do interpolation
            # (can perform interp for both cases but its slower)
            if dst_y > int(dst_y) or dst_x > int(dst_x):
                new_img[y, x, :] = interp_bilinear(dst_x, dst_y, img_src)/255
            else:
                new_img[y, x, :] = img_src[dst_y, dst_x, :]/255

    return new_img, t


def test_homography_full(H, mp_src, mp_dst, max_err):
    pts_src = np.array(mp_src).T.reshape(-1, 1, 2)
    pts_dst = np.array(mp_dst).T.reshape(-1, 1, 2)

    pts_src_in_dst = cv2.perspectiveTransform(pts_src, H)

    pts_src_in_dst = pts_src_in_dst.reshape(-1, 2)
    pts_dst = pts_dst.reshape(-1, 2)
    dist = np.linalg.norm(pts_src_in_dst - pts_dst, axis=1)

    fit_percent = sum(dist < max_err)/len(dist)

    valid_norm = dist[dist < max_err]
    valid_norm_idx = np.argwhere(dist < max_err)
    avg_mse = sum(valid_norm)/len(valid_norm)

    return fit_percent, avg_mse, valid_norm_idx


def test_homography(H, mp_src, mp_dst, max_err):
    fit_percent, avg_mse, _ = test_homography_full(H, mp_src, mp_dst, max_err)
    return fit_percent, avg_mse
# End of test_homography()


def compute_homography(mp_src, mp_dst, inliers_percent, max_err):
    # Experssion from lecture
    # Needed_iters = log(1-p)/log(1-w^n)
    w = inliers_percent
    P = 0.95
    MAX_ITERS = np.log(1 - P) / np.log(1 - w**4)
    MAX_ITERS = int(MAX_ITERS + 0.5)

    best_result = {
        'inliners_num': 0,
        'inliners_set': {
            'src': None,
            'dst': None
        }
    }

    indices = np.arange(len(mp_src[0]))

    for iter in range(MAX_ITERS):
        np.random.shuffle(indices)  # TODO Should I only shuffle this once ?
        pts_to_use = indices[:4]
        src_pts = mp_src[:, pts_to_use]
        dst_pts = mp_dst[:, pts_to_use]
        H = compute_homography_naive(src_pts, dst_pts)  # Compute with minimal needed points (n=4)
        fit, mse, valid_idxs = test_homography_full(H, mp_src, mp_dst, max_err)  # Test over full points dataset
        # Check if this is a new best result - if so, store it
        inliers_num = int(len(indices)*fit)
        if inliers_num > best_result['inliners_num']:
            best_result['inliners_num'] = inliers_num
            best_result['inliners_set']['src'] = mp_src[:, valid_idxs].reshape(2, -1)
            best_result['inliners_set']['dst'] = mp_dst[:, valid_idxs].reshape(2, -1)

    # Compute homography with the largest set of inliners found
    H = compute_homography_naive(best_result['inliners_set']['src'],  best_result['inliners_set']['dst'])
    return H


def panorama(img_src, img_dst, mp_src, mp_dst, inliers_percent, max_err):
    h_dst,w_dst = img_dst.shape[:2]
    H = compute_homography(mp_src, mp_dst, inliers_percent, max_err)
    pano_img, t = backward_mapping(img_src, img_dst, H)
    pano_img[t[1]:h_dst + t[1], t[0]:w_dst + t[0]] = img_dst/255

    return pano_img


###############################################################################
# UNIT TEST
###############################################################################
def warpTwoImages(img_dst, img_src, H, visualize = False):
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

    if visualize:
        plt.figure()
        plt.imshow(result)
        plt.xlabel("warpTwoImages: src image BEFORE appending of dst")
        plt.show()

    # 6) Append dst image
    result[t[1]:h_dst+t[1],t[0]:w_dst+t[0]] = img_dst

    return result


def main():
    img_src = mpimg.imread('src.jpg')
    img_dst = mpimg.imread('dst.jpg')

    ######################################
    # Part A
    ######################################
    print("################# PART A #################")
    # q2:5
    print("Matches_Perfect")
    visualize = False

    matches = scipy.io.loadmat('matches_perfect') #loading perfect matches
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)

    H = compute_homography_naive(match_p_src, match_p_dst)
    print(H)

    if visualize:
        print("Using openCV cv2.warpPerspective")
        cv2_warp_src = cv2.warpPerspective(img_src, H, (img_dst.shape[1], img_dst.shape[0]))
        print("Using HW forward Mapping")
        hw_warp_src = forward_mapping(img_src, img_dst, H)
        f, axarr = plt.subplots(2)
        axarr[0].imshow(cv2_warp_src)
        axarr[0].set_title("cv2 warp src")
        axarr[1].imshow(hw_warp_src)
        axarr[1].set_title("HW FWD Mapping")
        plt.show()

    if visualize:
        f, axarr = plt.subplots(2, 2)
        axarr[0][0].imshow(img_src)
        axarr[0][0].set_title("src")
        axarr[0][1].imshow(img_dst)
        axarr[0][1].set_title("dst")
        axarr[1][0].imshow(hw_warp_src)
        axarr[1][0].set_title("hw_warp_src")
        axarr[1][1].imshow(img_dst)
        axarr[1][1].set_title("dst")
        plt.show()

    """
    if False:  # This belongs to Part C
        combined = warpTwoImages(img_dst, img_src, H, visualize)
        if visualize:
            plt.figure()
            plt.imshow(combined)
            plt.xlabel("combined")
            plt.show()
    """

    # q6
    print("Matches")
    visualize = False

    matches = scipy.io.loadmat('matches')  # loading matches
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)

    H = compute_homography_naive(match_p_src, match_p_dst)
    print(H)

    if visualize:
        print("Using openCV cv2.warpPerspective")
        cv2_warp_src = cv2.warpPerspective(img_src, H, (img_dst.shape[1], img_dst.shape[0]))
        print("Using HW forward Mapping")
        hw_warp_src = forward_mapping(img_src, img_dst, H)
        f, axarr = plt.subplots(2)
        axarr[0].imshow(cv2_warp_src)
        axarr[0].set_title("cv2 warp src")
        axarr[1].imshow(hw_warp_src)
        axarr[1].set_title("HW FWD Mapping")
        plt.show()

    if visualize:
        f, axarr = plt.subplots(2, 2)
        axarr[0][0].imshow(img_src)
        axarr[0][0].set_title("src")
        axarr[0][1].imshow(img_dst)
        axarr[0][1].set_title("dst")
        axarr[1][0].imshow(hw_warp_src)
        axarr[1][0].set_title("hw_warp_src")
        axarr[1][1].imshow(img_dst)
        axarr[1][1].set_title("dst")
        plt.show()

    """
    if False:  # This belongs to Part C
        combined = warpTwoImages(img_dst, img_src, H, visualize)
        if visualize:
            plt.figure()
            plt.imshow(combined)
            plt.xlabel("combined")
            plt.show()
    """

    ######################################
    # Part B
    ######################################
    print("################# PART B #################")
#    src_shape = img_src.shape[:2]
#    dst_shape = img_dst.shape[:2]
#    min_dim = min(dst_shape + src_shape)
#    min_err = min_dim*0.05
    min_err = 25
    inliers_percent = 0.8

    # q7
    print("Test homography for matches_perfect")
    matches = scipy.io.loadmat('matches_perfect')
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)
    H = compute_homography_naive(match_p_src, match_p_dst)
    print(H)
    fit, mse = test_homography(H, match_p_src, match_p_dst, min_err)
    print("Fit percent = {}, mse = {}".format(fit, mse))

    print("Test homography for matches")
    matches = scipy.io.loadmat('matches')
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)
    H = compute_homography_naive(match_p_src, match_p_dst)
    print(H)
    fit, mse = test_homography(H, match_p_src, match_p_dst, min_err)
    print("Fit percent = {}, mse = {}".format(fit, mse))

    # q8
    print("Compute homography with RANSAC for matches")
    visualize = False
    matches = scipy.io.loadmat('matches')
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)
    H = compute_homography(match_p_src, match_p_dst, inliers_percent, min_err)
    print(H)
    fit, mse = test_homography(H, match_p_src, match_p_dst, min_err)
    print("Fit percent = {}, mse = {}".format(fit, mse))

    if visualize:
        print("Using openCV cv2.warpPerspective")
        cv2_warp_src = cv2.warpPerspective(img_src, H, (img_dst.shape[1], img_dst.shape[0]))
        print("Using HW forward Mapping")
        hw_warp_src = forward_mapping(img_src, img_dst, H)
        f, axarr = plt.subplots(2)
        axarr[0].imshow(cv2_warp_src)
        axarr[0].set_title("cv2 warp src")
        axarr[1].imshow(hw_warp_src)
        axarr[1].set_title("HW FWD Mapping")
        plt.show()

    ######################################
    # Part C
    ######################################
    print("################# PART C #################")
    print("backward mapping with bilinear interp")
    visualize = False
    if visualize:
        hw_warp_src, _ = backward_mapping(img_src, img_dst, H)
        plt.figure()
        plt.imshow(hw_warp_src)
        plt.title("Backward Mapping with bilinear interp")
        plt.show()

    print("panoarama - using matches.mat")
    print("Test homography for matches")
    matches = scipy.io.loadmat('matches')
    match_p_dst = matches['match_p_dst'].astype(float)
    match_p_src = matches['match_p_src'].astype(float)
    max_err = 25
    inliers_percent = 0.8
    visualize = False
    if visualize:
        pano_img = panorama(img_src, img_dst, match_p_src, match_p_dst, inliers_percent, max_err)
        plt.figure()
        plt.imshow(pano_img)
        plt.title("PANORAMA")
        plt.show()



# End of main()


if __name__ == "__main__":
    main()
