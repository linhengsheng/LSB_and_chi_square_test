import cv2
import numpy as np

"""
    for a pixels array G, point out a number belong to R
    definition of the smoothness function:
        f = |x2-x1| + |x3-x2| + |x4-x3| + ... + |xn-xn-1|
        describe the space correlation of local image. 
        the smaller f is, the smoother the adjacent pixels of the image will be.
        so the space correlation of image will be stronger.
        
    usually, if the image was written by message, the noise will grow up.
    so f turn larger
    
    flip operation:
        -positive masked operator
            F_lsb = 0<->1 2<->3 ... 254<->255 = x+1-2(x mod 2)
        -negative masked operator
            F_lsb_-1 = -1<->0 1<->2 ... 255<->256
        self replacement(do nothing)
            F_0 = x

    Define three kinds of pixels:
        -f(F(G)) > f(G): Regular group
        -f(F(G)) < f(G): Singular group
        -f(F(G)) = f(G): Unchanged group

        where F(G) = F_LSB_-1(p for p in pixels)

    for F_lsb, define a positive masked operator M, which is an n bits group of {0, 1}.
        FM(G) = FM1(x1), FM2(x2), ..., FM(xn)

    for F_lsb_-1, define a negative masked operator -M, which is an n bits group of {0, -1}
        F-M(G) = F-M1(x1), F-M2(x2), ..., F-M(xn)
    
    position of 0 and 1 in M or -M can be random, but the percentage of each(0 or 1) must be 5%
    define
        RM = regular group's percentage in FM
        SM = singular group's percentage in FM
        
        R-M = regular group's percentage in F-M
        S-MM = singular group's percentage in F-M
    so:
        RM + SM <= 1, R-M + S-M <= 1
"""


def positive_masked(data):
    return np.where(data % 2 == 0, data + 1, data - 1)


def negative_masked(data):
    return np.where(data % 2 == 0, data - 1, data + 1)


def smooth_func(data):
    minuend = data.flatten()[1:]
    subtrahend = data.flatten()[:-1]
    return np.sum(np.abs(minuend - subtrahend))


def image2patches(data: np.ndarray, patch_size=2):
    h, w, c = data.shape
    rows = h // patch_size
    cols = w // patch_size
    ret = np.empty((rows, cols, patch_size, patch_size, c), dtype=data.dtype)
    for x in range(rows):
        for y in range(cols):
            ret[x, y] = data[x * patch_size: (x + 1) * patch_size, y * patch_size: (y + 1) * patch_size]
    return ret

def calculate_correlation(data: np.ndarray):
    # initialization
    """
        RM = regular group's percentage in FM
        SM = singular group's percentage in FM
    """
    positive_RM = 0
    positive_SM = 0
    negative_RM = 0
    negative_SM = 0
    """
        -f(F(G)) > f(G): Regular group
        -f(F(G)) < f(G): Singular group
        -f(F(G)) = f(G): Unchanged group
    """
    patches = image2patches(data)
    for row_data in patches:
        for patch in row_data:
            # col_data: patch_size, patch_size, channels
            # mask
            positive_masked_data = positive_masked(patch)
            negative_masked_data = negative_masked(patch)

            positive_smoothness = smooth_func(positive_masked_data)
            original_smoothness = smooth_func(patch)
            negative_smoothness = smooth_func(negative_masked_data)

            if positive_smoothness > original_smoothness:
                positive_RM += 1
            else:
                positive_SM += 1
            if negative_smoothness > original_smoothness:
                negative_RM += 1
            else:
                negative_SM += 1
    return positive_RM, positive_SM, negative_RM, negative_SM

def rs_analysis(image_path):
    """
        split the pixels to several groups:
            -each group composed of its adjacent pixels
        calculate the correlation of groups by smoothness function
            -separately calculate after FM and F-M
            -to compare the value of RM, SM, R-M, S-M
        flip the pixel's least significant bit, repeat the second work
        get the results

        usually, if the image hasn't written by LSB
            -RM will similar to R-M,
            -SM will similar to S-M
            -RM > SM, R-M > S-M
        else:
            |RM - SM| will tend to 0 with message rate grow
            while |R-M - S-M| will tend to be larger
        so:
            |R-M - S-M| >> |RM - SM|
    """
    image = cv2.imread(image_path)
    RM0, SM0, _RM0, _SM0 = calculate_correlation(image)
    RM1, SM1, _RM1, _SM1 = calculate_correlation(image ^ 1)
    """
        assume the embedding rate is p,
        statistical experiments show that 
            -RM/SM = a*p**2 + b*p + c
            -R-M/S-M = a*p + bias
    """
    a = 2 * (RM0 - SM0 + RM1 - SM1)
    b = (_RM0 - _SM0) - (_RM1 - _SM1) - (RM1 - SM1) - 3 * (RM0 - SM0)
    c = (RM0 - SM0) - (_RM0 - _SM0)
    roots = np.roots([a, b, c])
    z = roots[np.argmin(np.abs(roots))]
    return z/(z-0.5)

if __name__ == '__main__':
    p = rs_analysis('./images/miku.png')
    print('Original image')
    print('the embedding rate is:', p)

    p = rs_analysis('./secrets/miku.png')
    print('Secret image')
    print('the embedding rate is:', p)
