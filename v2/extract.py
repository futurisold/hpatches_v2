import numpy as np
import cv2
import matplotlib.pyplot as plt

# TODO: wrap everything inside a class

def extract_random_patch(im, patch_size):
    # get the size of the image
    im_h, im_w = im.shape[:2]
    # get the top left corner of the patch
    x = np.random.randint(0, im_w-patch_size)
    y = np.random.randint(0, im_h-patch_size)
    # extract the patch
    patch = im[y: y+patch_size, x: x+patch_size]

    return patch, (x, y)

def get_patch_position_tgt(im, ref_patch_pos, H, patch_size):
    # get the corners of the patch in the target image
    x_min, y_min = ref_patch_pos
    corners = np.array([
        [x_min, y_min],
        [x_min, y_min+patch_size],
        [x_min+patch_size, y_min+patch_size],
        [x_min+patch_size, y_min]],
    ).astype(np.float32)
    # apply the homography to the corners
    corners = cv2.perspectiveTransform(corners[None, ...], H).squeeze()
    # find the bounding box of the transformed patch in the target image
    x_min, y_min = np.min(corners, axis=0).astype(int)
    x_max, y_max = np.max(corners, axis=0).astype(int)

    return im[y_min: y_max, x_min: x_max], (x_min, y_min)


def plot_patches(ref_im, ref_patch, ref_patch_pos, tgt_im, tgt_patch, tgt_patch_pos):
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(ref_im, cmap='gray')
    ax[0].add_patch(plt.Rectangle(ref_patch_pos, *ref_patch.shape[::-1], fill=False, edgecolor='r', linewidth=1))
    ax[1].imshow(tgt_im, cmap='gray')
    ax[1].add_patch(plt.Rectangle(tgt_patch_pos, *tgt_patch.shape[::-1], fill=False, edgecolor='r', linewidth=1))
    plt.show()


if __name__ == '__main__':
    ref_im = cv2.imread('./data/hpatches-sequences-release/v_artisans/1.ppm')
    ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
    H_1_2 = np.loadtxt('./data/hpatches-sequences-release/v_artisans/H_1_2')
    tgt_im = cv2.warpPerspective(ref_im, H_1_2, ref_im.shape[::-1])

    # extract a random patch from the reference image
    ref_patch, ref_patch_pos = extract_random_patch(ref_im, 65)
    # extract the corresponding patch from the target image
    tgt_patch, tgt_patch_pos = get_patch_position_tgt(tgt_im, ref_patch_pos, H_1_2, 65)
    # plot the patches
    plot_patches(ref_im, ref_patch, ref_patch_pos, tgt_im, tgt_patch, tgt_patch_pos)

