import numpy as np
import cv2
import matplotlib.pyplot as plt
from kornia.feature import harris_response, hessian_response, gftt_response
import torch
from sklearn.cluster import KMeans

# TODO: wrap everything inside a class

def extract_random_patch(im, patch_size):
    # get the size of the image
    im_h, im_w = im.shape[:2]
    # get the top left corner of the patch
    x = np.random.randint(0, im_w-patch_size)
    y = np.random.randint(0, im_h-patch_size)
    print(y, x)
    # extract the patch
    patch = im[y: y+patch_size, x: x+patch_size]

    return patch, (x, y)

def dog_response(im, sigma1=1., sigma2=1.6):
    # compute the gaussian blur of the image
    im_blur1 = cv2.GaussianBlur(im.squeeze().numpy(), (0, 0), sigma1)
    im_blur2 = cv2.GaussianBlur(im.squeeze().numpy(), (0, 0), sigma2)
    # compute the difference of gaussian
    dog = im_blur1 - im_blur2

    return dog

def extract_patch(im, patch_pos, patch_size):
    # get the top left corner of the patch
    x, y = patch_pos
    # extract the patch
    patch = im[y: y+patch_size, x: x+patch_size]

    return patch

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

    # discard rectangles that are outside the image
    if x_min < 0 or y_min < 0 or x_max > im.shape[1] or y_max > im.shape[0]:
        return np.array([]), (0, 0)

    # extract the patch
    patch = im[y_min: y_max, x_min: x_max]

    return patch, (x_min, y_min)

def keep_patch_if_valid(tgt_patch):
    # given how the patches are extracted, some of them might be outside the image
    # if this happens, the patch is empty and we discard it
    return tgt_patch.size > 0

def reconstruction_error(ref_patch, tgt_patch, display=False):
    tgt_patch = cv2.resize(tgt_patch, ref_patch.shape[::-1]) # resize the target patch to the reference patch size
    # plot the patches
    if display:
        plt.imshow(np.hstack([ref_patch, tgt_patch]), cmap='gray')
        plt.show()

    return np.mean(np.abs(ref_patch - tgt_patch))

def plot_patches(ref_im, ref_patch, ref_patch_pos, tgt_im, tgt_patch, tgt_patch_pos):
    _, ax = plt.subplots(1, 2)
    ax[0].imshow(ref_im, cmap='gray')
    ax[0].add_patch(plt.Rectangle(ref_patch_pos, *ref_patch.shape[::-1], fill=False, edgecolor='r', linewidth=1))
    ax[1].imshow(tgt_im, cmap='gray')
    if tgt_patch.size > 0:
        ax[1].add_patch(plt.Rectangle(tgt_patch_pos, *tgt_patch.shape[::-1], fill=False, edgecolor='r', linewidth=1))
    plt.show()


if __name__ == '__main__':
    ref_im = cv2.imread('./data/hpatches-sequences-release/v_artisans/1.ppm')
    ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
    H = np.loadtxt('./data/hpatches-sequences-release/v_artisans/H_1_5')
    tgt_im = cv2.warpPerspective(ref_im, H, ref_im.shape[::-1])
    # normalize images to [0, 1]
    ref_im = ref_im / 255.
    tgt_im = tgt_im / 255.

    # extract a random patch from the reference image
    ref_patch, ref_patch_pos = extract_random_patch(ref_im, 100)
    # extract the corresponding patch from the target image
    tgt_patch, tgt_patch_pos = get_patch_position_tgt(tgt_im, ref_patch_pos, H, 100)
    # plot the patches
    if keep_patch_if_valid(tgt_patch):
        print(reconstruction_error(ref_patch, tgt_patch, display=True))
    plot_patches(ref_im, ref_patch, ref_patch_pos, tgt_im, tgt_patch, tgt_patch_pos)

    ref_im = torch.from_numpy(ref_im).unsqueeze(0).unsqueeze(0).float()
    # harris_response
    harris = harris_response(ref_im, k=0.04, grads_mode='diff').squeeze().numpy()
    # hessian_response
    hessian = hessian_response(ref_im, grads_mode='diff').squeeze().numpy()
    # dog_response
    dog = dog_response(ref_im)
    # shi_tomasi_response
    shi_tomasi = gftt_response(ref_im, grads_mode='diff').squeeze().numpy()

    # plot the results
    fig, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(dog, cmap='gray')
    ax[0, 0].set_title('DoG map')
    ax[0, 1].imshow(harris, cmap='gray')
    ax[0, 1].set_title('Harris map')
    ax[1, 0].imshow(hessian, cmap='gray')
    ax[1, 0].set_title('Hessian map')
    ax[1, 1].imshow(shi_tomasi, cmap='gray')
    ax[1, 1].set_title('Shi-Tomasi map')
    plt.show()

