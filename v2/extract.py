import numpy as np
import cv2
import matplotlib.pyplot as plt
from kornia.feature import harris_response, hessian_response, gftt_response
import torch
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from collections import defaultdict
import matplotlib.patches as mpatches


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

def extract_good_features(im, nfeatures=1000, quality_level=0.01, min_dist=10, block_size=3):
    assert im.ndim == 2, 'image must be grayscale'
    # convert the image to float32
    im = im.astype(np.float32)
    features = cv2.goodFeaturesToTrack(im, nfeatures, quality_level, min_dist, blockSize=block_size)

    return features.squeeze().astype(np.int16)

def plot_ensemble_features(ref_im, ensemble_features):
    # plot the reference image
    plt.imshow(ref_im, cmap='gray')
    # plot the ensemble_features
    responses, features = zip(*ensemble_features)
    x, y = zip(*features)
    mapping = {response: plt.get_cmap('tab10')(i) for i, response in enumerate(sorted(set(responses)))}
    colors = [mapping[response] for response in responses]
    plt.scatter(x, y, c=colors, s=10, alpha=0.6)
    plt.legend(handles=[mpatches.Patch(color=color, label=response) for response, color in mapping.items()])
    plt.show()

def plot_feature_maps(dog, harris, hessian, shi_tomasi):
    _, ax = plt.subplots(2, 2, figsize=(10, 10))
    ax[0, 0].imshow(dog, cmap='gray')
    ax[0, 0].set_title('DoG map')
    ax[0, 1].imshow(harris, cmap='gray')
    ax[0, 1].set_title('Harris map')
    ax[1, 0].imshow(hessian, cmap='gray')
    ax[1, 0].set_title('Hessian map')
    ax[1, 1].imshow(shi_tomasi, cmap='gray')
    ax[1, 1].set_title('Shi-Tomasi map')
    plt.show()

def identify_rois(ref_im: np.ndarray, tgt_im: np.ndarray, H: np.ndarray, ensemble_features: list[list[tuple]], patch_size: int = 65, n_samples: int = 500, iou_threshold: float = 0.5):
    '''
    each feature is from one of 4 responses: hessian, harris, shi-tomasi, dog
    the goal is to randomly sample without replacement features, create a patch centered at the feature, and check if there are features from at least 3 responses in the patch
    if that's true, we say the patch is a valid region of interest (ROI) and continue sampling
    while we save the ROI's centroid, as we get new ROIs, we only keep those that have an IoU with the previous ROIs lower than 50%
    '''
    def point_within_patch(feature, patch):
        x, y = feature
        x_min, y_min = patch
        x_max, y_max = x_min + patch_size, y_min + patch_size
        return x_min <= x <= x_max and y_min <= y <= y_max

    def iou(ref_patch_pos, tgt_patch_pos):
        x1, y1 = ref_patch_pos
        x2, y2 = tgt_patch_pos
        x_min = max(x1, x2)
        y_min = max(y1, y2)
        x_max = min(x1 + patch_size, x2 + patch_size)
        y_max = min(y1 + patch_size, y2 + patch_size)
        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
        union = patch_size ** 2 + patch_size ** 2 - intersection
        return intersection / union

    # extract the patches
    patches = []
    # NOTE: this loop is slow if using all the features; optimize or threshold the number of features
    for name, feature in ensemble_features:
        # extract the patches and discard those which are outside the image in the target image
        patch = extract_patch(ref_im, feature, patch_size)
        tgt_patch, tgt_patch_pos = get_patch_position_tgt(tgt_im, feature, H, patch_size)
        if keep_patch_if_valid(tgt_patch):
            # find all the feature points that lie inside the ref patch
            features_in_patch = [f for f in ensemble_features if point_within_patch(f[1], feature)]

            # compare the current patch against all existing patch and check iou > 0.5; if it is, we discard the current patch
            if len(patches):
                for ref_patch_pos, _, _, _ in patches:
                    if iou(ref_patch_pos, feature) > iou_threshold:
                        break

            # lastly, if there are 4 responses in the patch, we keep it
            if len(set([name for name, _ in features_in_patch])) >= 3:
                patches.append((feature, patch, tgt_patch_pos, tgt_patch))

        if len(patches) == n_samples: break

    return patches

def test_sift(im1, im2):
    def find_matches(des0: np.ndarray, des1: np.ndarray, ratio: float | None, topk: int | None) -> list[tuple[int, int]]:
        distances = cdist(des0, des1)
        matches = np.argmin(distances, axis=1) # find the closest descriptor

        # sort by distances
        sorted_matches = sorted(zip(range(len(matches)), matches), key=lambda x: distances[x[0], x[1]])
        matches = sorted_matches[:topk] if topk is not None else sorted_matches

        # ratio test
        if ratio is not None: #NOTE: this is a really slow op
            good_matches = []
            for i, j in matches:
                # find the second closest descriptor
                second_closest = np.partition(distances[i], 1)[1] # partition is faster than sort here
                if distances[i, j] < ratio * second_closest: good_matches.append((i, j))
            return good_matches
        return matches

    def plot_matches(img0, img1, kp0, kp1, matches):
        plt.style.use('classic')
        plt.imshow(np.concatenate([img0, img1], axis=1), cmap='gray')
        plt.xticks([])
        plt.yticks([])
        for i, j in matches:
            p1 = np.array(kp0[i].pt)
            p2 = np.array(kp1[j].pt)
            plt.plot([p1[0], p2[0] + img0.shape[1]], [p1[1], p2[1]], c='orange')
        plt.draw()

    sift = cv2.SIFT_create()
    kp0, des0 = sift.detectAndCompute(im1, None)
    kp1, des1 = sift.detectAndCompute(im2, None)
    assert des0 is not None and des1 is not None, 'descriptors are None'
    print(f'Number of keypoints in image 1: {len(kp0)}')
    print(f'Number of keypoints in image 2: {len(kp1)}')
    print(f'Shape of descriptors in image 1: {des0.shape}')
    print(f'Shape of descriptors in image 2: {des1.shape}')
    matches = find_matches(im1, im2, ratio=None, topk=100)
    print(f'Number of matches: {len(matches)}')
    plot_matches(im1, im2, kp0, kp1, matches)

if __name__ == '__main__':
    ref_im = cv2.imread('./data/hpatches-sequences-release/v_charing/1.ppm')
    ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
    H = np.loadtxt('./data/hpatches-sequences-release/v_charing/H_1_5')
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
    plot_feature_maps(dog, harris, hessian, shi_tomasi)

    ref_im = (ref_im.squeeze().numpy() * 255).astype('uint8')
    tgt_im = (tgt_im * 255).astype('uint8')
    ensemble_features = []
    for x in ['harris', 'hessian', 'dog', 'shi_tomasi']:
        features = extract_good_features(eval(x), nfeatures=1000, quality_level=0.01, min_dist=10, block_size=3)
        ensemble_features.extend((x, feat) for feat in features)
    plot_ensemble_features(ref_im, ensemble_features)
    patches = identify_rois(ref_im, tgt_im, H, ensemble_features)

    # plot the patches
    _, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.imshow(ref_im, cmap='gray')
    for ref_pos, ref_patch, _, _ in patches:
        # draw a rectangle around the patch
        ax.add_patch(plt.Rectangle(ref_pos, *ref_patch.shape[::-1], fill=False, edgecolor='red', linewidth=2))
    plt.show()

