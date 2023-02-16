import glob
import os
from collections import defaultdict

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from kornia.feature import gftt_response, harris_response, hessian_response

# TODO: multiproc generate dataset func
# TODO: extend the config file to include the params

class FeatureExtractor:
    def __init__(self, config_path: str):
        self.config            = yaml.safe_load(open(config_path))
        self.files             = glob.glob(os.path.join(self.config.get('dataset').get('path'), '*'))
        self.resolution        = tuple(self.config.get('dataset').get('resolution'))
        self.patch_size        = self.config.get('dataset').get('patch_size')
        self.keep_aspect_ratio = self.config.get('dataset').get('keep_aspect_ratio')

    def generate_dataset(self):
        pass

    def get_regions_of_interest(self, class_path: str, plot: bool = False):
        '''
        Each feature is from one of 4 responses: Hessian, Harris, Shi-Tomasi, DoG.                 * (the feature maps)
        The goal is to identify regions of interest containing features from at least 3 responses. * (the voting scheme)
        We then create a patch centered at the feature position and check:                         * (the contraints)
            1) if the patch can be constructed in the reference image
            2) if the patch can be constructed in the target image after applying the homography
            3) if the patch doesn't overlap for more than 50% with any other patch already selected
        '''
        # get class name
        class_name = class_path.split('/')[-1]
        # load ref image
        ref_im = cv2.imread(os.path.join(class_path, '1.ppm')).astype('uint8')
        ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
        original_resolution = ref_im.shape
        if self.resolution is not None: ref_im = self.__resize(ref_im)
        # get feature maps
        feat_maps = self.get_feature_maps(ref_im)
        # get good features
        ensemble_features = self.get_good_features(feat_maps)
        # plot feature maps and good features
        if plot:
            self.__plot_feature_maps(*feat_maps)
            self.__plot_ensemble_features(ref_im, ensemble_features)

        ground_truth = defaultdict(dict)
        # homographies
        Hs = [f'H_1_{i}' for i in range(2, 7)]
        for Hpath in Hs:
            H = np.loadtxt(os.path.join(class_path, Hpath))
            if self.resolution is not None: H = self.__adapt_homography(H, original_resolution)
            if class_name.startswith('v_'): # viewpoint change
                tgt_im = cv2.warpPerspective(ref_im, H, ref_im.shape[::-1]).astype('uint8')
            else: # illumination change
                tgt_im = cv2.imread(os.path.join(class_path, f'{Hpath[-1]}.ppm')).astype('uint8')
                tgt_im = cv2.cvtColor(tgt_im, cv2.COLOR_BGR2GRAY)
                if self.resolution is not None: tgt_im = self.__resize(tgt_im) if tgt_im.shape != self.resolution else tgt_im

            assert ref_im.shape == tgt_im.shape == self.resolution, f'Failed to resize images: {ref_im.shape} | {tgt_im.shape} != {self.resolution}'

            # get patches
            patches = []
            for name, ref_centroid in ensemble_features:
                # extract the patches and discard 1) those which can't form a patch and 2) those which are outside the target image
                ref_patch = self.__extract_patch(ref_im, ref_centroid)
                if ref_patch is None: continue
                tgt_patch, tgt_centroid = self.__rpc2tpc(tgt_im, ref_centroid, H)
                if not tgt_patch.size > 0: continue
                # find all the feature points that lie inside the ref patch
                features_inside_patch = [f for f in ensemble_features if self.__point_inside_patch(f[1], ref_centroid)]
                # compare the current patch against all existing patches and check if iou > 0.5; if it is, we discard the current patch
                # lastly, if there are at least 3 responses in the patch, we keep it
                shared_features = set([name for name, _ in features_inside_patch])
                if len(shared_features) >= 3 and self.__is_valid_patch(ref_centroid, patches):
                    patches.append((ref_centroid, ref_patch, tgt_centroid, tgt_patch))

                if len(patches) == 100: break

            if plot: self.__plot_patches(ref_im, tgt_im, patches)

            # save patches
            ground_truth[class_name][Hpath] = patches

    def get_good_features(self, feature_maps: list[tuple]):
        ensemble_features = []
        for response, feat_map in feature_maps:
            features = cv2.goodFeaturesToTrack(feat_map, maxCorners=1000, qualityLevel=0.01, minDistance=10, blockSize=3).squeeze()
            ensemble_features.extend((response, feat.astype(int)) for feat in features)

        return ensemble_features

    def get_feature_maps(self, im: np.ndarray):
        # convert to tensor and normalize to match Kornia's format
        tmp_im = torch.from_numpy(im)[None, None, ...].float() / 255.
        return [
            ('harris', harris_response(tmp_im, k=0.04, grads_mode='diff').squeeze().numpy()),
            ('hessian', hessian_response(tmp_im, grads_mode='diff').squeeze().numpy()),
            ('shi_tomasi', gftt_response(tmp_im, grads_mode='diff').squeeze().numpy()),
            ('dog', self.__dog_response(tmp_im))
        ]

    def __dog_response(self, im, sigma1=1., sigma2=1.6):
        # compute the gaussian blur of the image
        im_blur1 = cv2.GaussianBlur(im.squeeze().numpy(), (0, 0), sigma1)
        im_blur2 = cv2.GaussianBlur(im.squeeze().numpy(), (0, 0), sigma2)
        # compute the difference of gaussian
        dog = im_blur1 - im_blur2

        return dog

    def __is_valid_patch(self, ref_centroid: np.ndarray, patches: list[tuple]):
        # compare the current patch against all existing patches
        if len(patches) == 0: return True
        for centroid, _, _, _ in patches:
            if self.__iou(centroid, ref_centroid) > 0.5: return False
        return True

    def __iou(self, centroid: np.ndarray, ref_centroid: np.ndarray):
        x1, y1 = centroid
        x2, y2 = ref_centroid
        # keep into account the fact that the patches are centered on the feature points
        x_left   = max(x1 - self.patch_size // 2, x2 - self.patch_size // 2)
        x_right  = min(x1 + self.patch_size // 2, x2 + self.patch_size // 2)
        y_top    = max(y1 - self.patch_size // 2, y2 - self.patch_size // 2)
        y_bottom = min(y1 + self.patch_size // 2, y2 + self.patch_size // 2)

        intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
        union = self.patch_size ** 2 + self.patch_size ** 2 - intersection
        iou = intersection / union
        assert 0 <= iou <= 1

        return iou

    def __extract_patch(self, im: np.ndarray, centroid: np.ndarray):
        x, y = centroid
        # extract the patch centered on (x, y)
        x_min = x - self.patch_size // 2
        x_max = x + self.patch_size // 2
        y_min = y - self.patch_size // 2
        y_max = y + self.patch_size // 2
        # check if the patch is inside the image
        if x_min < 0 or x_max > im.shape[1] or y_min < 0 or y_max > im.shape[0]: return None
        # extract the patch
        patch = im[y_min:y_max, x_min:x_max]

        return patch

    def __rpc2tpc(self, im: np.ndarray, ref_centroid: np.ndarray, H: np.ndarray):
        # get the corners of the patch in the target image
        x_min, y_min = ref_centroid
        corners = np.array(
            [[x_min, y_min],
             [x_min, y_min + self.patch_size],
             [x_min + self.patch_size, y_min + self.patch_size],
             [x_min + self.patch_size, y_min]]
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

        return patch, np.array([x_min, y_min])

    def __point_inside_patch(self, point: np.ndarray, ref_centroid: np.ndarray):
        x, y = point
        x_min = ref_centroid[0] - self.patch_size // 2
        x_max = ref_centroid[0] + self.patch_size // 2
        y_min = ref_centroid[1] - self.patch_size // 2
        y_max = ref_centroid[1] + self.patch_size // 2

        return x_min <= x <= x_max and y_min <= y <= y_max

    def __resize(self, im: np.ndarray):
        h, w = self.resolution
        h0, w0 = im.shape[:2]
        if self.keep_aspect_ratio:
            if h0 / w0 > h / w: h = int(h0 * w / w0) # the height is the limiting factor
            else: w = int(w0 * h / h0)               # the width is the limiting factor
        # resize and pad
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA if h < h0 or w < w0 else cv2.INTER_CUBIC)
        if not self.keep_aspect_ratio: # if we don't keep the aspect ratio, we must pad the image
            im = cv2.copyMakeBorder(im, 0, h - im.shape[0], 0, w - im.shape[1], cv2.BORDER_CONSTANT, value=0)

        return im

    def __adapt_homography(self, H: np.ndarray, original_resolution: tuple):
        # because we changed the resolution of the reference image, we must adapt the homography accordingly
        # formula: H' = S * H * S^-1 (where S is the scaling matrix)
        h, w = self.resolution
        h0, w0 = original_resolution
        S = np.array([
            [w / w0, 0, 0],
            [0, h / h0, 0],
            [0, 0, 1]
        ])
        H = S @ H @ np.linalg.inv(S)

        return H

    # viz helpers
    # =============================================================================================================
    def __plot_ensemble_features(self, im: np.ndarray, ensemble_features: list[tuple]):
        # plot the reference image
        plt.imshow(im, cmap='gray')
        # plot the ensemble_features
        responses, features = zip(*ensemble_features)
        x, y = zip(*features)
        # helper to map responses to colors
        mapping = {response: plt.get_cmap('tab10')(i) for i, response in enumerate(sorted(set(responses)))}
        colors = [mapping[response] for response in responses]
        # display
        plt.scatter(x, y, c=colors, s=10, alpha=0.6)
        plt.legend(handles=[mpatches.Patch(color=color, label=response) for response, color in mapping.items()])
        plt.show()

    def __plot_feature_maps(self, *args):
        _, ax = plt.subplots(2, 2, figsize=(10, 10))
        for i, response in enumerate(args):
            name, feat_map = response
            ax[i // 2, i % 2].imshow(feat_map, cmap='gray')
            ax[i // 2, i % 2].set_title(name)
        plt.show()

    def __plot_patches(self, ref_im: np.ndarray, tgt_im: np.ndarray, patches: list[tuple]):
        _, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(ref_im, cmap='gray')
        ax[1].imshow(tgt_im, cmap='gray')
        for ref_centroid, _, _, _ in patches:
            # draw a rectangle centered on the centroid
            x, y = ref_centroid
            rect = mpatches.Rectangle((x - self.patch_size // 2, y - self.patch_size // 2), self.patch_size, self.patch_size, linewidth=1, edgecolor='r', facecolor='none')
            ax[0].add_patch(rect)
            ax[0].scatter(x, y, c='navy', marker='+', s=5)

        plt.show()


if __name__ == '__main__':
    extractor = FeatureExtractor('./config.yaml')
    extractor.get_regions_of_interest('./data/hpatches-sequences-release/v_charing', plot=True)

