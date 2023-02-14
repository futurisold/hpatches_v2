import glob
import os
from collections import defaultdict

import cv2
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
from kornia.feature import gftt_response, harris_response, hessian_response

# TODO: create a config file
# TODO: multiproc generate dataset func

class FeatureExtractor:
    def __init__(self,
                 hsr_path: str, # hpatches-sequences-release
                 patch_size: int = 65):
        self.files = glob.glob(os.path.join(hsr_path, '*'))
        self.patch_size = patch_size

    def generate_dataset(self):
        pass

    def get_regions_of_interest(self, class_path: str, plot: bool = False):
        # get class name
        class_name = class_path.split('/')[-1]
        # load ref image
        ref_im = cv2.imread(os.path.join(class_path, '1.ppm')).astype('uint8')
        ref_im = cv2.cvtColor(ref_im, cv2.COLOR_BGR2GRAY)
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
        Hs = [f'H_1_{i}' for i in range(2, 6)]
        for Hpath in Hs:
            H = np.loadtxt(os.path.join(class_path, Hpath))
            if class_name.startswith('v_'): # viewpoint change
                tgt_im = cv2.warpPerspective(ref_im, H, ref_im.shape[::-1]).astype('uint8')
            else: # illumination change
                tgt_im = cv2.imread(os.path.join(class_path, f'{Hpath[-1]}.ppm')).astype('uint8')
                tgt_im = cv2.cvtColor(tgt_im, cv2.COLOR_BGR2GRAY)
            # get patches
            patches = []
            for name, ref_centroid in ensemble_features:
                # extract the patches and discard those which are outside the target image
                ref_patch = self.__extract_patch(ref_im, ref_centroid)
                tgt_patch, tgt_centroid = self.__rpc2tpc(tgt_im, ref_centroid, H)
                if self.__is_valid_patch(tgt_patch):
                    # find all the feature points that lie inside the ref patch
                    features_inside_patch = [f for f in ensemble_features if self.__point_inside_patch(f[1], ref_centroid)]

                    # compare the current patch against all existing patch and check if iou > 0.5; if it is, we discard the current patch
                    if len(patches):
                        for centroid, _, _, _ in patches:
                            if self.__iou(centroid, tgt_centroid) > 0.5: break

                    # lastly, if there are at least 3 responses in the patch, we keep it
                    if len(set([name for name, _ in features_inside_patch])) >= 3:
                        patches.append((ref_centroid, ref_patch, tgt_centroid, tgt_patch))

                if len(patches) == 500: break

            if plot: self.__plot_patches(ref_im, patches)

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

    def __is_valid_patch(self, tgt_patch: np.ndarray):
        # given how the patches are extracted, some of them might be outside the image
        # if this happens, the patch is empty and we discard it
        return tgt_patch.size > 0

    def __iou(self, ref_centroid: np.ndarray, tgt_centroid: np.ndarray):
        x1, y1 = ref_centroid
        x2, y2 = tgt_centroid
        x_min = max(x1, x2)
        y_min = max(y1, y2)
        x_max = min(x1 + self.patch_size, x2 + self.patch_size)
        y_max = min(y1 + self.patch_size, y2 + self.patch_size)
        intersection = max(0, x_max - x_min) * max(0, y_max - y_min)
        union = self.patch_size ** 2 + self.patch_size ** 2 - intersection

        return intersection / union

    def __extract_patch(self, im: np.ndarray, centroid: np.ndarray):
        # get the top left corner of the patch
        x, y = centroid
        # extract the patch
        patch = im[y: y + self.patch_size, x: x + self.patch_size]

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

        return patch, (x_min, y_min)

    def __point_inside_patch(self, feature: np.ndarray, patch: np.ndarray):
        x, y = feature
        x_min, y_min = patch
        x_max, y_max = x_min + self.patch_size, y_min + self.patch_size

        return x_min <= x <= x_max and y_min <= y <= y_max

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

    def __plot_patches(self, ref_im: np.ndarray, patches: list[tuple]):
        _, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.imshow(ref_im, cmap='gray')
        for ref_centroid, ref_patch, _, _ in patches:
            # draw a rectangle around the patch
            ax.add_patch(plt.Rectangle(ref_centroid, *ref_patch.shape[::-1], fill=False, edgecolor='red', linewidth=2))
        plt.show()


if __name__ == '__main__':
    extractor = FeatureExtractor('./data/hpatches-sequences-release')
    extractor.get_regions_of_interest('./data/hpatches-sequences-release/v_charing', plot=True)

