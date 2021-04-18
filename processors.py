import cv2
import numpy as np
import math
import webcolors
import traceback
import sys
from sklearn.metrics import mean_squared_error


def rotate_image(image, angle):
    # print("rotate", image.shape)
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result


def longest_subsequence(sequence):
    max_neg = 0
    max_pos = 0
    curr_neg = 0
    curr_pos = 0
    # print(sequence)
    for i in range(len(sequence)):
        if sequence[i] < 0:
            curr_neg += 1
            if curr_neg > max_neg:
                max_neg = curr_neg
            curr_pos = 0
        elif sequence[i] > 0:
            curr_pos += 1
            if curr_pos > max_pos:
                max_pos = curr_pos
            curr_neg = 0
        else:
            curr_pos += 1
            if curr_pos > max_pos:
                max_pos = curr_pos
            curr_neg = 0

    return max_neg, max_pos


def rgb2hex(c):
    return "#{:02x}{:02x}{:02x}".format(
        int(c[0]), int(c[1]), int(c[2])
    )  # format(int(c[0]), int(c[1]), int(c[2]))


def hex2name(c):
    h_color = "#{:02x}{:02x}{:02x}".format(int(c[0]), int(c[1]), int(c[2]))
    try:
        nm = webcolors.hex_to_name(h_color, spec="css3")
    except ValueError as v_error:
        # print("{}".format(v_error))
        rms_lst = []
        for img_clr, img_hex in webcolors.CSS3_NAMES_TO_HEX.items():
            cur_clr = webcolors.hex_to_rgb(img_hex)
            rmse = np.sqrt(mean_squared_error(c, cur_clr))
            rms_lst.append(rmse)

        closest_color = rms_lst.index(min(rms_lst))

        nm = list(webcolors.CSS3_NAMES_TO_HEX.items())[closest_color][0]
    return nm


def get_image_with_matches2(my_image, small_images, centroids):
    new_gray = my_image.copy()
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    ellipses = []
    for i, small_image in enumerate(
        small_images
    ):  # range(150, 151):#range(len(small_images)):#range(148, 150):#range(len(small_images)):
        # print(i)
        try:
            # small_image = small_images[i]
            if np.min(small_image.shape) < 1:
                continue
            sobel_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
            points = []
            max_angle = 0
            max_size = 0
            max_size2 = 0

            for j in range(0, 13):
                # print(sobel_image.shape, "sobel")
                rotated_image = rotate_image(sobel_image, 15 * j)
                # print("small", small_image.shape)
                rotated_small_image = rotate_image(small_image, 15 * j)

                grad_x = cv2.Sobel(
                    rotated_image,
                    ddepth,
                    1,
                    0,
                    ksize=7,
                    scale=scale,
                    delta=delta,
                    borderType=cv2.BORDER_DEFAULT,
                )
                grad_y = cv2.Sobel(
                    rotated_image,
                    ddepth,
                    0,
                    1,
                    ksize=7,
                    scale=scale,
                    delta=delta,
                    borderType=cv2.BORDER_DEFAULT,
                )
                if grad_x is None or grad_y is None:
                    print("grad_x, grad_y")
                size = np.min(longest_subsequence(grad_x[grad_x.shape[0] // 2]))
                size2 = np.min(longest_subsequence(grad_y.T[grad_y.shape[1] // 2]))
                # print((15*j, size, size2))
                if size > max_size:
                    max_size = size
                    max_angle = 15 * j
                    max_size2 = size2
                # print(max_size)
                # print(max_size2)

                # rotated_small_image = cv2.circle(rotated_small_image, tuple(np.array(rotated_small_image.shape[1::-1]) // 2), int(size), (0, 255, 0), 5)
                # rotated_small_image = cv2.circle(rotated_small_image, (int(centroids[i+1,0]), int(centroids[i+1,1])), int(size), (0, 255, 0), 5)
                # plt.imshow(rotated_image)

                # plt.show()
            # print((int(centroids[i+1,0]), int(centroids[i+1,1])))
            # print(int(max_size), int(max_size2))
            # print(int(max_angle))

            # new_gray = cv2.circle(new_gray, (int(centroids[i+1,0]), int(centroids[i+1,1])), int(max_size), (0,255,0), 5)
            if max_size > 10:
                new_gray = cv2.ellipse(
                    new_gray,
                    (int(centroids[i + 1, 0]), int(centroids[i + 1, 1])),
                    (int(max_size), int(max_size2)),
                    180 + max_angle,
                    0,
                    360,
                    (0, 255, 0),
                    2,
                )
                ellipses.append(
                    [
                        int(centroids[i + 1, 0]),
                        int(centroids[i + 1, 1]),
                        int(max_size),
                        int(max_size2),
                        i,
                    ]
                )
            # plt.imshow(grad_x)
            # plt.show()
            # print(size)
            # print(size2)
        except Exception:
            traceback.print_exc(file=sys.stdout)
            exit(1)
            continue
    return new_gray, ellipses


class FeaturesExtractor:
    def __init__(self):
        pass

    def compute_features(self, image):
        # image = cv2.imread("/home/jovyan/frames_F1_2_4_1/frame%d.jpg" % j)
        image_gaussian = cv2.GaussianBlur(image, (5, 5), 0)
        image_gray = cv2.cvtColor(image_gaussian, cv2.COLOR_BGR2GRAY)
        threshold, image_binary = cv2.threshold(image_gray, 160, 255, cv2.THRESH_BINARY)
        # +cv2.THRESH_OTSU)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(image_binary)
        image_copy = image.copy()
        bound_size = 6
        small_images = []
        for i in range(1, count):
            # print(stats[i])
            # image_copy = cv2.circle(image_copy, (int(centroids[i,0]), int(centroids[i,1])), 1, (0, 255, 0), 5)
            # image_copy = cv2.circle(image_copy, (int(centroids[i,0]), int(centroids[i,1])),  int(bound_size * abs(centroids[i, 1] - stats[i, 1])), (0, 255, 0), 5)
            max_bound = max(
                abs(centroids[i, 1] - stats[i, 1]), abs(centroids[i, 0] - stats[i, 0])
            )
            # image_copy = cv2.rectangle(image_copy, (stats[i, 0], stats[i, 1]),  (stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]), (0, 255, 0), 5)
            small_images.append(
                image[
                    max(0, int(centroids[i, 1] - bound_size * max_bound)) : int(
                        centroids[i, 1] + bound_size * max_bound
                    ),
                    max(0, int(centroids[i, 0] - bound_size * max_bound)) : int(
                        centroids[i, 0] + bound_size * max_bound
                    ),
                ]
            )
        # print(len(small_images))
        image_with_matches, ellipses = get_image_with_matches2(
            image, small_images, centroids
        )
        ellipses_np = np.array(ellipses)
        # print(ellipses_np)
        if len(ellipses_np) == 0:
            mean_diam = -1
        else:
            mean_diam = np.mean(ellipses_np[:, 2]) + np.mean(ellipses_np[:, 3])
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = [0, 0, 0]
        index = 0
        for ellipse in ellipses:
            try:
                color += image[ellipse[0] + ellipse[2], ellipse[1] + ellipse[3]]
                index += 1
            except:
                continue
        color = np.asarray(color)
        if index > 0:
            color = color // index
        # print(color)
        hex_col = rgb2hex(color)
        color_name = hex2name(color)
        if len(ellipses_np) > 0:
            brightness = np.mean(
                np.take(stats, ellipses_np[:, 4] + 1, 0)[:, 3]
            ) / np.mean(ellipses_np[:, 2])
            near_exit = (ellipses_np[:, 1] > 400).sum() / float(len(ellipses))
            uniformness = min(
                (ellipses_np[:, 0] > 400).sum()
                / float((ellipses_np[:, 0] <= 400).sum()),
                (ellipses_np[:, 0] <= 400).sum()
                / float((ellipses_np[:, 0] > 400).sum()),
            )
            mean_roundness = np.mean(ellipses_np[:, 3] / ellipses_np[:, 2])
            large_frac = (ellipses_np[:, 2] > 20).sum() / float(len(ellipses_np))
            brightness = "%.3f" % brightness
            near_exit = "%.3f" % near_exit
            uniformness = "%.3f" % uniformness
            mean_roundness = "%.3f" % mean_roundness
            large_frac = "%.3f" % large_frac
        else:
            brightness = "-"
            near_exit = "-"
            uniformness = "-"
            mean_roundness = "-"
            large_frac = "-"
        fontScale = 1
        lineType = 2
        image_with_padding = cv2.copyMakeBorder(
            image_with_matches, 0, 180, 0, 0, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        image_with_text = cv2.putText(
            image_with_padding,
            "Froth number:   %d" % len(ellipses),
            (0, 630),
            font,
            fontScale,
            [255, 255, 255],
            lineType,
        )
        image_with_text = cv2.putText(
            image_with_text,
            "Mean froth diam: %d" % int(mean_diam),
            (0, 670),
            font,
            fontScale,
            [255, 255, 255],
            lineType,
        )
        image_with_text = cv2.putText(
            image_with_text,
            "Froth color: " + str(color_name),
            (0, 710),
            font,
            fontScale,
            [255, 255, 255],
            lineType,
        )
        image_with_text = cv2.putText(
            image_with_text,
            "Froth brightness: %s" % brightness,
            (0, 750),
            font,
            fontScale,
            [255, 255, 255],
            lineType,
        )
        image_with_text = cv2.putText(
            image_with_text,
            "Near exit rate: %s" % near_exit,
            (400, 630),
            font,
            fontScale,
            [255, 255, 255],
            lineType,
        )
        image_with_text = cv2.putText(
            image_with_text,
            "Uniformness: %s" % uniformness,
            (400, 670),
            font,
            fontScale,
            [255, 255, 255],
            lineType,
        )
        image_with_text = cv2.putText(
            image_with_text,
            "Mean roundness: %s" % mean_roundness,
            (400, 710),
            font,
            fontScale,
            [255, 255, 255],
            lineType,
        )
        image_with_text = cv2.putText(
            image_with_text,
            "Large bubble frac: %s" % large_frac,
            (400, 750),
            font,
            fontScale,
            [255, 255, 255],
            lineType,
        )
        return image_with_padding


class ImageProcessor:
    def __init__(self, processor=FeaturesExtractor()):
        self.prev_frame = None
        self.fe = processor  # FeaturesExtractor()

    def __call__(self, frame):
        processed_frame = self.fe.compute_features(frame)

        return processed_frame


if __name__ == "__main__":
    from speed import SpeedExtractor

    processor = ImageProcessor()  #processor=SpeedExtractor())
    filename = "video/F2_1_1_2.ts"
    from utils import emulate_stream

    emulate_stream(filename, "bubbles_part1.mp4", processor=processor, max_frames=20)
