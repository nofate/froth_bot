import cv2
import numpy as np
import os
import logging


def get_point_pairs(old_frame, new_frame, cutoff=0.8):
    sift = cv2.xfeatures2d.SIFT_create()
    
    old_frame_ = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    new_frame_ = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
    kp0, des0 = sift.detectAndCompute(old_frame_, None)
    kp1, des1 = sift.detectAndCompute(new_frame_, None)
    if len(kp0) == 0 or len(kp1) == 0:
        return ([], [])
    pos0 = np.stack([np.asarray(k.pt) for k in kp0])
    pos1 = np.stack([np.asarray(k.pt) for k in kp1])
    
    from sklearn.neighbors import NearestNeighbors

    point_knn = NearestNeighbors(n_neighbors=4)
    point_knn.fit(pos0)
    p_ids = point_knn.kneighbors(pos1, return_distance=False) #p_dist, 

    knn = NearestNeighbors(n_neighbors=8)
    knn.fit(des0)
    dist, ids = knn.kneighbors(des1, return_distance=True)
    
    sel_ids = np.where((dist[:, 0]/dist[:, 1] < cutoff) & [(best_point in nn) for best_point, nn in zip(ids[:, 0], p_ids)])[0]
    logging.info(sel_ids.shape)
    old_ids = ids[sel_ids, 0]
    if len(old_ids) == 0 or len(sel_ids) == 0:
        return ([], [])

    old_p = np.stack([np.asarray(kp0[i].pt) for i in old_ids])
    new_p = np.stack([np.asarray(kp1[i].pt) for i in sel_ids])
    return (old_p, new_p)


class SpeedExtractor:
    def __init__(self):
        self.prev_frame = None

    def draw_lines(self, image, old_p, new_p):
        img = image.copy()
        if len(old_p) == 0:
            return img
        shifts = np.sqrt(((new_p - old_p)**2).sum(1))
        mean, std = shifts.mean(), shifts.std()
        ids = shifts < mean + 3*std
        old_p = old_p[ids]
        new_p = new_p[ids]
        for o, n in zip(old_p, new_p):
            x0, y0 = o
            x1, y1 = n
            sx = int(x1 + (x1 - x0))
            sy = int(y1 + (y1 - y0))
            img = cv2.line(img, (int(x1), int(y1)), (sx, sy), (0, 0, 255), 1)
            img = cv2.line(img, (sx-2, sy-2), (sx-1, sy-1), (0, 0, 255), 2)
            # print(x1, y1, sx, sy)
        return img
        pass

    def draw_on_image(self, image, points=([], [])):
        old_p, new_p = points
        image_with_matches = self.draw_lines(image.copy(), old_p, new_p)
        # print((image_with_matches-image).max())
        image_with_padding = cv2.copyMakeBorder(
            image_with_matches, 0,180,0,0,cv2.BORDER_CONSTANT,
            value=[0,0,0])
        speed = "-"
        direction = "-"
        if len(new_p) > 0:
            shifts = np.sqrt(((new_p - old_p)**2).sum(1))
            speed = "%.3f" % shifts.mean()
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 1
        lineType = 2

        image_with_text = cv2.putText(
            image_with_padding,'Speed:   %s' % speed,  (0, 630), font, fontScale, [255,255,255], lineType)
        return image_with_text

    def compute_features(self, image):
        if self.prev_frame is None:
            self.prev_frame = image
            return None  #self.draw_on_image(image)
            
        old_p, new_p = get_point_pairs(self.prev_frame, image)
        # print(old_p)
        self.prev_frame = image
        return self.draw_on_image(image, (old_p, new_p))
        # return image


class ImageDynamicProcessor:
    def __init__(self):
        self.fe = SpeedExtractor()

    def __call__(self, frame):
        processed_frame = self.fe.compute_features(frame)

        return processed_frame


if __name__=="__main__":
    processor = ImageDynamicProcessor()
    filename = "video/F1_1_1_1.ts"
    from utils import emulate_stream
    emulate_stream(filename, "bubbles_.mp4", processor=processor, max_frames=15)