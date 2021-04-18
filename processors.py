import cv2
import numpy as np
import math


def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result


def get_image_with_matches2(my_image, small_images):
    new_gray = my_image.copy()
    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    ellipses = []
    for i in range(len(small_images)):#range(150, 151):#range(len(small_images)):#range(148, 150):#range(len(small_images)):
        #print(i)
        try:
            small_image = small_images[i]
            sobel_image = cv2.cvtColor(small_image, cv2.COLOR_BGR2GRAY)
            points = []
            max_angle = 0
            max_size = 0
            max_size2 = 0

            for j in range(0,13):
                rotated_image = rotate_image(sobel_image, 15*j)
                rotated_small_image = rotate_image(small_image, 15*j)

                grad_x = cv2.Sobel(rotated_image, ddepth, 1, 0, ksize=7, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
                grad_y = cv2.Sobel(rotated_image, ddepth, 0, 1, ksize=7, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
                size = np.min(longest_subsequence(grad_x[grad_x.shape[0]//2]))
                size2 = np.min(longest_subsequence(grad_y.T[grad_y.shape[1]//2]))
                #print((15*j, size, size2))
                if size > max_size:
                    max_size = size
                    max_angle = 15*j
                    max_size2 = size2
                #print(max_size)
                #print(max_size2)

                #rotated_small_image = cv2.circle(rotated_small_image, tuple(np.array(rotated_small_image.shape[1::-1]) // 2), int(size), (0, 255, 0), 5)
                #rotated_small_image = cv2.circle(rotated_small_image, (int(centroids[i+1,0]), int(centroids[i+1,1])), int(size), (0, 255, 0), 5)
                #plt.imshow(rotated_image)

                #plt.show()
            #print((int(centroids[i+1,0]), int(centroids[i+1,1])))
            #print(int(max_size), int(max_size2))        
            #print(int(max_angle))        

            #new_gray = cv2.circle(new_gray, (int(centroids[i+1,0]), int(centroids[i+1,1])), int(max_size), (0,255,0), 5)
            if max_size > 10:
                new_gray = cv2.ellipse(new_gray, 
                    (
                    int(centroids[i+1,0]),
                    int(centroids[i+1,1])),
                    (int(max_size), int(max_size2)),
                    180+max_angle, 0, 360, (0,255,0), 2)
                ellipses.append([
                    int(centroids[i+1,0]),
                    int(centroids[i+1,1]),
                    int(max_size),
                    int(max_size2), i])
            #plt.imshow(grad_x)
            #plt.show()
            #print(size)
            #print(size2)
        except:
            continue
    return new_gray, ellipses


class FeaturesExtractor:
    def __init__(self):
        pass

    def compute_features(self, frame_bgr):
        image_gaussian = cv2.GaussianBlur(frame_bgr, (5, 5), 0)
        image_gray = cv2.cvtColor(image_gaussian, cv2.COLOR_BGR2GRAY)
        threshold, image_binary = cv2.threshold(image_gray, 160, 255, cv2.THRESH_BINARY)
        #+cv2.THRESH_OTSU)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(image_binary)
        image_copy = frame_bgr.copy()
        bound_size = 6
        small_images = []
        for i in range(1, count):
            #print(stats[i])
            #image_copy = cv2.circle(image_copy, (int(centroids[i,0]), int(centroids[i,1])), 1, (0, 255, 0), 5)
            #image_copy = cv2.circle(image_copy, (int(centroids[i,0]), int(centroids[i,1])),  int(bound_size * abs(centroids[i, 1] - stats[i, 1])), (0, 255, 0), 5)
            max_bound = max(abs(centroids[i, 1] - stats[i, 1]), abs(centroids[i, 0] - stats[i, 0]))
            #image_copy = cv2.rectangle(image_copy, (stats[i, 0], stats[i, 1]),  (stats[i, 0] + stats[i, 2], stats[i, 1] + stats[i, 3]), (0, 255, 0), 5)
            small_images.append(frame_bgr[max(0,int(centroids[i, 1] - bound_size * max_bound)) : 
                                           int(centroids[i, 1] + bound_size * max_bound), 
                                           max(0,int(centroids[i, 0] - bound_size*max_bound)) : 
                                           int(centroids[i, 0] + bound_size*max_bound)])
        # print(len(small_images))
        image_with_matches, ellipses = get_image_with_matches2(frame_bgr, small_images)
        ellipses_np = np.array(ellipses)
        mean_diam = (np.mean(ellipses_np[:,2]) + np.mean(ellipses_np[:,3])) 
        font = cv2.FONT_HERSHEY_SIMPLEX
        color = [0,0,0]
        index = 0
        for ellipse in ellipses:
            try:
                color += image[ellipse[0] + ellipse[2], ellipse[1] + ellipse[3]]
                index += 1
            except:
                continue
        color = color // index
        hex_col = rgb2hex(color)
        color_name = hex2name(color)
        brightness = np.mean(np.take(stats, ellipses_np[:,4] + 1, 0)[:,3])/np.mean(ellipses_np[:,2])
        near_exit = (ellipses_np[:,1] > 400).sum()/float(len(ellipses))
        uniformness = min((ellipses_np[:,0] > 400).sum() / float((ellipses_np[:,0] <= 400).sum()),(ellipses_np[:,0] <= 400).sum() / float((ellipses_np[:,0] > 400).sum()))
        mean_roundness = np.mean(ellipses_np[:,3]/ellipses_np[:,2])
        large_frac = (ellipses_np[:, 2] > 20).sum()/float(len(ellipses_np))
        fontScale = 1
        lineType = 2
        image_with_padding = cv2.copyMakeBorder(image_with_matches, 0,180,0,0, cv2.BORDER_CONSTANT,value=[0,0,0])
        image_with_text = cv2.putText(image_with_padding,'Froth number:   %d' % len(ellipses),  (0, 630), font, fontScale, [255,255,255], lineType)
        image_with_text = cv2.putText(image_with_text,'Mean froth diam: %d' % int(mean_diam),  (0, 670), font, fontScale, [255,255,255], lineType)
        image_with_text = cv2.putText(image_with_text,'Froth color: ' + str(color_name),  (0, 710), font, fontScale, [255,255,255], lineType)
        image_with_text = cv2.putText(image_with_text,'Froth brightness: %.3f' % brightness,  (0, 750), font, fontScale, [255,255,255], lineType)
        image_with_text = cv2.putText(image_with_text,'Near exit rate: %.3f' % near_exit,  (400, 630), font, fontScale, [255,255,255], lineType)
        image_with_text = cv2.putText(image_with_text,'Uniformness: %.3f' % uniformness,  (400, 670), font, fontScale, [255,255,255], lineType)
        image_with_text = cv2.putText(image_with_text,'Mean roundness: %.3f' % mean_roundness,  (400, 710), font, fontScale, [255,255,255], lineType)
        image_with_text = cv2.putText(image_with_text,'Large bubble frac: %.3f' % large_frac,  (400, 750), font, fontScale, [255,255,255], lineType)
        # images_to_glue2.append(image_with_padding)
        return image_with_padding



class ImageProcessor:
    def __init__(self):
        self.prev_frame = None
        self.fe = FeaturesExtractor()

    def __call__(self, frame):
        processed_frame = self.fe.compute_features(frame)
        return processed_frame
        if self.prev_frame is not None:
            pass



if __name__ == "__main__":
    processor = ImageProcessor()
    filename = "video/F1_1_1_1.ts"
    from utils import emulate_stream
    emulate_stream(filename, "bubbles.mp4", processor=processor, max_frames=30)