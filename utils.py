import itertools as it
from io import BufferedIOBase
import cv2
import logging
import os

class IterReaderIO(BufferedIOBase):
    def __init__(self, iterable=None):
        iterable = iterable or []
        self.iter = it.chain.from_iterable(iterable)

    def not_newline(self, s):
        return s not in {'\n', '\r', '\r\n'}

    def write(self, iterable):
        to_chain = it.chain.from_iterable(iterable)
        self.iter = it.chain.from_iterable([self.iter, to_chain])

    def read(self, n=None):
        return bytearray(it.islice(self.iter, None, n))

    def readline(self, n=None):
        to_read = it.takewhile(self.not_newline, self.iter)
        return bytearray(it.islice(to_read, None, n))


def draw_flux(i):
    old_points, new_points = all_point_pairs[i]
    #plt.figure(figsize=(6, 8))
    # frame0 = frames[i]
    plt.clf()
    frame = frames[i]
    plt.imshow(frame, cmap="Greys", vmin=0, vmax=255)
    for o, n in zip(old_points, new_points):
        #print(o)
        plt.plot([o[0], n[0]], [o[1], n[1]], color="red")
        plt.plot([n[0], n[0]+1], [n[1], n[1]+1], color="blue")
        #break
    plt.title(f"Frame {i}")
    plt.axis("off")


def emulate_stream(path, output_path, processor=None, max_frames=-1):
    """reads the file if it were infinite"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise StopIteration

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), 5.0, (800, 600))
    current_frame=0

    while cap.isOpened():
        # Capture frame-by-frame
        if max_frames > 0:
            if current_frame >= max_frames:
                break
        ret, frame = cap.read()
        current_frame += 1
        
        # if frame is read correctly ret is True
        if not ret:
            logging.info("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if processor is not None:
            frame = processor(frame)

        out.write(frame)

    cap.release()
    out.release()


