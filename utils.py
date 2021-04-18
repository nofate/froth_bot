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


def emulate_stream(path, processor=None):
    """reads the file if it were infinite"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise StopIteration
    while cap.isOpened():
        # Capture frame-by-frame
        ret, frame = cap.read()
        # if frame is read correctly ret is True
        if not ret:
            logging.info("Can't receive frame (stream end?). Exiting ...")
            break
        # Our operations on the frame come here
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if processor is not None:
            frame = processor(frame)
        (flag, enc_frame) = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
            bytearray(frame) + b'\r\n')
        #yield bytes(frame)

