# import the necessary packages
import sys
import time
from queue import Queue
from threading import Lock, Thread

import cv2


class FileVideoStream:

    def __init__(self, path):
        # initialize the file video stream along with the boolean
        # used to indicate if the thread should be stopped or not
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        # initialize the queue used to store frames read from
        # the video file
        self.read_lock = Lock()

    def start(self):
        # start a thread to read frames from the file video stream
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()
        return self

    def update(self):
        # keep looping infinitely
        while True:
            # if the thread indicator variable is set, stop the
            # thread
            if self.stopped:
                return
            # read the next frame from the file
            (grabbed, frame) = self.stream.read()
            # otherwise, ensure the queue has room in it
            with self.read_lock:
                # if the `grabbed` boolean is `False`, then we have
                # reached the end of the video file
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def more(self):
        # return True if there are still frames in the queue
        return 1

    def release(self):
        # indicate that the thread should be stopped
        self.stopped = True
        self.t.join()
