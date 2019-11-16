import os
import time

import cv2
import numpy as np
import prnet

from src import RectCoordinates
from src.CaptureAreaDrawer import CaptureAreaDrawer
from src.FaceDetectingGrabber import FaceDetectingGrabber, DetectedObject
from src.SearchingFaceAreaProvider import SearchingFaceAreaProvider
from src.ThreadedAnglePublisher import ThreadedAnglePublisher
from src.ThreadedImageGrabber import ThreadedImageGrabber
from src.ThreadedImagePublisher import ThreadedImagePublisher
from src.ThreadedImageShower import ThreadedImageShower
from src.ThreadedPRNet import ThreadedPRNet, PRNResult
from src.detectors import FaceDetector
from src.detectors.DnnFaceDetector import DnnFaceDetector


class ReflectionAppThreaded:
    DETECTED_FACE_WINDOW = "Detected face"
    FACE_SEARCHING_AREA_WINDOW = "Face searching area"
    PRNET_WINDOW = "PRNET_WINDOW"

    __last_frame_id: int
    __last_result_id: int
    __stream_width: int
    __stream_height: int
    __sfad: SearchingFaceAreaProvider
    __windows_shower: ThreadedImageShower
    __capture_area_drawer: CaptureAreaDrawer
    __image_grabber: ThreadedImageGrabber
    __face_detecting_grabber: FaceDetectingGrabber
    __threaded_prnet: ThreadedPRNet
    __threaded_image_publisher: ThreadedImagePublisher
    __threaded_angle_publisher: ThreadedAnglePublisher
    __prn: prnet.PRN

    def __init__(self, source_device="/dev/video0", output_device="/dev/video1", rotation: int = None,
                 height: int = None, width: int = None, is_gray: bool = False,
                 serial_device: str = None, enable_projection_angle: bool = False):
        self.__is_gray = is_gray
        self.__serial_device = serial_device
        self.__prn = prnet.PRN(is_dlib=False)
        self.__stream_width, self.__stream_height = self.get_video_capturer_dimensions(source_device, rotation, height,
                                                                                       width)
        self.__sfad = SearchingFaceAreaProvider(self.__stream_width, self.__stream_height)

        if os.environ['DEBUG'] == "1":
            self.__windows_shower = ThreadedImageShower({
                self.FACE_SEARCHING_AREA_WINDOW: None,
                self.DETECTED_FACE_WINDOW: None,
                self.PRNET_WINDOW: None
            }).start()

        self.__last_frame_id = 0
        self.__last_result_id = 0
        self.__image_grabber = ThreadedImageGrabber(source_device, rotation, height, width).start()
        if os.environ['DEBUG'] == "1":
            self.__capture_area_drawer = CaptureAreaDrawer(
                self.__image_grabber.read().copy(),
                self.__sfad.face_searching_area
            ).start()

        face_detector: FaceDetector = DnnFaceDetector(True)

        self.__face_detecting_grabber = FaceDetectingGrabber(
            face_detector,
            self.__image_grabber.read()
        )

        self.__threaded_prnet = ThreadedPRNet(
            self.__prn
        )

        self.__face_detecting_grabber.start(self.__image_grabber.read())
        self.__threaded_prnet.start()

        self.__threaded_image_publisher = ThreadedImagePublisher(np.zeros((992, 992, 3), np.uint8),
                                                                 output_device).start()

        if self.__serial_device is not None:
            self.__threaded_angle_publisher = ThreadedAnglePublisher(serial_device, (-0.6, 0.6), (-100, 100),
                                                                     enable_projection_angle).start()

    def loop(self):
        frame_id = self.__image_grabber.read_frame_id()
        if frame_id == self.__last_frame_id:
            time.sleep(0.025)
            return

        self.__last_frame_id = frame_id

        camera_frame = self.__image_grabber.read()

        self.__face_detecting_grabber.update_source_frame(camera_frame)

        if os.environ['DEBUG'] == "1":
            self.__capture_area_drawer.update_source_frame(camera_frame.copy())
            rectangled_frame: RectCoordinates = self.__capture_area_drawer.read()

        detected_object: DetectedObject = self.__face_detecting_grabber.read()
        prn_result: PRNResult = self.__threaded_prnet.read()

        detected_face_area = detected_object.detected_face.detected_face_area
        if detected_object.detected_face.is_face_detected:
            self.__sfad.update_next_searching_frame(detected_face_area)
            self.__threaded_prnet.update_source_frame(self.__face_detecting_grabber.read().output_frame)
        else:
            self.__sfad.update_not_found_face()

        if os.environ['DEBUG'] == "1":
            self.__capture_area_drawer.update_rectangle(self.__sfad.face_searching_area)
            self.__windows_shower.update_window(self.FACE_SEARCHING_AREA_WINDOW,
                                                cv2.resize(rectangled_frame, (432, 768)))

        # self.__write_to_v4l2_loopback(cv2.resize(rectangled_frame, (640, 480), interpolation=cv2.INTER_AREA))

        if prn_result is not None and self.__last_result_id != prn_result.id:
            self.__last_result_id = prn_result.id
            if self.__is_gray:
                self.__send_result(
                    cv2.cvtColor(prn_result.face_texture,
                                 cv2.COLOR_BGR2GRAY), prn_result.face_angle)
            else:
                self.__send_result(
                    prn_result.face_texture, prn_result.face_angle)
            if os.environ['DEBUG'] == "1":
                self.__windows_shower.update_window(
                    "face_texture",
                    self.__scale_cropped_face_image(prn_result.face_texture.copy(), self.CROPPED_FACE_BASE_HEIGHT,
                                                    self.CROPPED_FACE_BASE_WIDTH))
                self.__windows_shower.update_window(
                    "face_with_pose",
                    self.__scale_cropped_face_image(prn_result.face_with_pose, self.CROPPED_FACE_BASE_HEIGHT,
                                                    self.CROPPED_FACE_BASE_WIDTH))
                self.__windows_shower.update_window(
                    "face_with_landmarks",
                    self.__scale_cropped_face_image(prn_result.face_with_landmarks, self.CROPPED_FACE_BASE_HEIGHT,
                                                    self.CROPPED_FACE_BASE_WIDTH))

        if os.environ['DEBUG'] == "1":
            self.__windows_shower.update_window(self.DETECTED_FACE_WINDOW,
                                                self.__scale_cropped_face_image(detected_object.output_frame.copy()))

    def __send_result(self, source_frame: np.ndarray, face_angle: float):
        self.__threaded_image_publisher.update_frame(source_frame)
        if self.__serial_device is not None:
            self.__threaded_angle_publisher.update_angle(face_angle)

    CROPPED_FACE_BASE_HEIGHT = 480
    CROPPED_FACE_BASE_WIDTH = 480

    def __scale_cropped_face_image(self, source_frame: np.ndarray, height=CROPPED_FACE_BASE_HEIGHT,
                                   width=CROPPED_FACE_BASE_WIDTH):
        original_height, original_width, colors = source_frame.shape
        if original_height > original_width:
            height_percent = (height / float(original_height))
            new_width_size = int(float(original_width) * float(height_percent))

            width_diff = (width - new_width_size) / 2

            rounding_fix_fill = np.zeros((width, int(not width_diff.is_integer()), colors), np.uint8)
            width_diff = int(width_diff)
            width_fill = np.zeros((width, width_diff, colors), np.uint8)

            new_image = cv2.resize(
                source_frame,
                (new_width_size, height),
                interpolation=cv2.INTER_LINEAR
            )
            new_image = np.concatenate((width_fill, new_image, width_fill, rounding_fix_fill), axis=1)
        else:
            width_percent = (width / float(original_width))
            new_height_size = int((float(original_height) * float(width_percent)))

            height_diff = (height - new_height_size) / 2

            rounding_fix_fill = np.zeros((int(not height_diff.is_integer()), height, colors), np.uint8)
            height_diff = int(height_diff)
            height_fill = np.zeros((height_diff, height, colors), np.uint8)

            new_image = cv2.resize(
                source_frame,
                (width, new_height_size),
                interpolation=cv2.INTER_LINEAR
            )
            new_image = np.concatenate((height_fill, new_image, height_fill, rounding_fix_fill), axis=0)

        return new_image

    def start(self):
        while True:
            if self.is_stopped():
                self.stop()
                break

            self.loop()

    def get_video_capturer_dimensions(self, source, rotation: int, height: int = None, width: int = None):
        # Init input video stuff
        video_capture = cv2.VideoCapture(source)
        if not video_capture.isOpened():
            print("Video not opened. Exiting.")
            exit()
        (grabbed, frame) = video_capture.read()

        if not grabbed:
            print("could not grab")
            exit(1)

        if height is not None:
            frame = cv2.resize(frame, (height, width))

        if rotation is not None:
            frame = cv2.rotate(frame, rotation)

        (height, width) = frame.shape[:2]
        stream_width = width
        stream_height = height

        del video_capture

        return stream_width, stream_height

    def stop(self):
        self.__image_grabber.stop()
        if os.environ['DEBUG'] == "1":
            self.__capture_area_drawer.stop()
            self.__windows_shower.stop()

    def is_stopped(self):
        return self.__image_grabber.stopped \
               or (os.environ['DEBUG'] == "1" and self.__capture_area_drawer.stopped) \
               or (os.environ['DEBUG'] == "1" and self.__windows_shower.stopped)


os.environ['DEBUG'] = "0"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
ReflectionAppThreaded(source_device="/dev/video0",
                      output_device="/dev/video2",
                      rotation=cv2.ROTATE_90_CLOCKWISE,
                      is_gray=False,
                      serial_device='/dev/ttyUSB0',
                      enable_projection_angle=False
                      ).start()
