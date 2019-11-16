import os
import threading
import time

import cv2
import numpy as np
import prnet
from prnet.utils import estimate_pose, plot_pose_box, plot_kpt

TEXTURE_SIZE = 256


class PRNResult:
    id: int
    face_texture: np.ndarray
    face_with_pose: np.ndarray
    face_with_landmarks: np.ndarray
    face_angle: float

    def __init__(self, id: int, face_texture: np.ndarray, face_angle: float,
                 face_with_landmarks: np.ndarray = None,
                 face_with_pose: np.ndarray = None) -> None:
        self.id = id
        self.face_texture = face_texture
        self.face_angle = face_angle
        self.face_with_landmarks = face_with_landmarks
        self.face_with_pose = face_with_pose


class ThreadedPRNet:
    stopped: bool = False
    __source_frame: np.ndarray = None

    __prn_result: PRNResult = None

    __thread: threading.Thread
    __write_lock: threading.Lock = threading.Lock()
    __prn: prnet.PRN

    __original_face_floated: np.ndarray
    __original_face_pos: any
    __original_face_vertices: any

    __prn_pos: any

    def __init__(self, prn: prnet.PRN):
        self.__prn = prn

    def start(self) -> 'ThreadedPRNet':
        self.__event = threading.Event()
        self.__last_result_id = 0

        self.__thread = threading.Thread(target=self.grab, args=(), daemon=True)
        self.__thread.start()

        return self

    def stop(self) -> None:
        self.stopped = True
        self.__thread.join()

    __last_result_id: int

    def grab(self) -> None:
        start_time = time.time()
        x = 1  # displays the frame rate every 1 second
        counter = 0
        while not self.stopped:
            self.__event.wait()
            self.__event.clear()
            with self.__write_lock:
                if self.__source_frame is None:
                    continue
                source_frame = self.__source_frame
                [h, w, c] = source_frame.shape

            box = np.array(
                [0, source_frame.shape[1] - 1, 0, source_frame.shape[0] - 1])  # cropped with bounding box
            prn_pos = self.__prn.process(source_frame, box)

            if prn_pos is None:
                continue

            # 3D vertices
            vertices = self.__prn.get_vertices(prn_pos)
            save_vertices = vertices.copy()
            save_vertices[:, 1] = h - 1 - save_vertices[:, 1]

            # # get landmarks
            kpt = self.__prn.get_landmarks(prn_pos)
            #
            # # estimate pose
            camera_matrix, pose = estimate_pose(vertices)

            face_angle_x, face_angle_y, face_angle_z = pose

            if os.environ['DEBUG'] == "1":
                # 0 < face_angle_x : Face positioned to the right
                print(f"Face angle: {face_angle_x}")

            try:
                if os.environ['DEBUG'] == "1":
                    self.__prn_result = PRNResult(
                        self.__last_result_id,
                        self.__extract_texture((source_frame / 255).astype(np.float32), prn_pos),
                        face_angle_x,
                        face_with_landmarks=plot_kpt(source_frame.copy(), kpt),
                        face_with_pose=plot_pose_box(source_frame.copy(), camera_matrix, kpt))
                else:
                    self.__prn_result = PRNResult(
                        self.__last_result_id,
                        self.__extract_texture((source_frame / 255.).astype(np.float32), prn_pos),
                        face_angle_x)
            except Exception as err:
                print("errrrrrror: " + str(err))

            self.__last_result_id += 1

            counter += 1
            if (time.time() - start_time) > x:
                print("PRNET FPS: ", counter / (time.time() - start_time))
                counter = 0
                start_time = time.time()

    def update_source_frame(self, source_frame) -> None:
        with self.__write_lock:
            self.__source_frame = source_frame
        self.__event.set()

    def read(self) -> PRNResult:
        return self.__prn_result

    def __extract_texture(self, new_face: np.ndarray, prn_pos: any):
        new_texture = cv2.remap(new_face, cv2.resize(prn_pos[:, :, :2].astype(np.float32), (992, 992),
                                                     interpolation=cv2.INTER_LINEAR), None,
                                interpolation=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=(0))

        return (new_texture * 255).astype(np.uint8)
