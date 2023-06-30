"""
Measure Face Distance
By: Toan Tran
Email: chitoantran@outlook.com
Phone: +84919730811
"""

import cv2
import mediapipe as mp
import math


class FaceMeshDetector:
    """
    Face Mesh Detector to find 468 Landmarks using the mediapipe library.
    Helps acquire the landmark points in pixel format
    """

    def __init__(self, staticMode=False, maxFaces=2, minDetectionCon=0.5, minTrackCon=0.5, refineLM=False):
        self.staticMode = staticMode
        self.maxFaces = maxFaces
        self.minDetectionCon = minDetectionCon
        self.minTrackCon = minTrackCon
        self.refineLM = refineLM

        self.mpDraw = mp.solutions.drawing_utils
        self.mpFaceMesh = mp.solutions.face_mesh
        self.faceMesh = self.mpFaceMesh.FaceMesh(
                        static_image_mode = self.staticMode,
                        max_num_faces = self.maxFaces,
                        min_detection_confidence = self.minDetectionCon,
                        min_tracking_confidence = self.minTrackCon,
                        refine_landmarks = self.refineLM
                        )
        self.drawSpec = self.mpDraw.DrawingSpec(thickness=1, circle_radius=2)

    def findFaceMesh(self, img, draw=True):
        """
        Finds Face Landmarks in BGR image
        :param img: BGR Image to find the face landmark in 
        :param draw: Draw the output on the image or not
        """

        self.imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.faceMesh.process(self.imgRGB)
        faces = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpFaceMesh.FACEMESH_CONTOURS,self.drawSpec, self.drawSpec)
                face = []
                for id, lm in enumerate(facelms.landmark):
                    ih, iw, ic = img.shape
                    x, y = int(lm.x * iw), int(lm.y * ih)
                    face.append([x,y])
                faces.append(face)
        return img, faces
    
    def findDistance(self, p1, p2, img=None):
        """
        Find the distance between two landmarks based on their
        index numbers.
        :param p1: Point1
        :param p2: Point2
        :param img: Image to draw on.
        :param draw: Flag to draw the output on the image.
        :return: Distance between the points
                    Image with output drawn
                    Line information
        """
        x1, y1 = p1
        x2, y2 = p2
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        length = math.hypot(x2 - x1, y2 - y1)
        info = (x1,y1,x2,y2,cx,cy)
        if img is not None:
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
            return length,info, img
        else:
            return length, info