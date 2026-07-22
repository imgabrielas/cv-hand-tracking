import cv2
import mediapipe as mp
import numpy as np
import time
import math as math
from PIL import Image, ImageDraw, ImageFont

try:
    import AVFoundation
except ImportError:
    AVFoundation = None


def find_builtin_camera_index():
    """Locate the Mac's built-in camera, skipping Continuity Camera (iPhone)
    devices which can otherwise take index 0 and get opened instead."""
    if AVFoundation is None:
        return 0
    devices = AVFoundation.AVCaptureDevice.devicesWithMediaType_(AVFoundation.AVMediaTypeVideo)
    for i, device in enumerate(devices):
        if "iphone" not in device.localizedName().lower():
            return i
    return 0


FRAME_WIDTH = 1280
FRAME_HEIGHT = 800


BUTTON_COLOR = (150, 138, 66)
TEXT_COLOR = (211, 194, 205)
HAND_FRAME_COLOR = (239, 206, 123)
HAND1_POINT_COLOR = (170, 204, 150)
HAND1_LINE_COLOR = (155, 192, 204)
HAND2_POINT_COLOR = (255, 207, 230)
HAND2_LINE_COLOR = (164, 199, 255)

FONT_PATH = "/System/Library/Fonts/SFCompact.ttf"

BUTTON_WIDTH = 240
BUTTON_HEIGHT = 60
BUTTON_PADDING = 10
BUTTON_CORNER_RADIUS = 14


def _font(size):
    font = ImageFont.truetype(FONT_PATH, size)
    font.set_variation_by_name("Light")
    return font


def put_text(frame, text, org, font_size, color, anchor="la"):
    imgPIL = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(imgPIL)
    rgb_color = (color[2], color[1], color[0])
    draw.text(org, text, font=_font(font_size), fill=rgb_color, anchor=anchor)
    return cv2.cvtColor(np.array(imgPIL), cv2.COLOR_RGB2BGR)


def draw_rounded_rect(frame, top_left, bottom_right, radius, color):
    x1, y1 = top_left
    x2, y2 = bottom_right
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, cv2.FILLED)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, cv2.FILLED)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, cv2.FILLED)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, cv2.FILLED)


class QuitButton:
    def __init__(self, frame_width, frame_height):
        self.x1 = frame_width - BUTTON_PADDING - BUTTON_WIDTH
        self.y1 = frame_height - BUTTON_PADDING - BUTTON_HEIGHT
        self.x2 = frame_width - BUTTON_PADDING
        self.y2 = frame_height - BUTTON_PADDING
        self.clicked = False

    def contains(self, x, y):
        return self.x1 <= x <= self.x2 and self.y1 <= y <= self.y2

    def on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and self.contains(x, y):
            self.clicked = True

    def draw(self, frame):
        draw_rounded_rect(frame, (self.x1, self.y1), (self.x2, self.y2),
                           BUTTON_CORNER_RADIUS, BUTTON_COLOR)
        cx = (self.x1 + self.x2) // 2
        cy = (self.y1 + self.y2) // 2
        return put_text(frame, "Quit", (cx, cy), 26, TEXT_COLOR, anchor="mm")


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.__mode__ = mode
        self.__maxHands__ = maxHands
        self.__detectionCon__ = detectionCon
        self.__trackCon__ = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw= mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handNo, handLms in enumerate(self.results.multi_hand_landmarks):
                if draw:
                    pointColor = HAND1_POINT_COLOR if handNo == 0 else HAND2_POINT_COLOR
                    lineColor = HAND1_LINE_COLOR if handNo == 0 else HAND2_LINE_COLOR
                    self.mpDraw.draw_landmarks(
                        frame, handLms, self.handsMp.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mpDraw.DrawingSpec(color=pointColor, thickness=2, circle_radius=4),
                        connection_drawing_spec=self.mpDraw.DrawingSpec(color=lineColor, thickness=2))

        return frame

    def findPosition( self, frame, handNo=0, draw=True):
        xList =[]
        yList =[]
        bbox = []
        self.lmsList=[]
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):

                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                xList.append(cx)
                yList.append(cy)
                self.lmsList.append([id, cx, cy])

            xmin, xmax = min(xList), max(xList)
            ymin, ymax = min(yList), max(yList)
            bbox = xmin, ymin, xmax, ymax
            if draw:
                cv2.rectangle(frame, (xmin - 20, ymin - 20),(xmax + 20, ymax + 20),
                               HAND_FRAME_COLOR , 2)

        return self.lmsList, bbox

    def findFingerUp(self):
         fingers=[]

         if self.lmsList[self.tipIds[0]][1] > self.lmsList[self.tipIds[0]-1][1]:
              fingers.append(1)
         else:
              fingers.append(0)

         for id in range(1, 5):
              if self.lmsList[self.tipIds[id]][2] < self.lmsList[self.tipIds[id]-2][2]:
                   fingers.append(1)
              else:
                   fingers.append(0)

         return fingers

    def findDistance(self, p1, p2, frame, draw= True, r=15, t=3):

        x1 , y1 = self.lmsList[p1][1:]
        x2, y2 = self.lmsList[p2][1:]
        cx , cy = (x1+x2)//2 , (y1 + y2)//2

        if draw:
              cv2.line(frame,(x1, y1),(x2,y2) ,(255,0,255), t)
              cv2.circle(frame,(x1,y1),r,(255,0,255),cv2.FILLED)
              cv2.circle(frame,(x2,y2),r, (255,0,0),cv2.FILLED)
              cv2.circle(frame,(cx,cy), r,(0,0.255),cv2.FILLED)
        len= math.hypot(x2-x1,y2-y1)

        return len, frame , [x1, y1, x2, y2, cx, cy]

def main():

        ctime=0
        ptime=0
        cap = cv2.VideoCapture(find_builtin_camera_index(), cv2.CAP_AVFOUNDATION)
        detector = HandTrackingDynamic()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        if not cap.isOpened():
            print("Cannot open camera")
            exit()

        windowName = 'Hand up!!!'
        cv2.namedWindow(windowName)
        quitButton = QuitButton(FRAME_WIDTH, FRAME_HEIGHT)
        cv2.setMouseCallback(windowName, quitButton.on_mouse)

        while True:
            ret, frame = cap.read()
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            frame = cv2.flip(frame, 1)

            frame = detector.findFingers(frame)
            lmsList = detector.findPosition(frame)
            if len(lmsList)!=0:
                ctime = time.time()
            fps =1/(ctime-ptime) if ctime != ptime else 0
            ptime = ctime

            frame = put_text(frame, f"FPS : {int(fps)}", (10, 10), 34, TEXT_COLOR)
            frame = quitButton.draw(frame)

            cv2.imshow(windowName, frame)

            key = cv2.waitKey(1)
            if quitButton.clicked or key in (ord('q'), ord('Q'), 27):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
            main()
