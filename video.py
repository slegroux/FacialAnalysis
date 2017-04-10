#!/usr/bin/env python
import cv2
import numpy as np
import math
from OSC import OSCMessage, OSCClient
from facial_features import FaceDetector, EyeDetector, SmileDetector, MouthDetector, NoseDetector, LeftEyeDetector, RightEyeDetector, RightEarDetector


def setup():

	# OSC
	global client
	client = OSCClient()
	client.connect(("localhost", 7300))

	# DISPLAY
	window_width = 640
	window_height = 480

	# OPENCV
	# img = cv2.imread(filename)
	global camera
	camera = cv2.VideoCapture(0)
	camera.set(3, window_width)
	camera.set(4, window_height)
	# ds_factor = 1.0

	# FACIAL FEATURES DETECTORS
	global face_detector, mouth_detector, smile_detector, nose_detector, right_eye_detector, left_eye_detector
	face_detector = FaceDetector()
	# eye_detector = EyeDetector()
	mouth_detector = MouthDetector()
	smile_detector = SmileDetector()
	nose_detector = NoseDetector()
	# right_ear_detector = RightEarDetector()
	right_eye_detector = RightEyeDetector()
	left_eye_detector = LeftEyeDetector()


def loop():
	while True:
		ret, frame = camera.read()
		# frame = cv2.resize(frame, None, fx=ds_factor, fy=ds_factor, interpolation=cv2.INTER_AREA)
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# corners = cv2.goodFeaturesToTrack(gray, 7, 0.05, 25)
		# corners = np.float32(corners)

		# for item in corners:
		# 	x, y = item[0]
		# 	cv2.circle(frame, (x,y), 5, 255, -1)

		# EAR
		# r_ear = right_ear_detector.detect(gray)
		# right_ear_detector.draw(frame,bgr=(100,100,200))

		# FACE
		faces = face_detector.detect(gray)
		# face_detector.draw(frame, bgr=(200, 100, 100))
		for (x, y, w, h) in faces:

			face_detector.draw_one(frame, (x, y, w, h), (200, 100, 100))

			msg = OSCMessage("/face")
			msg.append((x, y, w, h))
			client.send(msg)

			# face area
			# face_gray = gray[y:y + h, x:x + w]
			# face_color = frame[y:y + h, x:x + w]

			# cv2.rectangle(
			# 	frame,
			# 	(x, y+h/2),
			# 	(x + w, y + h),
			# 	(200,200,200),
			# 	2
			# )

			# EYES
			# higher_gray = gray[y:y + 2 * h / 3, x:x + w]
			# higher_color = frame[y:y + 2 * h / 3, x:x + w]
			higher_right_gray = gray[y:y + 2 * h / 3, x + w / 2:x + w]
			higher_right_color = frame[y:y + 2 * h / 3, x + w / 2:x + w]
			higher_left_gray = gray[y:y + 2 * h / 3, x:x + w / 2]
			higher_left_color = frame[y:y + 2 * h / 3, x:x + w / 2]

			right_eye = right_eye_detector.detect(higher_right_gray)
			for (ex, ey, ew, eh) in right_eye:
				msg = OSCMessage("/r_eye")
				msg.append((ex, ey, ew, eh))
				client.send(msg)
			right_eye_detector.draw(higher_right_color, bgr=(200, 200, 250))

			left_eye = left_eye_detector.detect(higher_left_gray)
			for (lx, ly, lw, lh) in left_eye:
				msg = OSCMessage("/l_eye")
				msg.append((lx, ly, lw, lh))
				client.send(msg)

			left_eye_detector.draw(higher_left_color, bgr=(200, 200, 250))


			# eyes = eye_detector.detect(higher_gray)
			# eye_detector.draw(higher_color, bgr=(100, 200, 100))
			# for (xe, ye, we, he) in eyes:
			# 	# eye_detector.draw_one(frame, (x+xe, y+ye, we, he))
			# 	eye_gray = gray[y+ye:y+ye+he, x+xe:x+xe+we]
			# 	eye_color = frame[y+ye:y+ye+he, x+xe:x+xe+we]
			# # 	cl1 = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
			# 	clahe = cl1.apply(eye_gray)
			# 	## medianBlur the image to remove noise
			# 	blur = cv2.medianBlur(clahe, 7)
			# 	circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, 1, 100, param1=50, param2=5)
			# 	if circles is not None:
			# 		circles = np.round(circles[0, :]).astype("int")
			# 		for (xc,yc,rc) in circles:
			# 			cv2.circle(eye_color, (xc, yc), rc, (100, 200, 100), 1)

			# MOUTH
			lower_gray = gray[y + h / 2:y + h, x:x + w]
			lower_color = frame[y + h / 2:y + h, x:x + w]
			mouths = mouth_detector.detect(lower_gray)
			for (mx, my, mw, mh) in mouths:
				msg = OSCMessage("/mouth")
				msg.append((mx, my, mw, mh))
				client.send(msg)
			mouth_detector.draw(lower_color, bgr=(100, 100, 200))
			smiles = smile_detector.detect(lower_gray)
			smile_detector.draw(lower_color, bgr=(100, 100, 100))

			lower_third_gray = gray[y + h / 3:y + h, x:x + w]
			lower_third_color = frame[y + h / 3:y + h, x:x + w]
			nose_detector.detect(lower_third_gray)
			nose_detector.draw(lower_third_color, bgr=(250, 250, 250))

			n_smiles = len(smiles)
			msg = OSCMessage("/smile")
			msg.append(n_smiles)
			client.send(msg)

		cv2.imshow("camera", frame)

		# msg = OSCMessage()
		# msg.setAddress("/face_center")
		# msg.append(face_center)
		# client.sendto(msg,('127.0.0.1', 7300))
		# msg.clearData()

		if cv2.waitKey(100 / 12) & 0xff == ord("q"):
			break


def exit():
	camera.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	setup()
	loop()
	exit()
