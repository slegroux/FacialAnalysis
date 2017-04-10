#!/usr/bin/env python
import cv2
import numpy as np
import sys
import math
from OSC import OSCMessage, OSCClient, OSCClientError
from facial_features import FaceDetector, SmileDetector, MouthDetector, NoseDetector, LeftEyeDetector, RightEyeDetector
import pygame




def setup():

	# OSC
	global client
	client = OSCClient()
	client.connect(("localhost", 7300))

	# DISPLAY
	global full_screen
	full_screen = False
	# global window_width, window_height
	window_width = 640
	window_height = 480

	# PYGAME
	global FPS
	FPS = 30
	pygame.init()
	global screen
	screen = pygame.display.set_mode((window_width, window_height))#, 0, 32)
	pygame.display.set_caption("Le Reuz")
	global clock
	clock = pygame.time.Clock()
	global logo, grass
	logo = pygame.image.load("pictures/lereuz.jpg").convert()
	grass = pygame.image.load("pictures/test5.jpg").convert()
	global index
	index = 0
	global fontObj
	fontObj = pygame.font.Font('freesansbold.ttf', 16)

	# OPENCV
	global camera
	camera = cv2.VideoCapture(0)
	camera.set(3, window_width)
	camera.set(4, window_height)
	# ds_factor = 1.0
	# logo = cv2.imread("pictures/lereuz.jpg")

	# FACIAL FEATURES DETECTORS
	global face_detector, mouth_detector, smile_detector, nose_detector, right_eye_detector, left_eye_detector
	face_detector = FaceDetector()
	mouth_detector = MouthDetector()
	smile_detector = SmileDetector()
	nose_detector = NoseDetector()
	right_eye_detector = RightEyeDetector()
	left_eye_detector = LeftEyeDetector()


def exit():
	pygame.quit()
	camera.release()
	cv2.destroyAllWindows()


def loop():
	global index, clock, logo, full_screen, grass

	while True:
		# clear screen
		screen.fill([255, 255, 255])

		# EVENTS
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				exit()
				sys.exit(0)
			if event.type is pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
				exit()
				sys.exit(0)
			if event.type is pygame.KEYDOWN and event.key == pygame.K_f:
				full_screen = not full_screen
				if full_screen:
					pygame.display.set_mode((screen.get_width(), screen.get_height()), pygame.FULLSCREEN)
				else:
					pygame.display.set_mode((screen.get_width(), screen.get_height()))#, 0, 32)


		ret, frame = camera.read()

		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		gray = cv2.equalizeHist(gray)

		# # FACE
		faces = face_detector.detect(gray)
		# face_detector.draw(frame, bgr=(200, 100, 100))
		for (x, y, w, h) in faces:
			# text = fontObj.render('Face', True, (200,100,100))
			# screen.blit(text, (x, y))
			# grass_new = pygame.transform.scale(grass, (w, h))
			# screen.blit(grass_new, (x, y))

			face_detector.draw_one(frame, (x, y, w, h), (200, 100, 100))
		# 	# face_center = [(x + w / 2.) / window_width, (y + h / 2.) / window_height]
			msg = OSCMessage("/face")
			msg.append((x, y, w, h))
			try:
				client.send(msg)
			except OSCClientError:
				print "Check that your OSC client is connected"

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
			n_smiles = len(smiles)
			msg = OSCMessage("/smile")
			msg.append(n_smiles)
			client.send(msg)

			lower_third_gray = gray[y + h / 3:y + h, x:x + w]
			lower_third_color = frame[y + h / 3:y + h, x:x + w]
			# nose_detector.detect(lower_third_gray)
			# nose_detector.draw(lower_third_color, bgr=(250, 250, 250))
			

		# PYGAME SCREEN
		frame = cv2.flip(frame, 1)
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
		frame = np.rot90(frame)
		frame = pygame.surfarray.make_surface(frame)
		frame.set_alpha(200)
		# frame.set_alpha(250)
		screen.blit(frame, (0, 0))

		# pygame.draw.rect(screen, (200, 200, 200), [100, 75, 200, 150],4)
		# pygame.draw.ellipse(screen, (250,100,100), (200, 150, 40, 80), 3)


		logo = pygame.transform.scale(logo, (100, 50))
		logo.set_alpha(200)
		# logo = pygame.transform.rotate(logo, index)
		# index += 1
		screen.blit(logo, (10, 20))
		
		text = fontObj.render('Sonomason', True, (250, 250, 250))
		screen.blit(text, (10, 0))
		text = fontObj.render("F: ToggleFullscreen, ESC: Quit", True, (250, 250, 250))
		screen.blit(text, (10, screen.get_height()-20))
		# textSurfaceObj = fontObj.render('Hello world!', True, (200,0,10), (0,100,200))
		# textRectObj = textSurfaceObj.get_rect()
		# textRectObj.center = (200, 150)
		# screen.blit(textSurfaceObj, textRectObj)

		# pygame.display.flip()
		pygame.display.update()
		# limit to 60 FPS
		clock.tick(FPS)


if __name__ == "__main__":
	setup()
	loop()
	exit()
