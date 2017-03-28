#!/usr/bin/env python
import cv2
from OSC import OSCMessage, OSCClient
from facial_features import FaceDetector, EyeDetector, SmileDetector, MouthDetector

client = OSCClient()
client.connect(("localhost", 7300))

window_height = 480
window_width = 640


def detect():
	# img = cv2.imread(filename)
	camera = cv2.VideoCapture(0)
	camera.set(3, window_width)
	camera.set(4, window_height)

	face_detector = FaceDetector()
	eye_detector = EyeDetector()
	# mouth_detector = MouthDetector()
	smile_detector = SmileDetector()

	while True:
		ret, frame = camera.read()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = face_detector.detect(gray)
		# face_detector.draw(frame, bgr=(200, 100, 100))
		for (x, y, w, h) in faces:
			
			face_detector.draw_one(frame, (x, y, w, h), (200, 100, 100))
			face_center = [(x + w / 2.) / window_width, (y + h / 2.) / window_height]
			msg = OSCMessage("/face_center")
			msg.append(face_center)
			client.send(msg)
			roi_gray = gray[y:y + h, x:x + w]
			roi_color = frame[y:y + h, x:x + w]
			eye_detector.detect(roi_gray)
			eye_detector.draw(roi_color, bgr=(100, 200, 100))
			# mouth_detector.detect(roi_gray)
			# mouth_detector.draw(roi_color, bgr=(100, 100, 200))
			smiles = smile_detector.detect(roi_gray)	
			smile_detector.draw(roi_color, bgr=(100, 100, 100))

			

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

	camera.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	detect()
