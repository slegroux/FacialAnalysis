import cv2


class Detector(object):
	def __init__(self, cascade):
		self.cascade = cv2.CascadeClassifier(cascade)
		# print type(self.cascade), self.cascade.empty()
		if self.cascade.empty():
			print 'Warning: Could not load face cascade:', cascade
			raise SystemExit
		self.objects = []
		self.type = None
		self.scale_factor = None
		self.minNeighbors = None
		self.minSize = (None, None)
		# self.flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		self.flags = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT

	def detect(self, image):
		self.objects = self.cascade.detectMultiScale(
			image,
			scaleFactor=self.scale_factor,
			minNeighbors=self.minNeighbors,
			minSize=self.minSize,
			flags=self.flags
		)
		# print "scale", self.cascade, self.scale_factor
		return self.objects

	def draw(self, frame, bgr=(255, 0, 0), width=2):
		for (x, y, w, h) in self.objects:
			cv2.rectangle(
				frame,
				(x, y),
				(x + w, y + h),
				(bgr[0], bgr[1], bgr[2]),
				width
			)
			cv2.putText(frame, self.type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.CV_AA, False)

	def draw_one(self, frame, (x, y, w, h), bgr=(255, 0, 0), width=2):
		cv2.rectangle(
			frame,
			(x, y),
			(x + w, y + h),
			(bgr[0], bgr[1], bgr[2]),
			width
		)
		cv2.putText(frame, self.type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.CV_AA, False)


class FaceDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_frontalface_default.xml')
		self.scale_factor = 1.1
		self.minNeighbors = 5
		self.minSize = (80, 80)
		self.type = "Face"


class EyeDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_eye.xml')
		self.scale_factor = 1.05
		self.minNeighbors = 8
		self.minSize = (50, 50)
		self.type = "Eye"


class MouthDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_mcs_mouth.xml')
		self.scale_factor = 1.05
		self.minNeighbors = 5
		self.minSize = (50, 50)
		self.type = "Mouth"


class SmileDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_smile.xml')
		self.scale_factor = 1.7
		self.minNeighbors = 22
		self.minSize = (25, 25)
		self.type = "Smile"