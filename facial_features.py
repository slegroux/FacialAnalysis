import cv2

if 'cv' in dir(cv2):
	# <3.0
	cv2.CASCADE_DO_CANNY_PRUNING = cv2.cv.CV_HAAR_DO_CANNY_PRUNING
	cv2.CASCADE_FIND_BIGGEST_OBJECT = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
	cv2.CASCADE_DO_ROUGH_SEARCH = cv2.cv.CV_HAAR_DO_ROUGH_SEARCH
	cv2.FONT_HERSHEY_SIMPLEX = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 0.5, 0.5, 0, 1, cv2.cv.CV_AA)
	cv2.LINE_AA = cv2.cv.CV_AA

	def getTextSize(buf, font, scale, thickness):
		return cv2.cv.GetTextSize(buf, font)

	def putText(im, line, pos, font, scale, color, thickness, lineType):
		return cv2.cv.PutText(cv2.cv.fromarray(im), line, pos, font, color)

	cv2.getTextSize = getTextSize
	# cv2.putText = putText


class Detector(object):
	def __init__(self, cascade):
		self.cascade = cv2.CascadeClassifier(cascade)
		# print type(self.cascade), self.cascade.empty()
		if self.cascade.empty():
			print 'Warning: Could not load face cascade:', cascade
			raise SystemExit
		self.objects = []
		self.type = None
		self.scale_factor = 1.
		self.minNeighbors = 0
		self.minSize = (0, 0)
		# self.maxSize = (200, 200)

		# self.flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		# self.flags = cv2.cv.CV_HAAR_FIND_BIGGEST_OBJECT
		# self.flags = cv2.CASCADE_SCALE_IMAGE|cv2.CASCADE_DO_ROUGH_SEARCH
		self.flags = cv2.CASCADE_FIND_BIGGEST_OBJECT
		# self.flags = cv2.cv.CV_HAAR_SCALE_IMAGE
		# self.flags =0
		# not as accurate (bigger blobs) but faster
		# self.flags = cv2.cv.CV_HAAR_DO_ROUGH_SEARCH

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
			cv2.putText(frame, self.type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA, False)
			# cv2.putText(frame, self.type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA, False)

	def draw_one(self, frame, (x, y, w, h), bgr=(255, 0, 0), width=2):
		cv2.rectangle(
			frame,
			(x, y),
			(x + w, y + h),
			(bgr[0], bgr[1], bgr[2]),
			width
		)
		cv2.putText(frame, self.type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA, False)


class FaceDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_frontalface_default.xml')
		# Detector.__init__(self, './cascades/haarcascade_frontalface_alt_tree.xml')
		self.scale_factor = 1.1
		self.minNeighbors = 10
		self.minSize = (50, 50)
		self.type = "Face"
		# self.flags = cv2.cv.CV_HAAR_SCALE_IMAGE


class EyeDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_eye.xml')
		# self.scale_factor = 1.05
		# self.minNeighbors = 8
		# self.minSize = (20, 20)
		self.scale_factor = 1.3
		self.minNeighbors = 5
		# self.minSize = (10, 10)
		self.type = "Eye"

	def draw(self, frame, bgr=(255, 0, 0), width=2):
		for (x, y, w, h) in self.objects:
			center = (int(x + 0.5 * w), int(y + 0.5 * h))
			radius = int(0.3 * (w + h))
			cv2.circle(
				frame,
				center,
				radius,
				(bgr[0], bgr[1], bgr[2]),
				width
			)
			cv2.putText(frame, self.type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA, False)


class LeftEyeDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_mcs_lefteye.xml')
		self.scale_factor = 1.05
		self.minNeighbors = 40
		# self.minSize = (50, 50)
		self.type = "L_Eye"

	def draw(self, frame, bgr=(255, 0, 0), width=2):
		for (x, y, w, h) in self.objects:
			center = (int(x + 0.5 * w), int(y + 0.5 * h))
			radius = int(0.3 * (w + h))
			cv2.circle(
				frame,
				center,
				radius,
				(bgr[0], bgr[1], bgr[2]),
				width
			)
			cv2.putText(frame, self.type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA, False)


class RightEyeDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_mcs_righteye.xml')
		self.scale_factor = 1.05
		self.minNeighbors = 40
		# self.minSize = (50, 50)
		self.type = "R_Eye"

	def draw(self, frame, bgr=(255, 0, 0), width=2):
		for (x, y, w, h) in self.objects:
			center = (int(x + 0.5 * w), int(y + 0.5 * h))
			radius = int(0.3 * (w + h))
			cv2.circle(
				frame,
				center,
				radius,
				(bgr[0], bgr[1], bgr[2]),
				width
			)
			cv2.putText(frame, self.type, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA, False)


class LeftEarDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_mcs_leftear.xml')
		self.scale_factor = 1.3
		self.minNeighbors = 5
		self.minSize = (20, 20)
		self.type = "LEar"


class RightEarDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_mcs_rightear.xml')
		self.scale_factor = 1.3
		self.minNeighbors = 5
		self.minSize = (10, 10)
		self.type = "REar"


class NoseDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_mcs_nose.xml')
		self.scale_factor = 1.1
		self.minNeighbors = 10
		self.minSize = (30, 30)
		self.type = "Nose"


class MouthDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_mcs_mouth.xml')
		self.scale_factor = 1.05
		self.minNeighbors = 10
		self.minSize = (30, 30)
		self.type = "Mouth"


class SmileDetector(Detector):
	def __init__(self):
		Detector.__init__(self, './cascades/haarcascade_smile.xml')
		self.scale_factor = 1.2
		self.minNeighbors = 10
		self.minSize = (30, 30)
		self.type = "Smile"