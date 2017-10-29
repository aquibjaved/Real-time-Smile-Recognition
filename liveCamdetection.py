import dlib
import cv2
import numpy as np
import tensorflow as tf
import glob,os
import numpy as np
import time

detector = dlib.get_frontal_face_detector()
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
predictor = dlib.shape_predictor(PREDICTOR_PATH)  
(img_width, img_height) = (45, 25) 

W = tf.Variable(tf.zeros([1125, 2]),dtype = tf.float32, name="w1")
b = tf.Variable(tf.zeros([2]), dtype = tf.float32,name="bias")
saver = tf.train.Saver()

def pre_process(img):
	img1 = cv2.resize(img,(img_width, img_height)) # Resizing all the images
	image_test = np.array([img1])
	test_image_normalized = image_test.reshape(image_test.shape[0],-1)/255.
	return test_image_normalized

cap = cv2.VideoCapture(0)

print "Starting the session"

with tf.Session() as sess:
	saver.restore(sess, "./model/emotion_model")
	print("w1:", sess.run(W))
	print("bias:", sess.run(b))
	# print "time taken to load the model: ", (time.time() - start_time)
	while True:
		start_time = time.time()

		_,img_orig = cap.read()
		img = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
		face = detector(img)
		cordinates=[]
		for d in face:
			# print "left,top,right,bottom:", d.left(), d.top(), d.right(), d.bottom()
			pt1 = (int(d.left()), int(d.top()))
			pt2 = (int(d.right()), int(d.bottom()))
			# x,y,h,w = cv2.boundingRect(np.array([int(d.left()), int(d.top()), int(d.right()), int(d.bottom())]))
			cv2.rectangle(img,pt1,pt2,(0,255,0),2)
			dlib_rect = dlib.rectangle(d.left(), d.top(), d.right(), d.bottom()) 
	   		landmarks = np.matrix([[p.x, p.y] for p in predictor(img, dlib_rect).parts()]) 
	   		landmarks_display = landmarks[MOUTH_OUTLINE_POINTS]

	   		for idx, point in enumerate(landmarks_display):
	   			pos = (point[0, 0], point[0, 1])  
	   			(x, y, w, h) = cv2.boundingRect(np.array([(landmarks_display)]))
	   			# cv2.rectangle(img_orig,(x-8,y-10),(x+w+10,y+h+5),(0,255,0),2)
	   			roi_mouth = img[y:y-10+h+5,x:x-8+w+10]
	   			#cv2.imshow("mouth",roi_mouth)

	   			#prediction 
				test_image_normalized = pre_process(roi_mouth)
				x_ = tf.cast(test_image_normalized, tf.float32)
				y = tf.nn.softmax(tf.matmul(x_, W) + b)
				
				indx = sess.run(tf.argmax(y,1))
				if indx==1:
					print "smile"
				elif indx==0:
					print "neutral"
					
		print "time taken to detect face from dlib: ", (time.time() - start_time)

		cv2.imshow("Landmarks found", img_orig) 
	  	if cv2.waitKey(10) & 0xFF == ord('q'):
	  		cap.release()
	  		cv2.destroyAllWindows()
	  		break  

