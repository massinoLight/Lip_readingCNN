import cv2


# Opens the Video file
pathIn= './video/lip.mp4'
pathOut = './frame/'
def videoToFrame():

	cap = cv2.VideoCapture(pathIn)
	i = 0
	while (cap.isOpened()):
	    ret, frame = cap.read()
	    if ret == False:
	        break
	    cv2.imwrite(pathOut+'frame' + str(i) + '.jpg', frame)
	    i += 1

	cap.release()
	cv2.destroyAllWindows()
	
videoToFrame()	
