import cv2
import os

image_folder = '/home/massino/Bureau/M1BigData/s2/FouilleDonnées/projet/LipReadingCNN/dataset/dataset/F01/words/01/01'
video_name = 'video.avi'

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 1, (width,height))

for image in images:
    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()
