from extract_lip import *


for i in range(2,9):

  PATH = '/home/massino/Bureau/M1BigData/s2/FouilleDonnées/projet/LipReadingCNN/dataset/dataset/F01/words/01/0'+str(i) 
  image_list = os.listdir(PATH)
  h=0
  for image_name in image_list: 
     extract_lip(PATH+'/'+image_name,h,i)
     h=h+1
