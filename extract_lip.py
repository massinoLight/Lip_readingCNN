import dlib
import cv2
import os


# Some constants
RESULT_PATH = './result'       # The path that the result images will be saved

LOG_PATH = 'log.txt'            # The path for the working log file
LIP_MARGIN = 0.3                # Marginal rate for lip-only image.
RESIZE = (64, 64)                # Final image size


def shape_to_list(shape):
    coords = []
    for i in range(0, 68):
        coords.append((shape.part(i).x, shape.part(i).y))
    return coords


def extract_lip(imag_path,ind,filename):
    logfile = open(LOG_PATH, 'w')
    # Face detector and landmark detector
    face_detector = dlib.get_frontal_face_detector()
    landmark_detector = dlib.shape_predictor(
        "detector/shape_predictor_68_face_landmarks.dat")

    frame_buffer = []
    frame_buffer_color = []

    img = cv2.imread(imag_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    frame_buffer.append(gray)
    frame_buffer_color.append(img)
    landmark_buffer = []

    face_rects = face_detector(gray, 1)             # Detect face
    if len(face_rects) < 1:                 # No face detected
        print("No face detected: ", imag_path+str(ind))
        logfile.write(imag_path + " : No face detected \r\n")

    if len(face_rects) > 1:                  # Too many face detected
        print("Too many face: ", imag_path+str(ind))
        logfile.write(imag_path + " : Too many face detected \r\n")
    else:
        rect = face_rects[0]                    # Proper number of face
        landmark = landmark_detector(gray, rect)   # Detect face landmarks
        landmark = shape_to_list(landmark)
        landmark_buffer.append(landmark)
        print("good: ",imag_path+str(ind))
        logfile.write(imag_path + " : Good \r\n")
        # Crop images
        cropped_buffer = []
        lip_landmark = landmark[48:68]  # Landmark corresponding to lip
        # Lip landmark sorted for determining lip region
        lip_x = sorted(lip_landmark, key=lambda pointx: pointx[0])
        lip_y = sorted(lip_landmark, key=lambda pointy: pointy[1])
        # Determine Margins for lip-only image
        x_add = int((-lip_x[0][0]+lip_x[-1][0])*LIP_MARGIN)
        y_add = int((-lip_y[0][1]+lip_y[-1][1])*LIP_MARGIN)
        crop_pos = (lip_x[0][0]-x_add, lip_x[-1][0]+x_add,
                    lip_y[0][1]-y_add, lip_y[-1][1]+y_add)   # Crop image
        cropped = frame_buffer_color[0][crop_pos[2]            :crop_pos[3], crop_pos[0]:crop_pos[1]]
        cropped = cv2.resize(
            cropped, (RESIZE[0], RESIZE[1]), interpolation=cv2.INTER_CUBIC)        # Resize
        cropped_buffer.append(cropped)
        # Save result
        print("# Save result")
        directory = RESULT_PATH + "/begin"+str(filename)+"/"
        if not os.path.exists(directory):
                os.makedirs(directory)
        cv2.imwrite(directory + str(ind) + ".jpg", cropped_buffer[0])
        print(directory + str(ind) + ".jpg")
        print("Save complited")
        logfile.close()
