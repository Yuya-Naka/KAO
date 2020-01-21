import cv2
import numpy as np
import os
from math import ceil

in_jpg = "./saveimage/クリスティアーノロナウド/"
out_jpg = "./saveimage/クリスティアーノロナウド顔/"

cascades_dir = "/Users/naka/anaconda3/envs/OpenCV/lib/python3.7/site-packages/cv2/data/"

face_cascade = cv2.CascadeClassifier(os.path.join(cascades_dir,'haarcascade_frontalface_alt2.xml'))
eye_cascade = cv2.CascadeClassifier(os.path.join(cascades_dir,'haarcascade_eye.xml'))

class classifyPhoto:
    def __init__(self):
        print("init")

    def crop_face(salf, img_path):
        base_name= os.path.basename(img_path)
        name,ext = os.path.splitext(base_name)
        if (ext != '.jpg') and (ext != '.jpeg') :
            print('not a jpg image')
            return
        #print(img_path)
        #print(ext)
        img_src = cv2.imread(img_path,1)
        img_gray = cv2.cvtColor(img_src, cv2.COLOR_BGR2GRAY)

        cascade = face_cascade

        org_width = img_src.shape[1]
        org_height = img_src.shape[0]
        i = 0

        for j in range(-50, 51, 5):
            #print(j)
            big_img = np.zeros((org_height * 2, org_width * 2 ,3), np.uint8)
            big_img[ceil(org_height/2.0):ceil(org_height/2.0*3.0), ceil(org_width/2.0):ceil(org_width/2.0*3.0)] = img_src

            center = tuple(np.array([big_img.shape[1]*0.5, big_img.shape[0]*0.5]))
            size = tuple(np.array([big_img.shape[1], big_img.shape[0]]))
            angle = 5.0*float(j)
            scale = 1.0

            rotation_matrix = cv2.getRotationMatrix2D(center,angle,scale)
            img_rot = cv2.warpAffine(big_img, rotation_matrix, size, flags=cv2.INTER_CUBIC)
            rot_gray = cv2.cvtColor(img_rot, cv2.COLOR_BGR2GRAY)

            faces = cascade.detectMultiScale(img_rot, scaleFactor=1.2, minNeighbors=2, minSize=(50,50))
            #print(len(faces))
            if len(faces) > 0:
                for (x, y, w, h) in faces:
                    face = img_rot[y:y+h, x:x+w]
                    file_name = name + "_face_" + str(i) + ext
                    cv2.imwrite(out_jpg + file_name, face)
                    i += 1

            else:
                print('does not have any faces')

        return


def get_file(dir_path):
    filenames = os.listdir(dir_path)
    #print(filenames)
    return filenames

if __name__ == '__main__':
    classifier = classifyPhoto()
    pic = get_file(in_jpg)

    for i in pic:
        classifier.crop_face(in_jpg + i)
