import sys,cv2,os,random,main
import tensorflow as tf
import numpy as np

cascade_path =  "/Users/naka/anaconda3/envs/OpenCV/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_alt2.xml"

faceCascade = cv2.CascadeClassifier(cascade_path)

NAMES = {
    0: u"メッシ",
    2: u"クリスティアーノロナウド",
    1: u"ネイマール"}


def evaluation(img_path, ckpt_path):

    tf.reset_default_graph()

    f=open(img_path,'r')
    img=cv2.imread(img_path,cv2.IMREAD_COLOR)
    gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    face=faceCascade.detectMultiScale(gray, 1.1, 3)
    print(face)

    if len(face) > 0:
        for rect in face:
            random_str=str(random.random())
    

            cv2.rectangle(img,tuple(rect[0:2]), tuple(rect[0:2]+rect[2:4]),(0,0,255),thickness=2)
            face_detect_img_path = './statis-Flask/image/face_detect/'+random_str+'.jpg'

            cv2.imwrite(face_detect_img_path, img)
            x=rect[0]
            y=rect[1]
            w=rect[2]
            h=rect[3]

            cv2.imwrite('./statis-Flask/image/cut_face/' +random_str+'.jpg', img[y:y+h, x:x+w])
            target_image_path='./statis-Flask/image/cut_face/' +random_str+'.jpg'
        
    else:
        print('can not')
        return


    f =open(target_image_path,'r')
    image=[]
    img=cv2.imread(target_image_path)
    img=cv2.resize(img,(28,28))
    image.append(img.flatten().astype(np.float32)/255.0)
    image=np.asarray(image)

    logits=main.inference_deep(image, 1.0)

    sess=tf.InteractiveSession()
    saver=tf.train.Saver()
    sess.run(tf.initialize_all_variables())

    
    if ckpt_path:
        saver.restore(sess, ckpt_path)

        softmax=logits.eval()

        print('softmax-shape',softmax)
        result=softmax[0]

        rates=[round(n*100.0, 1) for n in result]

        humans=[]

        for index, rate in enumerate(rates):
            name=NAMES[index]
            humans.append({
                'label':index,
                'name':name,
                'rate':rate})

        rank=sorted(humans, key=lambda x:x['rate'],reverse=True)

        print(rank)
        return [rank, face_detect_img_path, target_image_path]


if __name__ == '__main__':
    imgfile = sys.argv[1]
    #print(imgfile)
    evaluation(imgfile,'./model.ckpt')
