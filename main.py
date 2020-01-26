import sys,cv2, random,os
import numpy as np
import tensorflow as tf
import tensorflow.python.platform

#import pdb; pdb.set_trace()


#import INPUT
#import model
#LOGDIR = './data'

NUM_CLASSES =3
IMAGE_SIZE = 28
IMAGE_PIXELS = IMAGE_SIZE*IMAGE_SIZE*3

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('train', './saveimage/Train/train.txt', 'File name of train data')
flags.DEFINE_string('test', './saveimage/Test/test.txt', 'File name of train data')
flags.DEFINE_string('train_dir', './saveimage', 'Directory to put the training data.')
flags.DEFINE_integer('max_steps', 100, 'Number of steps to run trainer.')
flags.DEFINE_integer('batch_size', 20, 'Batch size Must divide evenly into the dataset sizes.')
flags.DEFINE_float('learning_rate', 1e-4, 'Initial learning rate.')



def inference_deep(images_placeholder, keep_prob):

    x_image = tf.reshape(images_placeholder, [-1, IMAGE_SIZE, IMAGE_SIZE, 3])
    #print(x_image)

    with tf.name_scope('conv1') as scope:
        W_conv1 = weight_variable([5, 5, 3, 32])
        b_conv1 = bias_variable([32])
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        #print (h_conv1)

    with tf.name_scope('pool1') as scope:
        h_pool1 = max_pool_2x2(h_conv1)
        #print( h_pool1)

    with tf.name_scope('conv2') as scope:
        W_conv2 = weight_variable([5, 5, 32, 64])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
        #print (h_conv2)

    with tf.name_scope('pool2') as scope:
        h_pool2 = max_pool_2x2(h_conv2)
        #print( h_pool2)

    '''
    with tf.name_scope('conv3') as scope:
        W_conv3 = weight_variable([3, 3, 64, 128])
        b_conv3 = bias_variable([128])
        h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
        #print( h_conv3)

    with tf.name_scope('pool3') as scope:
        h_pool3 = max_pool_2x2(h_conv3)
        #print (h_pool3)
    '''
    
    with tf.name_scope('fc1') as scope:
        #w = int(IMAGE_SIZE / pow(2,3))
        W_fc1 = weight_variable([7*7*64,1024])
        b_fc1 = bias_variable([1024])
        print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
        #print(type(h_pool3))
        h_pool3_flat = tf.reshape(h_pool2,[-1,7*7*64])
        #print (h_pool3_flat)
        h_fc1 = tf.matmul(h_pool3_flat, W_fc1) + b_fc1
        h_fc1_drop = tf.nn.dropout(tf.nn.relu(h_fc1), keep_prob)
        #print (h_fc1_drop)

    with tf.name_scope('fc2') as scope:
        W_fc2 = weight_variable([1024, NUM_CLASSES])
        b_fc2 = bias_variable([NUM_CLASSES])
        h_fc2 = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
        #print( h_fc2)

    with tf.name_scope('softmax') as scope:
        y_conv=tf.nn.softmax(h_fc2)
        #print( y_conv)

    return y_conv

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def tf_print(tensor, name):
    print( name, tensor)
    return tensor

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')

def loss(logits, labels):
    cross_entropy = -tf.reduce_sum(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
    tf.summary.scalar("cross_entropy", cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    return train_step

def accuracy(logits, labels):
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    tf.summary.scalar("accuracy", accuracy)
    return accuracy


if __name__ == '__main__':
    f = open(FLAGS.train, 'r')

    train_image = []
    train_label = []

    for line in f:
        line = line.rstrip()
        l = line.split()

        img=cv2.imread(l[0])
        img=cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))

        train_image.append(img.flatten().astype(np.float32)/255.0)

        tmp=np.zeros(NUM_CLASSES)
        tmp[int(l[1])] = 1
        train_label.append(tmp)

    train_image=np.asarray(train_image)
    train_label=np.asarray(train_label)

    f.close()



    f=open(FLAGS.test, 'r')
    test_image=[]
    test_label=[]

    for line in f:

        line=line.rstrip()
        l=line.split()

        img=cv2.imread(l[0])
        img=cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))

        test_image.append(img.flatten().astype(np.float32)/255.0)

        tmp=np.zeros(NUM_CLASSES)
        tmp[int(l[1])]=1

        test_label.append(tmp)

    test_image=np.asarray(test_image)
    test_label=np.asarray(test_label)
    f.close()


    with tf.Graph().as_default():

        images_placeholder=tf.placeholder("float", shape=(None, IMAGE_PIXELS))
        labels_placeholder=tf.placeholder("float", shape=(None,NUM_CLASSES))
        keep_prob=tf.placeholder("float")

        logits=inference_deep(images_placeholder,keep_prob)
        loss_value=loss(logits, labels_placeholder)
        train_op=training(loss_value, FLAGS.learning_rate)
        acc=accuracy(logits,labels_placeholder)


        saver=tf.train.Saver()
        sess=tf.Session()
        sess.run(tf.initialize_all_variables())

        summary_op=tf.summary.merge_all()
        summary_writer=tf.summary.FileWriter(FLAGS.train_dir, sess.graph_def)



        for step in range(FLAGS.max_steps):
            for i in range(int(len(train_image)/FLAGS.batch_size)):

                batch=FLAGS.batch_size*i

                #print('bbbbbbbbbbb')
                sess.run(train_op, feed_dict={
                    images_placeholder: train_image[batch:batch+FLAGS.batch_size],
                    labels_placeholder: train_label[batch:batch+FLAGS.batch_size],keep_prob: 0.5})
                
            train_accuracy=sess.run(acc, feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob:1.0})

            print("step %d, training accuracy %g"%(step, train_accuracy))

            summary_str=sess.run(summary_op,feed_dict={
                images_placeholder: train_image,
                labels_placeholder: train_label,
                keep_prob:1.0})
            summary_writer.add_summary(summary_str, step)

    print("test accuracy %g"%sess.run(acc, feed_dict={
        images_placeholder: test_image,
        labels_placeholder: test_label,
        keep_prob:1.0}))

    save_path=saver.save(sess, "model.ckpt")

        
