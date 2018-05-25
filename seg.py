# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 23:51:38 2018

@author: justin tian
"""

import re
import random
import scipy.misc
import time
import zipfile
import shutil
import os.path
import tensorflow as tf
#import helper
import warnings
from distutils.version import LooseVersion
#import project_tests as tests
import matplotlib.pyplot as plt
#from tqdm import tqdm
import numpy as np
import sys
import cv2
import scipy
from glob import glob
# Check TensorFlow Version

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def brigthness(image, brigthness):
    table = np.array([i+ brigthness    for i in np.arange(0, 256)])
    table[table<0]=0
    table[table>255]=255
    table=table.astype("uint8")
    # apply gamma correction using the lookup table
    return cv2.LUT(image, table)

def translate(image,x,y):
    M = np.float32([[1,0,x],[0,1,y]])
    rows,cols,_ = image.shape
    return cv2.warpAffine(image,M,(cols,rows),borderMode=cv2.BORDER_REFLECT)

def gen_batch_function(data_folder, image_shape):
    """
    Generate function to create batches of training data
    :param data_folder: Path to folder that contains all the datasets
    :param image_shape: Tuple - Shape of image
    :return:
    """
    def get_batches_fn(batch_size):
        """
        Create batches of training data
        :param batch_size: Batch Size
        :return: Batches of training data
        """
        image_paths = glob(os.path.join(data_folder, 'image_2', '*.png'))
        label_paths = {
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(data_folder, 'gt_image_2', '*_road_*.png'))}
        background_color = np.array([0, 0, 255])

        random.shuffle(image_paths)
        images = []
        gt_images = []
        for batch_i in range(0, len(image_paths), batch_size):
            for aug in range(10):
                for image_file in image_paths[batch_i:batch_i+batch_size]:
                    gt_image_file = label_paths[os.path.basename(image_file)]

                    image = scipy.misc.imresize(cv2.imread(image_file), image_shape)
                    gt_image = scipy.misc.imresize(cv2.imread(gt_image_file), image_shape)

                    # Random translate x,y
                    x = random.randint(-100,100)
                    y = random.randint(-30,30)

                    image = translate(image,x,y)
                    gt_image = translate(gt_image,x,y)

                    # cv2.imshow('gt_img',gt_image)
                    gt_bg = np.all(gt_image == background_color, axis=2)
                    gt_bg = gt_bg.reshape(*gt_bg.shape, 1)
                    gt_image = np.concatenate((gt_bg, np.invert(gt_bg)), axis=2)

                    # Augmentation
                    image=brigthness(image, random.randint(-150,150))
                    #cv2.imshow('img',image)
                    #cv2.imshow('gt_image1',gt_image[:,:,1].reshape(-1,image.shape[1],1)*1.0)
                    #cv2.imshow('img_flip',image[:,::-1,:])
                    #cv2.imshow('gt_image_flip1',gt_image[:,::-1,1].reshape(-1,image.shape[1],1)*1.0)
                    #cv2.waitKey(1)

                    # print("image",image.shape,image.dtype,np.min(image),np.max(image))
                    # print("gt_image",gt_image.shape,gt_image.dtype,np.min(gt_image),np.max(gt_image))

                    images.append(image)
                    gt_images.append(gt_image)

                    images.append(image[:,::-1,:])
                    gt_images.append(gt_image[:,::-1,:])

                    if len(images)>=batch_size:
                        yield np.array(images), np.array(gt_images)
                        images=[]
                        gt_images=[]
    return get_batches_fn


def gen_test_output(sess, logits, keep_prob, image_pl, data_folder, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param data_folder: Path to the folder that contains the datasets
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    for image_file in glob(os.path.join(data_folder, 'image_2', '*.png')):
        image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask = scipy.misc.toimage(mask, mode="RGBA")
        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        yield os.path.basename(image_file), np.array(street_im)

def save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image, run_label=""):
    # Make folder for current run
    output_dir = os.path.join(runs_dir, run_label + str(time.time()))
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Run NN on test images and save them to HD
    print('Training Finished. Saving test images to: {}'.format(output_dir))
    image_outputs = gen_test_output(
        sess, logits, keep_prob, input_image, os.path.join(data_dir, 'data_road/testing'), image_shape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir, name), image)


#KEEP_PROB = 0.5
#EPOCHS = 50
#BATCH_SIZE = 16
def kernel_initializer():
    return tf.truncated_normal_initializer(stddev=0.01)


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'

    model = tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)

    _input = tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name)
    l3_out = tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4_out = tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7_out = tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)

    return _input, keep_prob, l3_out, l4_out, l7_out
#tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # 1x1 conv layers
    l7_conv = tf.layers.conv2d(vgg_layer7_out, num_classes, (1,1), (1,1), kernel_initializer=kernel_initializer())
    l4_conv = tf.layers.conv2d(vgg_layer4_out, num_classes, (1,1), (1,1), kernel_initializer=kernel_initializer())
    l3_conv = tf.layers.conv2d(vgg_layer3_out, num_classes, (1,1), (1,1), kernel_initializer=kernel_initializer())

    # Deconv layers
    l7_deconv = tf.layers.conv2d_transpose(l7_conv, num_classes,(4,4), (2,2), padding='SAME', kernel_initializer=kernel_initializer())

    # Add l7 deconv output to l4 output
    l4_sum = tf.add(l7_deconv, l4_conv)
    l4_deconv = tf.layers.conv2d_transpose(l4_sum, num_classes, (4,4), (2,2), padding='SAME', kernel_initializer=kernel_initializer())

    # Add l4 deconv output to l3 output
    l3_sum = tf.add(l4_deconv, l3_conv)
    out = tf.layers.conv2d_transpose(l3_sum, num_classes, (16,16), (8,8), padding='SAME', kernel_initializer=kernel_initializer())

    return out
#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """

    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labels))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, train_op, loss
#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    loss = 99999
    samples_plot=[]
    loss_plot=[]
    sample=0
    for epoch in range(epochs):
        counter = 0
        for image, image_c in get_batches_fn(batch_size):
            _,loss = sess.run([train_op, cross_entropy_loss], feed_dict={
                input_image: image,
                correct_label: image_c,
                keep_prob: KEEP_PROB,
                learning_rate: 0.0001
            })
            samples_plot.append(sample)
            loss_plot.append(loss)
            sample = sample + batch_size
            print("#%4d  (%10d): %.20f"%(counter, sample, loss))
            # if counter > 10:
            #     break
            counter = counter + 1
        print("%4d/%4d Loss: %f"%(epoch,epochs,loss))
    #plt.plot(samples_plot,loss_plot, 'ro')
    #plt.savefig('runs/E%04d-B%04d-K%f.png'%(EPOCHS, BATCH_SIZE, KEEP_PROB))
    #with open('runs/E%04d-B%04d-K%f.txt'%(EPOCHS, BATCH_SIZE, KEEP_PROB),'w') as f:
        #for s,l in zip(samples_plot,loss_plot):
            #f.write("%f\t%f\n"%(s,l))
    # plt.show()
#tests.test_train_nn(train_nn)


def gen_test_output_video(sess, logits, keep_prob, image_pl, video_file, image_shape):
    """
    Generate test output using the test images
    :param sess: TF session
    :param logits: TF Tensor for the logits
    :param keep_prob: TF Placeholder for the dropout keep robability
    :param image_pl: TF Placeholder for the image placeholder
    :param image_shape: Tuple - Shape of image
    :return: Output for for each test image
    """
    cap = cv2.VideoCapture(video_file)
    counter=0
    while True:
        ret, frame = cap.read()
        if frame is None:
            break
        image = scipy.misc.imresize(frame, image_shape)

        im_softmax = sess.run(
            [tf.nn.softmax(logits)],
            {keep_prob: 1.0, image_pl: [image]})
        im_softmax = im_softmax[0][:, 1].reshape(image_shape[0], image_shape[1])
        segmentation = (im_softmax > 0.5).reshape(image_shape[0], image_shape[1], 1)
        mask = np.dot(segmentation, np.array([[0, 255, 0, 127]]))
        mask_full = scipy.misc.imresize(mask, frame.shape)
        mask_full = scipy.misc.toimage(mask_full, mode="RGBA")
        mask = scipy.misc.toimage(mask, mode="RGBA")


        street_im = scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)

        street_im_full = scipy.misc.toimage(frame)
        street_im_full.paste(mask_full, box=None, mask=mask_full)

        cv2.imwrite("4k-result/4k_image%08d.jpg"%counter,np.array(street_im_full))
        counter=counter+1

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

def main():
    global EPOCHS, KEEP_PROB, BATCH_SIZE
    #if len(sys.argv)>1:
    EPOCHS = 50 #int(sys.argv[1])
    #if len(sys.argv)>2:
    BATCH_SIZE = 5 #int(sys.argv[2])
    #if len(sys.argv)>3:
    KEEP_PROB = 0.05 #float(sys.argv[3])
    num_classes = 2
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    run_label = 'E%04d-B%04d-K%f_'%(EPOCHS, BATCH_SIZE, KEEP_PROB)
    #tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    #helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    #config = tf.ConfigProto()
    #config.gpu_options.allocator_type = 'BFC'
    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join('./', 'vgg')
        # Create function to get batches
        print("starting making batch creator\n")
        get_batches_fn = gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        print("batch creator made\n")	
        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        # Build NN using load_vgg, layers, and optimize function
        print("start loading vgg16 model\n")
        _input, keep_prob, l3_out, l4_out, l7_out = load_vgg(sess, vgg_path)
        last_layer = layers(l3_out, l4_out, l7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)
        print("loading finished\n")
        #Train NN using the train_nn function
        sess.run(tf.global_variables_initializer())

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, _input, correct_label, keep_prob, learning_rate)

        # Save the variables to disk.
        save_path = saver.save(sess, "./runs/model_E%04d-B%04d-K%f.ckpt"%(EPOCHS, BATCH_SIZE, KEEP_PROB))
        print("Model saved in file: %s" % save_path)

        # Save inference data using helper.save_inference_samples
        save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, _input)#, run_label=run_label)
        # OPTIONAL: Apply the trained model to a video

        #video_file='0002-20170519-2.mp4'
        #gen_test_output_video(sess, logits, keep_prob, _input, video_file, image_shape)

if __name__ == '__main__':
    main()
