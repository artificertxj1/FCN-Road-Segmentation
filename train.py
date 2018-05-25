# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 01:02:29 2018

@author: justin tian
"""

import os
import sys
import tensorflow as tf
import utils
import matplotlib.pyplot as plt
import numpy as np
import cv2
import scipy
import scipy.misc

## detect GPU availability
if not tf.test.gpu_device_name():
    sys.stderr.write("Error: no gpu device detected\n")
    sys.exit()
else:
    print("Default GPU device: %s"%(tf.test.gpu_device_name()))

#KEEP_PROB=0.5
#EPOCHS=50
#BATCH_SIZE=16
def kernel_initializer():
    return tf.truncated_normal_initializer(stddev=0.01)

def loadVgg(sess, vggPath):
    vgg_tag='vgg16'
    vgg_input_tensor_name='image_input:0'
    vgg_keep_prob_tensor_name="keep_prob:0"
    vgg_layer3_out_tensor_name='layer3_out:0'
    vgg_layer4_out_tensor_name='layer4_out:0'
    vgg_layer7_out_tensor_name='layer7_out:0'
    model=tf.saved_model.loader.load(sess, [vgg_tag], vggPath)
    _input=tf.get_default_graph
    keep_prob=tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name)
    l3_out=tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name)
    l4_out=tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name)
    l7_out=tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)
    #print(l7_out.get_shape())
    return _input, keep_prob, l3_out, l4_out, l7_out

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    ## inputs are skip layers used in up-sampling
    l7_conv=tf.layers.conv2d(vgg_layer7_out, num_classes,(1,1),(1,1), kernel_initializer=kernel_initializer())
    l4_conv=tf.layers.conv2d(vgg_layer4_out, num_classes,(1,1),(1,1), kernel_initializer=kernel_initializer())
    l3_conv=tf.layers.conv2d(vgg_layer3_out, num_classes,(1,1),(1,1), kernel_initializer=kernel_initializer())
    
    l7_deconv=tf.layers.conv2d_transpose(l7_conv, num_classes, (4,4), (2,2), padding='SAME', kernel_initializer=kernel_initializer())
    l4_sum=tf.add(l7_deconv, l4_conv)
    l4_deconv=tf.layers.conv2d_transpose(l4_sum, num_classes,  (4,4), (2,2), padding='SAME', kernel_initializer=kernel_initializer())
    l3_sum=tf.add(l4_deconv,  l3_conv)
    out = tf.layers.conv2d_transpose(l3_sum, num_classes, (16,16), (8,8), padding='SAME', kernel_initializer=kernel_initializer())
    return out

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels = labels))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    return logits, train_op, loss


    
def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
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
    plt.plot(samples_plot,loss_plot, 'ro')
    plt.savefig('runs/E%04d-B%04d-K%f.png'%(EPOCHS, BATCH_SIZE, KEEP_PROB))
    with open('runs/E%04d-B%04d-K%f.txt'%(EPOCHS, BATCH_SIZE, KEEP_PROB),'w') as f:
        for s,l in zip(samples_plot,loss_plot):
            f.write("%f\t%f\n"%(s,l))
    # plt.show()
    
    
    
def run():
    global EPOCHS, KEEP_PROB, BATCH_SIZE
    #if len(sys.argv)>1:
    EPOCHS =200 #int(sys.argv[1])
    #f len(sys.argv)>2:
    BATCH_SIZE = 2 #int(sys.argv[2])
    #if len(sys.argv)>3:
    KEEP_PROB = 0.05#float(sys.argv[3])
    num_classes = 2
    image_shape = (100, 360)
    data_dir = './data'
    runs_dir = './runs'
    run_label = 'E%04d-B%04d-K%f_'%(EPOCHS, BATCH_SIZE, KEEP_PROB)
    

    # Download pretrained vgg model
    #utils.DLVgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/
    config = tf.ConfigProto()
    config.gpu_options.allocator_type = 'BFC'

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join('./', 'vgg')
        # Create function to get batches
        get_batches_fn = utils.makeBatch(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        correct_label = tf.placeholder(tf.int32)
        learning_rate = tf.placeholder(tf.float32)

        # Build NN using load_vgg, layers, and optimize function
        _input, keep_prob, l3_out, l4_out, l7_out = loadVgg(sess, vgg_path)
        last_layer = layers(l3_out, l4_out, l7_out, num_classes)
        logits, train_op, cross_entropy_loss = optimize(last_layer, correct_label, learning_rate, num_classes)

        #Train NN using the train_nn function
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess.run(tf.global_variables_initializer())

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        train_nn(sess, EPOCHS, BATCH_SIZE, get_batches_fn, train_op, cross_entropy_loss, _input, correct_label, keep_prob, learning_rate)

        # Save the variables to disk.
        save_path = saver.save(sess, "./runs/model_E%04d-B%04d-K%f.ckpt"%(EPOCHS, BATCH_SIZE, KEEP_PROB))
        print("Model saved in file: %s" % save_path)

        # Save inference data using helper.save_inference_samples
        #helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, _input)#, run_label=run_label)
        # OPTIONAL: Apply the trained model to a video

        #video_file='0002-20170519-2.mp4'
        #gen_test_output_video(sess, logits, keep_prob, _input, video_file, image_shape)

if __name__ == '__main__':
    run()
    
    
    