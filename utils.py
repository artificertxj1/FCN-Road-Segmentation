import os
import sys
import re
import numpy as np
import random
import cv2
import scipy.misc
import zipfile
import shutil
import time
from glob import glob
import tensorflow as tf
from urllib.request import urlretrieve



def DLVgg(vggPath):
    ## load or download vgg16 model 
    vggFilename="vgg.zip"
    vggFilePath=os.path.join(vggPath,'vgg')
    vggFiles=[ os.path.join(vggFilePath, 'variables/variables.data-00000-of-00001'),
        os.path.join(vggFilePath, 'variables/variables.index'),
        os.path.join(vggFilePath, 'saved_model.pb')]
    missingFiles=[vgg_part for vgg_part in vggFiles if not os.path.exists(vgg_part)]
    if missingFiles:
        if os.path.exists(vggFilePath):
            shutil.rmtree(vggFilePath)
        os.makedirs(vggFilePath)
        print("Downloading pre-trained vgg weights....\n")
       
        urlretrieve('https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip',os.path.join(vggPath, vggFilename))
        print("Extracting vgg file.... \n")
        zip_ref=zipfile.ZipFile(os.path.join(vggFilePath,vggFilename),'r')
        zip_ref.extractall(vggPath)
        zip_ref.close()
        os.remove(os.path.join(vggFilePath, vggFilename))
    else:
        print("vgg files are ready to use\n")
        sys.exit()

def brightness(image, brightness):
    table=np.array([i+brightness for i in np.arange(0,256)])
    table[table<0]=0
    table[table>255]=255
    table=table.astype("uint8")
    return cv2.LUT(image, table)

def translate(image,x,y):
    M=np.float32([[1,0,x],[0,1,y]])
    rows, cols, _=image.shape
    return cv2.warpAffine(image, M, (cols, rows), borderMode=cv2.BORDER_REFLECT)

    
def makeBatch(dataPath, imageShape):
    ## vgg requires fixed image shape
    ## generate training batch and make the image shape fit vgg requirement
    def getBatches(batchSize):
        imageFiles=glob(os.path.join(dataPath, 'image_2','*.png'))
        labelFiles={
            re.sub(r'_(lane|road)_', '_', os.path.basename(path)): path
            for path in glob(os.path.join(dataPath, 'gt_image_2', '*_road_*.png'))}
        background_color=np.array([0,0,255])
        random.shuffle(imageFiles)
        images=[]
        gt_images=[]
        for batch_i in range(0, len(imageFiles), batchSize):
            for aug in range(10):
                for image_file in imageFiles[batch_i:batch_i+batchSize]:
                    gt_image_file=labelFiles[os.path.basename(image_file)]
                    image=scipy.misc.imresize(cv2.imread(image_file), imageShape)
                    gt_image=scipy.misc.imresize(cv2.imread(gt_image_file),imageShape)
                    # random translate x, y
                    x=random.randint(-100,100)
                    y=random.randint(-30,30)
                    image=translate(image,x,y)
                    gt_image=translate(gt_image,x,y)
                    # show image
                    #cv2.imshow('gt_img', gt_image)
                    #cv2.waitKey(1)
                    gt_bg=np.all(gt_image == background_color,axis=2)
                    gt_bg=gt_bg.reshape(*gt_bg.shape,1)
                    gt_image=np.concatenate((gt_bg,np.invert(gt_bg)),axis=2)
                    ## data augmentation
                    image=brightness(image,random.randint(-150,150))
                    #cv2.imshow('img',image)
                    #cv2.imshow('gt_image1',gt_image[:,:,1].reshape(-1,image.shape[1],1)*1.0)
                    #cv2.imshow('img_flip',image[:,::-1,:])
                    #cv2.imshow('gt_image_flip1',gt_image[:,::-1,1].reshape(-1,image.shape[1],1)*1.0)
                    #cv2.waitKey(1)
                    images.append(image)
                    gt_images.append(gt_image)
                    #images.append(images[:,::-1,:])
                    #gt_images.append(gt_images[:,::-1,])
                    if len(images)>=batchSize:
                        yield np.array(images), np.array(gt_images)
                        images=[]
                        gt_images=[]
                        
    return getBatches

def genTestOutput(sess, logits, keep_prob, image_pl, dataFolder, imageShape):
    for imageFile in glob(os.path.join(dataFolder, 'image_2','*.png')):
        image= scipy.misc.imresize(cv2.imread(imageFile), imageShape)
        im_softmax=sess.run(
                [tf.nn.softmax(logits)],
                {keep_prob: 1.0, image_pl: [image]})
        im_softmax=im_softmax[0][:,1].reshape(imageShape[0], imageShape[1])
        segmentation=(im_softmax>0.5).reshape(imageShape[0],imageShape[1], 1)
        mask =np.dot(segmentation, np.array([[0,255,0,127]]))
        mask =scipy.misc.toimage(mask, mode="RGBA")
        street_im= scipy.misc.toimage(image)
        street_im.paste(mask, box=None, mask=mask)
        yield os.path.basename(imageFile), np.array(street_im)
        
def save_inference_samples(runs_dir, data_dir, sess, imageShape, logits, keep_prob, input_image, run_label=""):
    output_dir=os.path.join(runs_dir, run_label+str(time.time()))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print('Training finished. Saving test images to: %s'%(output_dir))
    image_outputs=genTestOutput(sess, logits, keep_prob, input_image, os.path.join(data_dir,'data_road/testing'), imageShape)
    for name, image in image_outputs:
        scipy.misc.imsave(os.path.join(output_dir,name),image)