import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import skimage.io as io
import cv2
import os
import sys
from pycocotools.coco import COCO
import functools
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.layers import Conv2D
from pycocotools.coco import COCO
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import *
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))


COCO_ROOT = '../coco_unet_segmentation/coco_dataset/'
sys.path.insert(0, os.path.join(COCO_ROOT, 'cocoapi/PythonAPI'))


gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
  except RuntimeError as e:
    # Visible devices must be set at program startup
    print(e)

def down_block(
    input_tensor,
    no_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    kernel_initializer="he_normal",
    max_pool_window=(2, 2),
    max_pool_stride=(2, 2)
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        input_shape = (32,32,3),
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    # conv for skip connection
    conv = Activation("relu")(conv)

    pool = MaxPooling2D(pool_size=max_pool_window, strides=max_pool_stride)(conv)

    return conv, pool
def bottle_neck(
    input_tensor,
    no_filters,
    kernel_size=(3, 3),
    strides=(1, 1),
    padding="same",
    kernel_initializer="he_normal"
):
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    return conv
def up_block(    
    input_tensor,
    no_filters,
    skip_connection, 
    kernel_size=(3, 3),
    strides=(1, 1),
    upsampling_factor = (2,2),
    max_pool_window = (2,2),
    padding="same",
    kernel_initializer="he_normal"):
    
    
    conv = Conv2D(
        filters = no_filters,
        kernel_size= max_pool_window,
        strides = strides,
        activation = None,
        padding = padding,
        kernel_initializer=kernel_initializer
    )(UpSampling2D(size = upsampling_factor)(input_tensor))
    
    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv) 
    
    
    conv = concatenate( [skip_connection , conv]  , axis = -1)
    
    
    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)

    conv = Conv2D(
        filters=no_filters,
        kernel_size=kernel_size,
        strides=strides,
        activation=None,
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)

    conv = BatchNormalization(scale=True)(conv)

    conv = Activation("relu")(conv)
    
    return conv
def output_block(input_tensor,
    padding="same",
    kernel_initializer="he_normal"
):
    
    conv = Conv2D(
        filters=2,
        kernel_size=(3,3),
        strides=(1,1),
        activation="relu",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(input_tensor)
    
    
    conv = Conv2D(
        filters=1,
        kernel_size=(1,1),
        strides=(1,1),
        activation="sigmoid",
        padding=padding,
        kernel_initializer=kernel_initializer
    )(conv)
    
    
    return conv
    
def UNet(input_shape = (128,128,3)):
    
    filter_size = [64,128,256,512,1024]
    
    inputs = Input(shape = input_shape)
    
    d1 , p1 = down_block(input_tensor= inputs,
                         no_filters=filter_size[0],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    d2 , p2 = down_block(input_tensor= p1,
                         no_filters=filter_size[1],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    
    d3 , p3 = down_block(input_tensor= p2,
                         no_filters=filter_size[2],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    
    d4 , p4 = down_block(input_tensor= p3,
                         no_filters=filter_size[3],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal",
                         max_pool_window=(2,2),
                         max_pool_stride=(2,2))
    
    
    b = bottle_neck(input_tensor= p4,
                         no_filters=filter_size[4],
                         kernel_size = (3,3),
                         strides=(1,1),
                         padding="same",
                         kernel_initializer="he_normal")
    
    
    
    u4 = up_block(input_tensor = b,
                  no_filters = filter_size[3],
                  skip_connection = d4,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    u3 = up_block(input_tensor = u4,
                  no_filters = filter_size[2],
                  skip_connection = d3,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    u2 = up_block(input_tensor = u3,
                  no_filters = filter_size[1],
                  skip_connection = d2,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    u1 = up_block(input_tensor = u2,
                  no_filters = filter_size[0],
                  skip_connection = d1,
                  kernel_size=(3, 3),
                  strides=(1, 1),
                  upsampling_factor = (2,2),
                  max_pool_window = (2,2),
                  padding="same",
                  kernel_initializer="he_normal")
    
    
    
    output = output_block(input_tensor=u1 , 
                         padding = "same",
                         kernel_initializer= "he_normal")
    
    model = tf.keras.models.Model(inputs = inputs , outputs = output)
    
    
    return model
    

class CustomDataLoader(tf.keras.utils.Sequence):

    def __init__(self, annFile, image_dir, mask_dir, categories, batch_size, prefix=''):
        self.annFile = annFile
        self.coco = COCO(annFile)
        catIds = self.coco.getCatIds(catNms=categories)
        # Get the corresponding image ids and images using loadImgs
        imgIds = self.coco.getImgIds(catIds=catIds)
        self.images = self.coco.loadImgs(imgIds)
        self.images_count = len(self.images)
        self.image_dir  = image_dir
        self.batch_size = batch_size
        self.mask_dir = mask_dir
        self.ids = os.listdir(self.image_dir)
        self.image_size = 512
        self.prefix = prefix


    def __load__(self, id_name):
        
        image_path = os.path.join(self.image_dir , id_name)
        mask_path = os.path.join(self.mask_dir , self.prefix + id_name)
        
        image = cv2.imread(image_path , 1) # 1 specifies RGB format

        image = cv2.resize(image , (self.image_size , self.image_size)) # resizing before inserting to the network
        
        mask = cv2.imread(mask_path , -1)
        if mask is None:
            print(image_path, mask_path)
            input("Enter")
        mask = cv2.resize(mask , (self.image_size , self.image_size))
        mask = mask.reshape((self.image_size , self.image_size , 1))
        
       
        image = image / 255.0
        mask = mask / 255.0
        
        return image , mask

    def  __getitem__(self , index):
       
        if (index + 1)*self.batch_size > len(self.ids):
            self.batch_size = len(self.ids) - index * self.batch_size
            
        file_batch = self.ids[index * self.batch_size : (index + 1) * self.batch_size]
        
        images = []
        masks = []
        
        for id_name in file_batch : 
        
            _img , _mask = self.__load__(id_name)
            images.append(_img)
            masks.append(_mask)
        
        
        images = np.array(images)
        masks = np.array(masks)
        
        
        return images , masks
       
    def __len__(self):
         return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def on_epoch_end(self):
        pass      
    
if __name__ == "__main__":
    image_size = 128 
    epochs = 10
    batch_size = 8
    input_shape = (512, 512, 3)
    model = UNet(input_shape = (512,512,3))
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-4),run_eagerly=True, loss = 'binary_crossentropy', metrics = ['accuracy'])
    base_lr = 0.02
    train_ds = CustomDataLoader(annFile='../fiftyone/coco-2017/train/labels.json', image_dir='../fiftyone/coco-2017/train/data/', mask_dir='mask_train_2017', categories=['person'], batch_size=8, prefix='COCO_train2017_')
    val_ds = CustomDataLoader(annFile='../fiftyone/coco-2017/validation/labels.json', image_dir='../fiftyone/coco-2017/validation/data/', mask_dir='mask_val_2017', categories=['person'], batch_size=8, prefix='COCO_val2017_')
    train_steps =  len(os.listdir( "../fiftyone/coco-2017/train/data/"))/ 8
    model.fit_generator(train_ds , validation_data = val_ds , steps_per_epoch = train_steps , epochs=epochs)

    