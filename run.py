# Main libraries needed to handle images ---------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import subprocess
import cv2
from helpers import *

# Machine learning libraries ---------------------------------------------------
import tensorflow as tf
# Check if GPU is active, for laptops primarly.
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from keras.regularizers import l2

# ==============================================================================
# Basic parameters
# ==============================================================================

Images_load_length = 5000
path = '/run/media/boris/Elements/ML_data/Public/'
path_labels = '/run/media/boris/Elements/ML_data/Public/'
label_name = 'image_catalog2.0train.csv'

train_ratio = 0.95
crit_pixels = 1
EPOCHS = 10

Images_load_length_test = 2000

# ==============================================================================
# Labels loading
# ==============================================================================

Labels = load_csv_data(path_labels+label_name,12)[0:Images_load_length]

#We define the ratio at wich the lenses are still visible, flux can also be a deciding factor
Labels = (Labels >= crit_pixels)*1

# ==============================================================================
# Vis images loading + log stretch
# ==============================================================================
print('Visible images loading')
Images_vis = load_images(path , '/*.fits' , Images_load_length , False)
Images_vis = logarithmic_scale(Images_vis,Images_load_length)

# Image_number = 94
# plt.imshow(Images[Image_number],origin='lower')
# plt.show()
# ==============================================================================
# IR images loading + log stretch (with interpolation + stacking)
# ==============================================================================

print('IR images loading')
Images_IR = load_images(path , '/*.fits' , Images_load_length , True)
Images_IR = logarithmic_scale(Images_IR,Images_load_length)

# Image_number = 94
# plt.imshow(Images[Image_number],origin='lower')
# plt.show()

# ==============================================================================
# Image combination
# ==============================================================================
Images = Img_combine(Images_vis,Images_IR)

# ==============================================================================
# Split the train / test data into separate sets
# ==============================================================================

train_images = Images[0:np.int( Images_load_length*train_ratio),:,:,:]
test_images = Images[np.int(Images_load_length*train_ratio) :,:,:,:]

train_labels = Labels[0:np.int( Images_load_length*train_ratio )]
test_labels = Labels[np.int(Images_load_length*train_ratio) :]

#Create batches ??
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices((test_images, test_labels)).batch(32)

# ==============================================================================
# Define the neural network hierachy and classes
# ==============================================================================

# The model itself as a class (keras model subclasing API)
class MyModel(Model):
  def __init__(self):
    super(MyModel, self).__init__()
    self.conv1 = Conv2D(50, 3, activation='relu', kernel_regularizer=l2(1.42))
    self.conv2 = Conv2D(40, 3, activation='relu', kernel_regularizer=l2(1.42))
    self.conv3 = Conv2D(30, 3, activation='relu', kernel_regularizer=l2(1.42))
    self.flatten = Flatten()
    self.d1 = Dense(70,  activation='relu' , kernel_regularizer=l2(1.52))
    self.d2 = Dense(4,  activation='softmax', kernel_regularizer=l2(1.12))

  def call(self, x):
    x = self.conv1(x)
    x = self.conv2(x)
    x = self.conv3(x)
    x = self.flatten(x)
    x = self.d1(x)
    return self.d2(x)

# Create an instance of the model
model = MyModel()

# The optimizer / loss for the model
loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
optimizer = tf.keras.optimizers.Adam()

# Define the metrics for the loss
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')

# ==============================================================================
# Use the gradient method to learn
# ==============================================================================

@tf.function
def train_step(images, labels):
  with tf.GradientTape() as tape:
    predictions = model(images)
    loss = loss_object(labels, predictions)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))

  train_loss(loss)
  train_accuracy(labels, predictions)

@tf.function
def test_step(images, labels):
  predictions = model(images)
  t_loss = loss_object(labels, predictions)

  test_loss(t_loss)
  test_accuracy(labels, predictions)

# ==============================================================================
# Train the model
# ==============================================================================


for epoch in range(EPOCHS):

  # Reset the metrics for the next epoch
  train_loss.reset_states()
  train_accuracy.reset_states()
  test_loss.reset_states()
  test_accuracy.reset_states()

  for images, labels in train_ds:
    train_step(images, labels)

  for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        test_loss.result(),
                        test_accuracy.result()*100))



# ==============================================================================
# Laoding fresh data for clean validation
# ==============================================================================


print('Data loading for testing:')

Images_test_vis = load_images(path , '/*.fits' , Images_load_length_test , False, True , 3000)
Images_test_IR = load_images(path , '/*.fits' , Images_load_length_test , True, True , 3000)

Labels_test = load_csv_data(path_labels+label_name,12)[Images_load_length:Images_load_length_test+Images_load_length]
print('Done loading data')

# ==============================================================================
# Log strech of the test data
# ==============================================================================

print('Logarithmic strech :')

Images_test_vis = logarithmic_scale(Images_test_vis,Images_load_length_test)
Images_test_IR = logarithmic_scale(Images_test_IR,Images_load_length_test)

print('Logarithmic data strech done')

# ==============================================================================
# Image combination
# ==============================================================================
Images_test = Img_combine(Images_test_vis,Images_test_IR)

# ==============================================================================
# Process the data and get the test accuracy
# ==============================================================================

Labels_test = (Labels_test >= crit_pixels)*1

test_ds = tf.data.Dataset.from_tensor_slices((Images_test, Labels_test)).batch(32)

for test_images, test_labels in test_ds:
    test_step(test_images, test_labels)

template = 'Test Loss: {}, Test Accuracy: {}'
print(template.format(test_loss.result(),test_accuracy.result()*100))
