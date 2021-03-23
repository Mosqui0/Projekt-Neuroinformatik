# created own model based on the proposed model in HandSeg
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, LeakyReLU, Softmax, Add, Conv2DTranspose, Dense, Reshape, concatenate

# three classes: background: 0, right_hand: 1, left_hand: 2
def handseg_model(input_layer, num_classes=2):
  # use stride_val for projection_shortcut
  stride_val = 2
  
  # decode part ================================================================
  # first layer
  conv2d_1 = Conv2D(64, 3, strides=stride_val, padding='same', data_format='channels_last')(input_layer)
  leaky_relu_1 = LeakyReLU()(conv2d_1)

  # second layer
  conv2d_2 = Conv2D(128, 3, strides=stride_val, padding='same', data_format='channels_last')(leaky_relu_1)
  bn_2 = BatchNormalization()(conv2d_2)
  leaky_relu_2 = LeakyReLU()(bn_2)

  # third layer
  conv2d_3 = Conv2D(256, 3, strides=stride_val, padding='same', data_format='channels_last')(leaky_relu_2)
  bn_3 = BatchNormalization()(conv2d_3)
  leaky_relu_3 = LeakyReLU()(bn_3)

  # fourth layer
  conv2d_4 = Conv2D(512, 3, strides=stride_val, padding='same', data_format='channels_last')(leaky_relu_3)
  bn_4 = BatchNormalization()(conv2d_4)
  leaky_relu_4 = LeakyReLU()(bn_4)

  # encode part ================================================================
  # first layer
  # do the skip connection
  skip = Conv2DTranspose(256, 3, strides=stride_val, padding='same', data_format='channels_last')(leaky_relu_4)
  encode_input_1 = Add()([leaky_relu_3, skip])
  encode_bn_1 = BatchNormalization()(encode_input_1)
  encode_relu_1 = ReLU()(encode_bn_1)
      
  # second layer
  # do the skip connection
  skip = Conv2DTranspose(128, 3, strides=stride_val, padding='same', data_format='channels_last')(encode_relu_1)
  encode_input_2 = Add()([leaky_relu_2, skip])
  encode_bn_2 = BatchNormalization()(encode_input_2)
  encode_relu_2 = ReLU()(encode_bn_2)

  # third layer
  # do the skip connection
  skip = Conv2DTranspose(64, 3, strides=stride_val, padding='same', data_format='channels_last')(encode_relu_2)
  encode_input_3 = Add()([leaky_relu_1, skip])
  encode_bn_3 = BatchNormalization()(encode_input_3)
  encode_relu_3 = ReLU()(encode_bn_3)

  # last layer
  last_layer = Conv2DTranspose(1, 3, strides=stride_val, padding='same', data_format='channels_last')(encode_relu_3)

  #output = Conv2D(num_classes, 1, activation="softmax", data_format = 'channels_last')(last_layer)
  output = Dense(num_classes, activation='softmax')(last_layer)

  return output