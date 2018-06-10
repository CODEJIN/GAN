import tensorflow as tf;

def Generator(input_Pattern, is_Training = True, reuse = False):
    with tf.variable_scope('generator', reuse=reuse):
        input_Pattern = tf.reshape(input_Pattern, (-1, 1, 1, input_Pattern.get_shape()[1]));

        convolution1_Activation = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.layers.conv2d_transpose(
                    inputs = input_Pattern, 
                    filters = 1024, 
                    kernel_size = [4,4],
                    strides = (1,1),
                    padding = "valid"
                    ),
                training = is_Training
                )
            )
        convolution2_Activation = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.layers.conv2d_transpose(
                    inputs = convolution1_Activation, 
                    filters = 512, 
                    kernel_size = [4,4],
                    strides = (2,2),
                    padding = "same"
                    ),
                training = is_Training
                )
            )
        convolution3_Activation = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.layers.conv2d_transpose(
                    inputs = convolution2_Activation, 
                    filters = 256, 
                    kernel_size = [4,4],
                    strides = (2,2),
                    padding = "same"
                    ),
                training = is_Training
                )
            )
        convolution4_Activation = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.layers.conv2d_transpose(
                    inputs = convolution3_Activation, 
                    filters = 128, 
                    kernel_size = [4,4],
                    strides = (2,2),
                    padding = "same"
                    ),
                training = is_Training
                )
            )
        convolution5_Activation = tf.nn.tanh(
            tf.layers.conv2d_transpose(
                inputs = convolution4_Activation, 
                filters = 1, 
                kernel_size = [4,4],
                strides = (2,2),
                padding = "same"
                )
            )

        return convolution5_Activation;

def Discriminator(input_Pattern, is_Training  = True, reuse = False):
    with tf.variable_scope('discriminator', reuse=reuse):
        convolution1_Activation = tf.nn.leaky_relu(
            tf.layers.conv2d(
                inputs = input_Pattern, 
                filters = 128, 
                kernel_size = [4,4],
                strides = (2,2),
                padding = "same"
                )
            )
        convolution2_Activation = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.layers.conv2d(
                    inputs = convolution1_Activation, 
                    filters = 256, 
                    kernel_size = [4,4],
                    strides = (2,2),
                    padding = "same"
                    ),
                training = is_Training
                )
            )
        convolution3_Activation = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.layers.conv2d(
                    inputs = convolution2_Activation, 
                    filters = 512, 
                    kernel_size = [4,4],
                    strides = (2,2),
                    padding = "same"
                    ),
                training = is_Training
                )
            )
        convolution4_Activation = tf.nn.leaky_relu(
            tf.layers.batch_normalization(
                tf.layers.conv2d(
                    inputs = convolution3_Activation, 
                    filters = 1024, 
                    kernel_size = [4,4],
                    strides = (2,2),
                    padding = "same"
                    ),
                training = is_Training
                )
            )
        convolution5_Logit = tf.layers.conv2d(
            inputs = convolution4_Activation, 
            filters = 1, 
            kernel_size = [4,4],
            strides = (1,1),
            padding = "valid"
            )
        convolution5_Logit = tf.reshape(convolution5_Logit, (-1, 1))
        convolution5_Activation = tf.nn.sigmoid(convolution5_Logit);

        return convolution5_Logit, convolution5_Activation