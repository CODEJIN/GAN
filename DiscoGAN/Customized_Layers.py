#Referred1: https://github.com/GunhoChoi/DiscoGAN-TF/blob/master/DiscoGAN.py
#Referred2: https://github.com/SKTBrain/DiscoGAN/blob/master/discogan/model.py

import tensorflow as tf;

def Generator(image_Pattern, is_Training = True, name = "generator", reuse = False):
    with tf.variable_scope(name, reuse=reuse):
        convolution_Activation = tf.nn.leaky_relu(
            tf.layers.conv2d(
                inputs = image_Pattern, 
                filters = 2 ** 6, 
                kernel_size = [4,4],
                strides = (2,2),
                padding = "same",
                use_bias = False,
                kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                )
            )

        for power in range(7, 10):
            convolution_Activation = tf.nn.leaky_relu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(
                        inputs = convolution_Activation, 
                        filters = 2 ** power, 
                        kernel_size = [4,4],
                        strides = (2,2),
                        padding = "same",
                        use_bias = False,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        ),
                    training = is_Training
                    )
                )

        convolution_Transpose_Activation = convolution_Activation;
        for power in reversed(range(6, 9)):
            convolution_Transpose_Activation = tf.nn.leaky_relu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d_transpose(
                        inputs = convolution_Transpose_Activation, 
                        filters = 2 ** power, 
                        kernel_size = [4,4],
                        strides = (2,2),
                        padding = "same",
                        use_bias = False,
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        ),
                    training = is_Training
                    )
                )

        generator_Logit = tf.layers.conv2d_transpose(
            inputs = convolution_Transpose_Activation, 
            filters = 3,    #RGB
            kernel_size = [4,4],
            strides = (2,2),
            padding = "same",
            use_bias = False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
            )
        
        generator_Activation = tf.nn.tanh(generator_Logit);

        return generator_Logit, generator_Activation;

def Discriminator(image_Pattern, is_Training  = True, name = "discriminator", reuse = False):
    with tf.variable_scope(name, reuse=reuse):
        convolution_Activation = tf.nn.leaky_relu(
                tf.layers.conv2d(
                    inputs = image_Pattern,                 
                    filters = 2 ** 5, 
                    kernel_size = [4,4],
                    strides = (2,2),
                    padding = "same",
                    use_bias = False,
                    kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                    )
                )

        for power in range(6, 10):
            convolution_Activation = tf.nn.leaky_relu(
                tf.layers.batch_normalization(
                    tf.layers.conv2d(
                        inputs = convolution_Activation,                 
                        filters = 2 ** power, 
                        kernel_size = [4,4],
                        strides = (2,2),
                        padding = "same",
                        use_bias = False,                    
                        kernel_initializer=tf.truncated_normal_initializer(stddev=0.02),
                        ),
                    training = is_Training
                    )                
                )                
                
        discriminator_Logit = tf.layers.dense(
            inputs = tf.contrib.layers.flatten(convolution_Activation),
            units = 1,
            use_bias = False,
            kernel_initializer=tf.truncated_normal_initializer(stddev=0.02)
            )

        return discriminator_Logit;