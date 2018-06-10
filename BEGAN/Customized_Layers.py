#https://github.com/carpedm20/BEGAN-tensorflow/blob/master/trainer.py#L189
#https://blog.heuritech.com/2017/04/11/began-state-of-the-art-generation-of-faces-with-generative-adversarial-networks/

import tensorflow as tf;
import numpy as np;

def Generator(input_Pattern, image_Size = 32, filter_Count = 128, repeat_Count = 2, reuse = False):
    with tf.variable_scope('generator', reuse=reuse) as scope:
        fullConnection1_Activation = tf.layers.dense(
            inputs = input_Pattern,
            units = filter_Count * 8 * 8,
            activation = None,
            use_bias = True,
            )
        convolution_Activation = tf.reshape(fullConnection1_Activation, shape=[-1, 8, 8, filter_Count]);

        current_Convolution_Size = 8;
        while current_Convolution_Size < image_Size:
            for index in range(repeat_Count):            
                convolution_Activation = tf.layers.conv2d(
                    inputs = convolution_Activation, 
                    filters = filter_Count, 
                    kernel_size = [3, 3],
                    strides = (1, 1),
                    padding = "same",
                    activation = tf.nn.elu
                    )
            convolution_Activation = tf.image.resize_nearest_neighbor(convolution_Activation, (current_Convolution_Size * 2, current_Convolution_Size * 2))   #In paper, the filter size increased after up-sampling. But I cannot understand about this....
            current_Convolution_Size *= 2;
        
        for index in range(repeat_Count):            
            convolution_Activation = tf.layers.conv2d(
                inputs = convolution_Activation, 
                filters = filter_Count, 
                kernel_size = [3, 3],
                strides = (1, 1),
                padding = "same",
                activation = tf.nn.elu
                )

        output_Activation = tf.layers.conv2d(
            inputs = convolution_Activation, 
            filters = 3, 
            kernel_size = [3, 3],
            strides = (1, 1),
            padding = "same",
            activation = tf.nn.elu
            )

    variables_List = tf.contrib.framework.get_variables(scope)
    return output_Activation, variables_List;
    
def Discriminator(image_Pattern, embedding_Size = 100, filter_Count = 128, repeat_Count = 2, reuse = False):
    with tf.variable_scope('discriminator', reuse=reuse) as scope:
        image_Size = image_Pattern.get_shape()[1].value;

        #Encoder
        encoder_Activation = tf.layers.conv2d(
            inputs = image_Pattern, 
            filters = filter_Count, 
            kernel_size = [3, 3],
            strides = (1, 1),
            padding = "same",
            activation = tf.nn.elu
            )
                
        current_Convolution_Size = image_Size;
        current_Filter_Magnification = 1;
        while current_Convolution_Size >= 8:
            for index in range(repeat_Count):            
                encoder_Activation = tf.layers.conv2d(
                    inputs = encoder_Activation, 
                    filters = filter_Count * current_Filter_Magnification if index < repeat_Count - 1 else filter_Count * (current_Filter_Magnification + 1), 
                    kernel_size = [3, 3],
                    strides = (1, 1) if index < repeat_Count - 1 else (2, 2),
                    padding = "same",
                    activation = tf.nn.elu
                    )
            current_Convolution_Size //= 2;
            current_Filter_Magnification += 1;

        for index in range(repeat_Count):            
            encoder_Activation = tf.layers.conv2d(
                inputs = encoder_Activation, 
                filters = filter_Count * current_Filter_Magnification, 
                kernel_size = [3, 3],
                strides = (1, 1),
                padding = "same",
                activation = tf.nn.elu
                )

        embedding_Activation = tf.layers.dense(
            inputs = tf.contrib.layers.flatten(encoder_Activation),
            units = embedding_Size,
            activation = None,
            use_bias = True,
            )

        #Decoder
        decoder_Activation = tf.layers.dense(
            inputs = embedding_Activation,
            units = filter_Count * 8 * 8,
            activation = None,
            use_bias = True,
            )
        decoder_Activation = tf.reshape(decoder_Activation, shape=[-1, 8, 8, filter_Count]);

        current_Convolution_Size = 8;
        while current_Convolution_Size < image_Size:
            for index in range(repeat_Count):            
                decoder_Activation = tf.layers.conv2d(
                    inputs = decoder_Activation, 
                    filters = filter_Count, 
                    kernel_size = [3, 3],
                    strides = (1, 1),
                    padding = "same",
                    activation = tf.nn.elu
                    )
            decoder_Activation = tf.image.resize_nearest_neighbor(decoder_Activation, (current_Convolution_Size * 2, current_Convolution_Size * 2))   #In paper, the filter size increased after up-sampling. But I cannot understand about this....
            current_Convolution_Size *= 2;
        
        for index in range(repeat_Count):            
            decoder_Activation = tf.layers.conv2d(
                inputs = decoder_Activation, 
                filters = filter_Count, 
                kernel_size = [3, 3],
                strides = (1, 1),
                padding = "same",
                activation = tf.nn.elu
                )

        output_Activation = tf.layers.conv2d(
            inputs = decoder_Activation, 
            filters = 3, 
            kernel_size = [3, 3],
            strides = (1, 1),
            padding = "same",
            activation = tf.nn.elu
            )

    variables_List = tf.contrib.framework.get_variables(scope)
    return output_Activation, variables_List;
