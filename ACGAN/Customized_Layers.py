import tensorflow as tf;

def Generator_with_Vanilla(noise_Pattern, label_Pattern, hidden_Unit_Size = 128, output_Unit_Size = 28*28, is_Training = True, reuse = False):
    with tf.variable_scope('generator', reuse=reuse):
        hidden_Activation = tf.layers.dense(
            inputs = tf.concat([noise_Pattern, label_Pattern], axis = 1),
            units = hidden_Unit_Size,
            activation = tf.nn.relu,
            use_bias = True,
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer = tf.zeros_initializer(),
            name = "hidden"
            )
        ouptut_Activation = tf.layers.dense(
            inputs = hidden_Activation,
            units = output_Unit_Size,
            activation = tf.nn.sigmoid,
            use_bias = True,
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer = tf.zeros_initializer(),
            name = "output"
            )

        return ouptut_Activation;

def Discriminator_with_Vanilla(input_Pattern, hidden_Unit_Size = 128, label_Unit_Size = 10, is_Training  = True, reuse = False):
    with tf.variable_scope('discriminator', reuse=reuse):
        hidden_Activation = tf.layers.dense(
            inputs = input_Pattern,
            units = hidden_Unit_Size,
            activation = tf.nn.relu,
            use_bias = True,
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer = tf.zeros_initializer(),
            name = "hidden"
            )
        discrimination_Logits = tf.layers.dense(
            inputs = hidden_Activation,
            units = 1,
            activation = None,
            use_bias = True,
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer = tf.zeros_initializer(),
            name = "discrimination"
            )
        discrimination_Activation = tf.nn.sigmoid(discrimination_Logits);

        label_Logits = tf.layers.dense(
            inputs = hidden_Activation,
            units = label_Unit_Size,
            activation = None,
            use_bias = True,
            kernel_initializer = tf.truncated_normal_initializer(stddev=0.1),
            bias_initializer = tf.zeros_initializer(),
            name = "label"
            )
        label_Activation = tf.nn.softmax(label_Logits);

        return discrimination_Logits, label_Logits, discrimination_Activation, label_Activation;


def Generator_with_DCGAN(noise_Pattern, label_Pattern, is_Training = True, reuse = False):
    with tf.variable_scope('generator', reuse=reuse):
        pattern_Size = noise_Pattern.get_shape()[1] + label_Pattern.get_shape()[1];
        input_Pattern = tf.reshape(tf.concat([noise_Pattern, label_Pattern], axis = 1), (-1, 1, 1, pattern_Size));                
        #input_Pattern = tf.reshape(noise_Pattern, (-1, 1, 1, noise_Pattern.get_shape()[1]));
        
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

def Discriminator_with_DCGAN(image_Pattern, label_Unit_Size = 10, is_Training  = True, reuse = False):
    with tf.variable_scope('discriminator', reuse=reuse):
        convolution1_Activation = tf.nn.leaky_relu(
            tf.layers.conv2d(
                inputs = image_Pattern, 
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
        convolution5_Activation = tf.layers.conv2d(
            inputs = convolution4_Activation, 
            filters = 1 + label_Unit_Size, 
            kernel_size = [4,4],
            strides = (1,1),
            padding = "valid"
            )

        discrimination_Logit, label_Logit = tf.split(tf.reshape(convolution5_Activation, (-1, 1 + label_Unit_Size)), [1, label_Unit_Size], axis = 1);
        
        discrimination_Activation = tf.nn.sigmoid(discrimination_Logit);
        label_Activation = tf.nn.softmax(label_Logit);

        return discrimination_Logit, label_Logit, discrimination_Activation, label_Activation;