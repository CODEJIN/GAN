#Semi-Supervised GAN
#https://www.youtube.com/watch?v=odpjk7_tGY0 40:00


#Referred: https://github.com/znxlwm/tensorflow-MNIST-GAN-DCGAN/blob/master/tensorflow_MNIST_DCGAN.py

import numpy as np;
import tensorflow as tf;
from tensorflow.examples.tutorials.mnist import input_data;
import time, os;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
import matplotlib.gridspec as gridspec;
from Customized_Layers import *;

class ACGAN_with_DCGAN:
    def __init__(self, noise_Size = 100, image_Size = (64, 64), learning_Rate = 0.0002):
        self.tf_Session = tf.Session();
        self.learning_Rate = learning_Rate;

        self.noise_Size = noise_Size;
        self.image_Size = image_Size;

        self.Tensor_Generate();
        self.tf_Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=6);
        
    def Tensor_Generate(self):        
        self.noise_Placeholder = tf.placeholder(tf.float32, shape=(None, self.noise_Size), name="noise_Placeholder");
        self.label_for_Fake_Placeholder = tf.placeholder(tf.float32, shape=(None, 10), name="label_for_Fake_Placeholder");
        self.label_for_Original_Placeholder = tf.placeholder(tf.float32, shape=(None, 10), name="label_for_Original_Placeholder");
        self.original_Placeholder = tf.placeholder(tf.float32, shape=(None, 28, 28, 1), name="original_Placeholder");
        self.is_Training_Placeholder = tf.placeholder(tf.bool, name="is_Training_Placeholder");
        batch_Size = tf.shape(self.noise_Placeholder)[0];

        generator_Activation = Generator_with_DCGAN(
            noise_Pattern = self.noise_Placeholder, 
            label_Pattern = self.label_for_Fake_Placeholder,
            is_Training = self.is_Training_Placeholder, 
            reuse = False
            )

        discrimination_Logit_for_Fake, label_Logit_for_Fake, discrimination_Activation_for_Fake, label_Activation_for_Fake = \
            Discriminator_with_DCGAN(                
                image_Pattern = generator_Activation,
                label_Unit_Size = 10,
                is_Training  = self.is_Training_Placeholder,
                reuse = False
                )
        
        discrimination_Logit_for_Original, label_Logit_for_Original, discrimination_Activation_for_Original, label_Activation_for_Original = \
            Discriminator_with_DCGAN(
                image_Pattern = (tf.image.resize_images(self.original_Placeholder, self.image_Size) - 0.5) * 2,
                label_Unit_Size = 10,
                is_Training  = self.is_Training_Placeholder,
                reuse = True
                )
        
        discrimination_Target_Pattern_for_Fake = tf.zeros(shape=(batch_Size, 1));
        discrimination_Target_Pattern_for_Original = tf.ones(shape=(batch_Size, 1));
        
        generator_Loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = discrimination_Target_Pattern_for_Original,   #Generator want that the result of discriminator is 1(Original).
                logits = discrimination_Logit_for_Fake
                )
            )

        label_Loss_for_Fake = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = self.label_for_Fake_Placeholder,
                logits = label_Logit_for_Fake
                )
            )

        label_Loss_for_Original = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels = self.label_for_Original_Placeholder,
                logits = label_Logit_for_Original
                )
            )

        discriminator_Loss_for_Fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = discrimination_Target_Pattern_for_Fake,
                logits = discrimination_Logit_for_Fake
                )
            )
        discriminator_Loss_for_Original = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = discrimination_Target_Pattern_for_Original,
                logits = discrimination_Logit_for_Original
                )
            )

        generator_Loss += label_Loss_for_Fake;
        discriminator_Loss = discriminator_Loss_for_Fake + discriminator_Loss_for_Original + label_Loss_for_Fake + label_Loss_for_Original;

        global_Step = tf.Variable(0, name='global_Step', trainable=False);

        trainables_Variable_List = tf.trainable_variables()
        generator_Variable_List = [x for x in trainables_Variable_List if x.name.startswith('generator')]
        discriminator_Variable_List = [x for x in trainables_Variable_List if x.name.startswith('discriminator')]
        
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            generator_Optimize = tf.train.AdamOptimizer(self.learning_Rate, beta1=0.5, beta2=0.999).minimize(
                generator_Loss, 
                var_list= generator_Variable_List,
                global_step = global_Step
                )
        
            discriminator_Optimize = tf.train.AdamOptimizer(self.learning_Rate, beta1=0.5, beta2=0.999).minimize(
                discriminator_Loss, 
                var_list= discriminator_Variable_List
                )
        
        self.training_Tensor_List_for_Generator = [global_Step, generator_Loss, generator_Optimize];
        self.training_Tensor_List_for_Disciriminator = [discriminator_Loss, discriminator_Optimize];        
        self.test_Tensor_List = [global_Step, tf.squeeze(generator_Activation, axis = [3])];

        summary_Dir = "Summary";
        if not os.path.exists(summary_Dir):
            os.makedirs(summary_Dir);
        graph_Writer = tf.summary.FileWriter(summary_Dir, self.tf_Session.graph);
        graph_Writer.close();

        self.tf_Session.run(tf.global_variables_initializer());

    def Train(self, batch_Size = 128, test_Step = 1000, checkpoint_Step = 1000):
        checkpoint_Dir = "Checkpoint";
        if not os.path.exists(checkpoint_Dir):
            os.makedirs(checkpoint_Dir);
        checkpoint_Path = checkpoint_Dir + '/GAN.ckpt';

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True, reshape = False);

        while True:            
            start_Time = time.time();
                
            original_Pattern, original_Label_Pattern = mnist.train.next_batch(batch_Size);

            fake_Label_List = np.random.randint(0, 10, batch_Size);
            fake_Label_Pattern = np.zeros((batch_Size, 10));
            fake_Label_Pattern[np.arange(batch_Size), fake_Label_List] = 1;
            
            feed_Dict = {
                self.noise_Placeholder: np.random.uniform(-1., 1., size=[batch_Size, self.noise_Size]),
                self.label_for_Fake_Placeholder: fake_Label_Pattern,
                self.original_Placeholder: original_Pattern,     # 'generator_Variable_List' requires the value of original_Placeholder.
                self.is_Training_Placeholder: True
                }
            global_Step, generator_Loss, _, = self.tf_Session.run(self.training_Tensor_List_for_Generator, feed_dict = feed_Dict);
                            
            feed_Dict = {
                self.noise_Placeholder: np.random.uniform(-1., 1., size=[batch_Size, self.noise_Size]),
                self.label_for_Fake_Placeholder: fake_Label_Pattern,
                self.label_for_Original_Placeholder: original_Label_Pattern,
                self.original_Placeholder: original_Pattern,
                self.is_Training_Placeholder: True
                }
            discriminator_Loss, _, = self.tf_Session.run(self.training_Tensor_List_for_Disciriminator, feed_dict = feed_Dict);

            print(
                "Spent_Time:", np.round(time.time() - start_Time, 3), "\t",
                "Global_Step:", global_Step, "\t",                
                "Generator_Loss:", generator_Loss, "\t",
                "Discriminator_Loss:", discriminator_Loss, "\t",
            )
            
            if (global_Step % checkpoint_Step) == 0:
                self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step=global_Step);

            if (global_Step % test_Step) == 0:
                self.Test();
        
    def Test(self):
        test_Dir = "Result";        
        if not os.path.exists(test_Dir):
            os.makedirs(test_Dir);

        test_Label_List = [0,1,2,3,4,5,6,7,8,9];
        test_Label_Pattern = np.zeros((len(test_Label_List), 10));
        test_Label_Pattern[np.arange(len(test_Label_List)), test_Label_List] = 1;            
        feed_Dict = {
            self.noise_Placeholder: np.random.uniform(-1., 1., size=[len(test_Label_List), self.noise_Size]),
            self.label_for_Fake_Placeholder: test_Label_Pattern,
            self.is_Training_Placeholder: False
            };   #Batch is constant in the test.
        global_Step, mnist_List = self.tf_Session.run(self.test_Tensor_List, feed_dict = feed_Dict);

        fig = plt.figure(figsize=(5, 2))
        gs = gridspec.GridSpec(2, 5)
        gs.update(wspace=0.05, hspace=0.05)

        for index, mnist in enumerate(mnist_List):
            ax = plt.subplot(gs[index])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(mnist, cmap='Greys_r')

        plt.savefig('%s/S%d.png' % (test_Dir, global_Step), bbox_inches='tight');
        plt.close();

if __name__ == "__main__":
    new_ACGAN = ACGAN_with_DCGAN(noise_Size = 100, image_Size = (64, 64), learning_Rate = 0.0002);
    new_ACGAN.Train(test_Step = 100, checkpoint_Step = 1000);

