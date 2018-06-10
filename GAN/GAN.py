#Referred : https://github.com/wiseodd/generative-models/blob/master/GAN/vanilla_gan/gan_tensorflow.py

import numpy as np;
import tensorflow as tf;
from tensorflow.examples.tutorials.mnist import input_data;
import time, os;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
import matplotlib.gridspec as gridspec;

class GAN:
    def __init__(self, noise_Size = 100, hidden_Size = 128, learning_Rate = 0.002):
        self.tf_Session = tf.Session();

        self.noise_Size = noise_Size;
        self.hidden_Size = hidden_Size;
        self.learning_Rate = learning_Rate;

        self.Weight_Generate();
        self.Tensor_Generate();
        self.tf_Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=6);

    def Weight_Generate(self, initializer_Std = 0.1):
        self.weightMatrix_Dict = {};
        self.biasMatrix_Dict = {};
        
        self.weightMatrix_Dict["Generator","IH"] = tf.get_variable(
            name = "Weight_Generator_IH",
            shape = (self.noise_Size, self.hidden_Size),
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=initializer_Std)
        )
        self.weightMatrix_Dict["Generator","HO"] = tf.get_variable(
            name = "Weight_Generator_HO",
            shape = (self.hidden_Size, 28*28),
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=initializer_Std)
        )
        self.weightMatrix_Dict["Discriminator","IH"] = tf.get_variable(
            name = "Weight_Discriminator_IH",
            shape = (28*28, self.hidden_Size),
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=initializer_Std)
        )
        self.weightMatrix_Dict["Discriminator","HO"] = tf.get_variable(
            name = "Weight_Discriminator_HO",
            shape = (self.hidden_Size, 1),
            dtype = tf.float32,
            initializer = tf.truncated_normal_initializer(stddev=initializer_Std)
        )
        
        self.biasMatrix_Dict["Generator","H"] = tf.get_variable(
            name = "Bias_Generator_IH",
            shape = (1, self.hidden_Size),
            dtype = tf.float32,
            initializer = tf.zeros_initializer()
        )
        self.biasMatrix_Dict["Generator","O"] = tf.get_variable(
            name = "Bias_Generator_HO",
            shape = (1, 28*28),
            dtype = tf.float32,
            initializer = tf.zeros_initializer()
        )
        self.biasMatrix_Dict["Discriminator","H"] = tf.get_variable(
            name = "Bias_Discriminator_IH",
            shape = (1, self.hidden_Size),
            dtype = tf.float32,
            initializer = tf.zeros_initializer()
        )
        self.biasMatrix_Dict["Discriminator","O"] = tf.get_variable(
            name = "Bias_Discriminator_HO",
            shape = (1, 1),
            dtype = tf.float32,
            initializer = tf.zeros_initializer()
        )

    def Tensor_Generate(self):        
        self.noise_Placeholder = tf.placeholder(tf.float32, shape=(None, self.noise_Size), name="noise_Placeholder");
        self.original_Placeholder = tf.placeholder(tf.float32, shape=(None, 28*28), name="original_Placeholder");
        batch_Size = tf.shape(self.noise_Placeholder)[0];

        generator_H_Activation = tf.nn.relu(tf.matmul(self.noise_Placeholder, self.weightMatrix_Dict["Generator","IH"]) + self.biasMatrix_Dict["Generator","H"]);
        generator_O_Activation = tf.nn.sigmoid(tf.matmul(generator_H_Activation, self.weightMatrix_Dict["Generator","HO"]) + self.biasMatrix_Dict["Generator","O"]);

        discriminator_H_Activation_for_Fake = tf.nn.relu(tf.matmul(generator_O_Activation, self.weightMatrix_Dict["Discriminator","IH"]) + self.biasMatrix_Dict["Discriminator","H"]);
        discriminator_O_Logit_for_Fake = tf.matmul(discriminator_H_Activation_for_Fake, self.weightMatrix_Dict["Discriminator","HO"]) + self.biasMatrix_Dict["Discriminator","O"];

        discriminator_H_Activation_for_Original = tf.nn.relu(tf.matmul(self.original_Placeholder, self.weightMatrix_Dict["Discriminator","IH"]) + self.biasMatrix_Dict["Discriminator","H"]);
        discriminator_O_Logit_for_Original = tf.matmul(discriminator_H_Activation_for_Original, self.weightMatrix_Dict["Discriminator","HO"]) + self.biasMatrix_Dict["Discriminator","O"];

        fake_Target_Pattern = tf.zeros(shape=(batch_Size, 1));
        original_Target_Pattern = tf.ones(shape=(batch_Size, 1));
        
        generator_Loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = original_Target_Pattern,   #Generator want that the result of discriminator is 1(Original).
                logits = discriminator_O_Logit_for_Fake
                )
            )

        discriminator_Loss_for_Fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = fake_Target_Pattern,
                logits = discriminator_O_Logit_for_Fake
                )
            )
        discriminator_Loss_for_Original = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels = original_Target_Pattern,
                logits = discriminator_O_Logit_for_Original
                )
            )
        discriminator_Loss = discriminator_Loss_for_Fake + discriminator_Loss_for_Original;

        global_Step = tf.Variable(0, name='generator_Global_Step', trainable=False);

        generator_Optimize = tf.train.AdamOptimizer(self.learning_Rate).minimize(
            generator_Loss, 
            var_list=[
                self.weightMatrix_Dict["Generator","IH"], 
                self.weightMatrix_Dict["Generator","HO"], 
                self.biasMatrix_Dict["Generator","H"], 
                self.biasMatrix_Dict["Generator","O"]
                ],
            global_step = global_Step
            )
        
        discriminator_Optimize = tf.train.AdamOptimizer(self.learning_Rate).minimize(
            discriminator_Loss, 
            var_list=[
                self.weightMatrix_Dict["Discriminator","IH"], 
                self.weightMatrix_Dict["Discriminator","HO"], 
                self.biasMatrix_Dict["Discriminator","H"], 
                self.biasMatrix_Dict["Discriminator","O"]
                ]
            )
        
        self.training_Tensor_List_for_Generator = [global_Step, generator_Loss, generator_Optimize];
        self.training_Tensor_List_for_Disciriminator = [discriminator_Loss, discriminator_Optimize];        
        self.test_Tensor_List = [global_Step, tf.reshape(generator_O_Activation, shape=(-1, 28, 28))];

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

        mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

        while True:
            start_Time = time.time();
            
            feed_Dict = {
                self.noise_Placeholder: np.random.uniform(-1., 1., size=[batch_Size, self.noise_Size]),
                }
            global_Step, generator_Loss, _, = self.tf_Session.run(self.training_Tensor_List_for_Generator, feed_dict = feed_Dict);
            
            feed_Dict = {
                self.noise_Placeholder: np.random.uniform(-1., 1., size=[batch_Size, self.noise_Size]),
                self.original_Placeholder: mnist.train.next_batch(batch_Size)[0],   #Vanila GAN does not use target label
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

        feed_Dict = {self.noise_Placeholder: np.random.uniform(-1., 1., size=[16, self.noise_Size])};   #Batch is constant in the test.
        global_Step, mnist_List = self.tf_Session.run(self.test_Tensor_List, feed_dict = feed_Dict);
        #self.Export_Plot(test_Result, test_Dir);          

        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(4, 4)
        gs.update(wspace=0.05, hspace=0.05)

        for index, mnist in enumerate(mnist_List):
            ax = plt.subplot(gs[index])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            plt.imshow(mnist.reshape(28, 28), cmap='Greys_r')

        plt.savefig('%s/S%d.png' % (test_Dir, global_Step), bbox_inches='tight');
        plt.close();

if __name__ == "__main__":
    new_GAN = GAN();
    new_GAN.Train();

