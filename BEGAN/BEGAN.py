import numpy as np;
import tensorflow as tf;
import time, os;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
import matplotlib.gridspec as gridspec;
from Customized_Layers import *;
from Pattern_Feeder import Pattern_Feeder;

class BEGAN:
    def __init__(self, noise_Size = 100, image_Size = 64, filter_Count = 128, gamma = 0.5, lambda_k = 0.001, learning_Rate = 0.00001, batch_Size = 16):
        self.tf_Session = tf.Session();

        self.noise_Size = noise_Size;
        self.image_Size = image_Size;
        self.filter_Count = filter_Count;
        self.gamma = gamma;
        self.lambda_k = lambda_k;
        self.learning_Rate = learning_Rate;
        
        self.Placeholder_Generate();

        self.pattern_Feeder = Pattern_Feeder(
            placeholder_List = [self.noise_Placeholder, self.image_Placeholder, self.is_Training_Placeholder],
            attribute_File_Path = "C:/CelebA/list_attr_celeba.txt", 
            image_Files_Dir = "C:/CelebA/Img",
            noise_Size = noise_Size,
            image_Size = image_Size,
            batch_Size = batch_Size
            )        

        self.Tensor_Generate();
        self.tf_Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=6);

    def Placeholder_Generate(self):
        self.noise_Placeholder = tf.placeholder(tf.float32, shape=(None, self.noise_Size), name = "noise_Placeholder");
        self.image_Placeholder = tf.placeholder(tf.float32, shape=(None, self.image_Size, self.image_Size, 3), name = "image_Placeholder");
        self.is_Training_Placeholder = tf.placeholder(tf.bool, name="is_Training_Placeholder");

    def Tensor_Generate(self):
        k = tf.Variable(0.0, trainable=False, name='k')

        generator_Activation, generator_Variable_List = Generator(self.noise_Placeholder, image_Size = self.image_Size);
        
        discriminator_Original_Activation, discriminator_Original_Variable_List = Discriminator(self.image_Placeholder);
        discriminator_Fake_Activation, discriminator_Fake_Variable_List = Discriminator(generator_Activation, reuse = True);
        discriminator_Variable_List = discriminator_Original_Variable_List + discriminator_Fake_Variable_List;

        loss_Generator = tf.reduce_mean(tf.abs(generator_Activation - discriminator_Fake_Activation));
        
        loss_Discriminator_Original = tf.reduce_mean(tf.abs(self.image_Placeholder - discriminator_Original_Activation));

        loss_Discriminator_Fake = tf.reduce_mean(tf.abs(generator_Activation - discriminator_Fake_Activation));
        loss_Discriminator = loss_Discriminator_Original - k * loss_Discriminator_Fake;

        balance = self.gamma * loss_Discriminator_Original - loss_Generator;
        convergence = loss_Discriminator_Original + tf.abs(balance);

        global_Step = tf.Variable(0, name='global_Step', trainable=False);

        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            generator_Optimize = tf.train.AdamOptimizer(self.learning_Rate, beta1=0.5, beta2=0.999).minimize(
                loss_Generator, 
                var_list= generator_Variable_List,
                global_step = global_Step
                )
        
            discriminator_Optimize = tf.train.AdamOptimizer(self.learning_Rate, beta1=0.5, beta2=0.999).minimize(
                loss_Discriminator, 
                var_list= discriminator_Variable_List
                )

            k_Assign = tf.assign(k, tf.clip_by_value(k + self.lambda_k * balance, 0, 1));

        self.training_Tensor_List = [global_Step, loss_Generator, loss_Discriminator, convergence, generator_Optimize, discriminator_Optimize, k_Assign];
        self.test_Tensor_List = [global_Step, generator_Activation];

        summary_Dir = "Summary";
        if not os.path.exists(summary_Dir):
            os.makedirs(summary_Dir);
        graph_Writer = tf.summary.FileWriter(summary_Dir, self.tf_Session.graph);
        graph_Writer.close();

        self.tf_Session.run(tf.global_variables_initializer());

    def Restore(self):
        checkpoint_Dir = "Checkpoint";
        checkpoint_Path = tf.train.latest_checkpoint(checkpoint_Dir);

        if checkpoint_Path is None:
            print("There is no checkpoint");
        else:
            self.tf_Saver.restore(self.tf_Session, checkpoint_Path);
            print("Checkpoint '", checkpoint_Path, "' is loaded.");

    def Train(self, test_Step = 1000, checkpoint_Step = 1000):
        checkpoint_Dir = "Checkpoint";
        if not os.path.exists(checkpoint_Dir):
            os.makedirs(checkpoint_Dir);
        checkpoint_Path = checkpoint_Dir + '/BEGAN.ckpt';

        self.Test();

        while True:            
            start_Time = time.time();

            global_Step, loss_Generator, loss_Discriminator, convergence, _, _, k = self.tf_Session.run(self.training_Tensor_List, feed_dict = self.pattern_Feeder.Get_Pattern());

            print(
                "Spent_Time:", np.round(time.time() - start_Time, 3), "\t",
                "Global_Step:", global_Step, "\t",                
                "Generator_Loss:", loss_Generator, "\t",
                "Discriminator_Loss:", loss_Discriminator, "\t",
                "Convergence:", convergence, "\t",
                "k:", k, "\t",
            )
            
            if (global_Step % checkpoint_Step) == 0:
                self.tf_Saver.save(self.tf_Session, checkpoint_Path, global_step=global_Step);

            if (global_Step % test_Step) == 0:
                self.Test();

    def Test(self, batch_Size = 16):
        test_Dir = "Result";        
        if not os.path.exists(test_Dir):
            os.makedirs(test_Dir);
           
        global_Step, result_Image_Array  = self.tf_Session.run(self.test_Tensor_List, feed_dict = self.pattern_Feeder.Get_Test_Pattern(batch_Size));
        
        self.Result_Plot(global_Step, result_Image_Array, '%s/S%d.png' % (test_Dir, global_Step))
        
    def Result_Plot(self, global_Step, result_Image_Array, file_Name):
        axis_Size = int(np.ceil(np.sqrt(np.shape(result_Image_Array)[0])));
        fig = plt.figure(figsize=(axis_Size, axis_Size))
        gs = gridspec.GridSpec(axis_Size, axis_Size)
        gs.update(wspace=0.05, hspace=0.05)

        for index, image in enumerate(result_Image_Array):
            ax = plt.subplot(gs[index])
            plt.axis('off')
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_aspect('equal')
            #plt.imshow(self.new_Pattern_Feeder.Pattern_to_Image(image), interpolation='nearest');
            plt.imshow(self.pattern_Feeder.Pattern_to_Image(image));

        plt.savefig(file_Name, bbox_inches='tight');
        plt.close();

if __name__ == "__main__":
    new_BEGAN = BEGAN(image_Size=64, batch_Size = 16);
    new_BEGAN.Restore();
    new_BEGAN.Train(test_Step = 1000, checkpoint_Step = 1000);