import numpy as np;
import tensorflow as tf;
import time, os;
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt;
import matplotlib.gridspec as gridspec;
from Customized_Layers import *;
from Pattern_Feeder import Pattern_Feeder;

class DiscoGAN:
    def __init__(self, image_Size = 64, learning_Rate = 0.0002, batch_Size = 128):
        self.tf_Session = tf.Session();

        self.new_Pattern_Feeder = Pattern_Feeder(
            attribute_File_Path = "C:/CelebA/list_attr_celeba.txt", 
            image_Files_Dir = "C:/CelebA/Img",
            image_Size = image_Size,
            batch_Size = batch_Size
            )

        self.image_Size = image_Size;
        self.learning_Rate = learning_Rate;

        self.Tensor_Generate();
        self.tf_Saver = tf.train.Saver(max_to_keep=5, keep_checkpoint_every_n_hours=6);

    def Tensor_Generate(self):
        self.image_A_Placeholder = tf.placeholder(tf.float32, shape=(None, self.image_Size, self.image_Size, 3), name = "image_A_Placeholder");
        self.image_B_Placeholder = tf.placeholder(tf.float32, shape=(None, self.image_Size, self.image_Size, 3), name = "image_B_Placeholder");
        self.is_Training_Placeholder = tf.placeholder(tf.bool, name="is_Training_Placeholder");

        generator_AB_Logit, generator_AB_Activation = Generator(self.image_A_Placeholder, is_Training = self.is_Training_Placeholder, name = "generator_AB");
        generator_BA_Logit, generator_BA_Activation = Generator(self.image_B_Placeholder, is_Training = self.is_Training_Placeholder, name = "generator_BA");
        
        generator_ABA_Logit, generator_ABA_Activation = Generator(generator_AB_Activation, is_Training = self.is_Training_Placeholder, name = "generator_BA", reuse = True);
        generator_BAB_Logit, generator_BAB_Activation = Generator(generator_BA_Activation, is_Training = self.is_Training_Placeholder, name = "generator_AB", reuse = True);

        discriminator_A_for_Original_Logit = Discriminator(self.image_A_Placeholder, is_Training = self.is_Training_Placeholder, name = "discriminator_A");
        discriminator_A_for_Fake_Logit = Discriminator(generator_BA_Activation, is_Training = self.is_Training_Placeholder, name = "discriminator_A", reuse = True);
        
        discriminator_B_for_Original_Logit = Discriminator(self.image_B_Placeholder, is_Training = self.is_Training_Placeholder, name = "discriminator_B");
        discriminator_B_for_Fake_Logit = Discriminator(generator_AB_Activation, is_Training = self.is_Training_Placeholder, name = "discriminator_B", reuse = True);
        
        loss_Dict = {};
        loss_Dict["Generator", "AB", "Reconstruction"] = tf.reduce_mean(
            tf.losses.mean_squared_error(
                labels = self.image_A_Placeholder, 
                predictions = generator_ABA_Activation
                )
            )
        loss_Dict["Generator", "BA", "Reconstruction"] = tf.reduce_mean(
            tf.losses.mean_squared_error(
                labels = self.image_B_Placeholder, 
                predictions = generator_BAB_Activation
                )
            )

        loss_Dict["Generator", "AB", "Construction"] = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = discriminator_B_for_Fake_Logit,
               labels = tf.ones_like(discriminator_B_for_Fake_Logit)
               )
           )
        loss_Dict["Generator", "BA", "Construction"] = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = discriminator_A_for_Fake_Logit,
               labels = tf.ones_like(discriminator_A_for_Fake_Logit)
               )
           )

        loss_Dict["Discriminator", "A", "Original"] = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = discriminator_A_for_Original_Logit,
               labels = tf.ones_like(discriminator_A_for_Original_Logit)
               )
           )
        loss_Dict["Discriminator", "A", "Fake"] = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = discriminator_A_for_Fake_Logit,
               labels = tf.zeros_like(discriminator_A_for_Fake_Logit)
               )
           )
        
        loss_Dict["Discriminator", "B", "Original"] = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = discriminator_B_for_Original_Logit,
               labels = tf.ones_like(discriminator_B_for_Original_Logit)
               )
           )
        loss_Dict["Discriminator", "B", "Fake"] = tf.reduce_mean(
           tf.nn.sigmoid_cross_entropy_with_logits(
               logits = discriminator_B_for_Fake_Logit,
               labels = tf.zeros_like(discriminator_B_for_Fake_Logit)
               )
           )

        generator_Loss = 0;
        discriminator_Loss = 0;
        for key_Tuple in loss_Dict.keys():
            if key_Tuple[0] == "Generator":
                generator_Loss += loss_Dict[key_Tuple];
            elif key_Tuple[0] == "Discriminator":
                discriminator_Loss += loss_Dict[key_Tuple];
        
        generator_Variable_List = [x for x in tf.trainable_variables() if x.name.startswith('generator')];
        discriminator_Variable_List = [x for x in tf.trainable_variables() if x.name.startswith('discriminator')];

        global_Step = tf.Variable(0, name='global_Step', trainable=False);

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
        self.test_Tensor_List = [global_Step, self.image_A_Placeholder, generator_AB_Activation, generator_ABA_Activation, self.image_B_Placeholder, generator_BA_Activation, generator_BAB_Activation];

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
        checkpoint_Path = checkpoint_Dir + '/DiscoGAN.ckpt';

        while True:            
            start_Time = time.time();
                
            feed_Dict = {
                self.image_A_Placeholder : self.new_Pattern_Feeder.Get_Domain_A_Pattern(),
                self.image_B_Placeholder : self.new_Pattern_Feeder.Get_Domain_B_Pattern(),
                self.is_Training_Placeholder: True
                }

            global_Step, generator_Loss, _, = self.tf_Session.run(self.training_Tensor_List_for_Generator, feed_dict = feed_Dict);                           
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
        if not os.path.exists(test_Dir + "/A"):
            os.makedirs(test_Dir + "/A");
        if not os.path.exists(test_Dir + "/B"):
            os.makedirs(test_Dir + "/B");
            
        index_List = [0,1,2,3]
        domain_A_Pattern, domain_B_Pattern = self.new_Pattern_Feeder.Get_Test_Pattern(index_List = index_List);
        feed_Dict = {
            self.image_A_Placeholder : domain_A_Pattern,
            self.image_B_Placeholder : domain_B_Pattern,
            self.is_Training_Placeholder: False
            }
        global_Step, pattern_A, pattern_AB, pattern_ABA, pattern_B, pattern_BA, pattern_BAB  = self.tf_Session.run(self.test_Tensor_List, feed_dict = feed_Dict);
        
        self.Result_Plot(pattern_A, pattern_AB, pattern_ABA, '%s/%s/S%d.png' % (test_Dir, "A", global_Step))
        self.Result_Plot(pattern_B, pattern_BA, pattern_BAB, '%s/%s/S%d.png' % (test_Dir, "B", global_Step))
        
    def Result_Plot(self, original_Pattern, generated_Pattern, restructured_Pattern, file_Name):
        fig = plt.figure(figsize=(3, len(original_Pattern)))
        gs = gridspec.GridSpec(len(original_Pattern), 3)
        gs.update(wspace=0.05, hspace=0.05)

        for index in range(0, len(original_Pattern)):
            for image_Index, image in enumerate([original_Pattern[index], generated_Pattern[index], restructured_Pattern[index]]):
                ax = plt.subplot(gs[index * 3 + image_Index])
                plt.axis('off')
                ax.set_xticklabels([])
                ax.set_yticklabels([])
                ax.set_aspect('equal')
                #plt.imshow(self.new_Pattern_Feeder.Pattern_to_Image(image), interpolation='nearest');
                plt.imshow(self.new_Pattern_Feeder.Pattern_to_Image(image));

        plt.savefig(file_Name, bbox_inches='tight');
        plt.close();


if __name__ == "__main__":
    new_DiscoGAN = DiscoGAN(image_Size=128, batch_Size = 64);
    new_DiscoGAN.Restore();
    new_DiscoGAN.Train(test_Step = 1000, checkpoint_Step = 1000);