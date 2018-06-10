import numpy as np;
import random, time;
from imageio import imread;
from skimage.transform import resize;
import _thread as thread;
from collections import deque;

class Pattern_Feeder:
    def __init__(
        self,
        placeholder_List,   #[noise_Placeholder, image_Placeholder, is_Training_Placeholder]
        attribute_File_Path = "list_attr_celeba.txt",
        image_Files_Dir = "img",
        attribute_List = [],  #List of (attribute_Index, assign_Value)        
        noise_Size = 100,
        image_Size = 64,
        batch_Size = 128,
        max_Queue = 100
        ):
        self.placeholder_List = placeholder_List;
        self.attribute_File_Path = attribute_File_Path;
        self.image_Files_Dir = image_Files_Dir;
        self.attribute_List = attribute_List;   
        
        self.noise_Size = noise_Size;
        self.image_Size = image_Size;
        self.batch_Size = batch_Size;
        self.max_Queue = max_Queue;

        self.Load_File_Name_List();
        self.pattern_Queue = deque();

        thread.start_new_thread(self.Pattern_Generate, ());

    def Load_File_Name_List(self):	
        with open(self.attribute_File_Path, "r") as f:
            readLines = f.readlines()[1:];
            raw_Data_List = [readLine.strip().split("\t") for readLine in readLines ];

        for attribute_Index, assign_Value in self.attribute_List:
            raw_Data_List = [raw_Data for raw_Data in raw_Data_List if raw_Data[attribute_Index] == str(assign_Value)];
        
        self.domain_File_List = [raw_Data[0] for raw_Data in raw_Data_List];
                       
    def Pattern_Generate(self):
        while True:
            if len(self.pattern_Queue) >= self.max_Queue:
                time.sleep(0.1)
                continue;
            
            new_Feed_Dict = {};
            new_Feed_Dict[self.placeholder_List[0]] = np.random.rand(self.batch_Size, self.noise_Size);            
            selected_File_Name_List = random.sample(self.domain_File_List, self.batch_Size);
            #shape: (batch_Size, image_Size, image_Size, 3), value_range: -1.0 to 1.0
            new_Feed_Dict[self.placeholder_List[1]] = (np.stack([resize(imread(self.image_Files_Dir + "/" + selected_File_Name), (self.image_Size, self.image_Size)) for selected_File_Name in selected_File_Name_List]) - 0.5) * 2;
            new_Feed_Dict[self.placeholder_List[2]] = True;

            self.pattern_Queue.append(new_Feed_Dict);

    def Get_Pattern(self):
        while len(self.pattern_Queue) == 0:
            time.sleep(0.1);
        return self.pattern_Queue.pop();

    def Get_Test_Pattern(self, batch_Size = 16):
        new_Feed_Dict = {};
        new_Feed_Dict[self.placeholder_List[0]] = np.random.rand(batch_Size, self.noise_Size); 
        new_Feed_Dict[self.placeholder_List[2]] = False; 
        
        return new_Feed_Dict; 

    def Pattern_to_Image(self, pattern):
        return pattern / 2 + 0.5