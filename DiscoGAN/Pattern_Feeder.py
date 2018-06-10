import numpy as np;
import random, time;
from imageio import imread;
from skimage.transform import resize;
import _thread as thread;
from collections import deque;

class Pattern_Feeder:
    def __init__(
        self,
        attribute_File_Path = "list_attr_celeba.txt",
        image_Files_Dir = "img",
        domain_A_Attribute = (9, 1),   #(attribute_Index, assign_Value)
        domain_B_Attribute = (10, 1),   #(attribute_Index, assign_Value)
        static_Attribute_List = [(21, -1)],  #List of (attribute_Index, assign_Value)
        file_Name_Index = 0,        
        image_Size = 64,
        batch_Size = 128,
        max_Queue = 100
        ):
        self.attribute_File_Path = attribute_File_Path;
        self.image_Files_Dir = image_Files_Dir;
        self.domain_A_Attribute = domain_A_Attribute;
        self.domain_B_Attribute = domain_B_Attribute;
        self.static_Attribute_List = static_Attribute_List;
        self.file_Name_Index = file_Name_Index;        
        self.image_Size = image_Size;
        self.batch_Size = batch_Size;
        self.max_Queue = max_Queue;

        self.Load_File_Name_List();
        self.pattern_Domain_A_Queue = deque();
        self.pattern_Domain_B_Queue = deque();
        
        thread.start_new_thread(self.Pattern_Domain_A_Generate, ());
        thread.start_new_thread(self.Pattern_Domain_B_Generate, ());

    def Load_File_Name_List(self):	
        with open(self.attribute_File_Path, "r") as f:
            readLines = f.readlines()[1:];
            raw_Data_List = [readLine.strip().split("\t") for readLine in readLines ];

        for attribute_Index, assign_Value in self.static_Attribute_List:
            raw_Data_List = [raw_Data for raw_Data in raw_Data_List if raw_Data[attribute_Index] == str(assign_Value)];
        
        self.domain_A_File_List = [raw_Data[0] for raw_Data in raw_Data_List if raw_Data[self.domain_A_Attribute[0]] == str(self.domain_A_Attribute[1])];
        self.domain_B_File_List = [raw_Data[0] for raw_Data in raw_Data_List if raw_Data[self.domain_B_Attribute[0]] == str(self.domain_B_Attribute[1])];

    def Pattern_Domain_A_Generate(self):        
        while True:
            if len(self.pattern_Domain_A_Queue) >= self.max_Queue:
                time.sleep(0.1)
                continue;
            selected_File_Name_List = random.sample(self.domain_A_File_List, self.batch_Size);

            #shape: (batch_Size, image_Size, image_Size, 3), value_range: -1.0 to 1.0            
            new_Pattern = (np.stack([resize(imread(self.image_Files_Dir + "/" + selected_File_Name), (self.image_Size, self.image_Size)) for selected_File_Name in selected_File_Name_List]) - 0.5) * 2;
            self.pattern_Domain_A_Queue.append(new_Pattern.astype("float32"));
            
    def Pattern_Domain_B_Generate(self):
        while True:
            if len(self.pattern_Domain_B_Queue) >= self.max_Queue:
                time.sleep(0.1)
                continue;
            selected_File_Name_List = random.sample(self.domain_B_File_List, self.batch_Size);

            #shape: (batch_Size, image_Size, image_Size, 3), value_range: -1.0 to 1.0
            new_Pattern = (np.stack([resize(imread(self.image_Files_Dir + "/" + selected_File_Name), (self.image_Size, self.image_Size)) for selected_File_Name in selected_File_Name_List]) - 0.5) * 2;
            self.pattern_Domain_B_Queue.append(new_Pattern);

    def Get_Domain_A_Pattern(self):
        while len(self.pattern_Domain_A_Queue) == 0:
            time.sleep(0.1);
        return self.pattern_Domain_A_Queue.pop();    
    
    def Get_Domain_B_Pattern(self):
        while len(self.pattern_Domain_B_Queue) == 0:
            time.sleep(0.1);
        return self.pattern_Domain_B_Queue.pop();

    def Get_Test_Pattern(self, index_List = None, batch_Size = None):
        if batch_Size is None and index_List is None:
            return False;
        elif index_List is not None:
            selected_Domain_A_File_Name_List = [self.domain_A_File_List[index] for index in index_List];
            selected_Domain_B_File_Name_List = [self.domain_B_File_List[index] for index in index_List];
        else:
            selected_Domain_A_File_Name_List = random.sample(self.domain_A_File_List, self.batch_Size);
            selected_Domain_B_File_Name_List = random.sample(self.domain_B_File_List, self.batch_Size);

        new_Domain_A_Pattern = (np.stack([resize(imread(self.image_Files_Dir + "/" + selected_File_Name), (self.image_Size, self.image_Size)) for selected_File_Name in selected_Domain_A_File_Name_List]) - 0.5) * 2;
        new_Domain_B_Pattern = (np.stack([resize(imread(self.image_Files_Dir + "/" + selected_File_Name), (self.image_Size, self.image_Size)) for selected_File_Name in selected_Domain_B_File_Name_List]) - 0.5) * 2;

        return new_Domain_A_Pattern, new_Domain_B_Pattern;

    def Pattern_to_Image(self, pattern):
        return pattern / 2 + 0.5