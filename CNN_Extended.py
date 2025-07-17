'''
This is file replicates ORACLE's CNN implementation to create an RF fingerprinting model
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import configs

class CNNFingerprinter(nn.Module):
    def __init__(self, num_transmitters=16, input_channels=2, input_window=128):
        super(CNNFingerprinter, self).__init__()
        
        self.datatype = configs.datatype
        self.feature_map_size = input_window
        
        self.ConvolutionalLayers = nn.Sequential(
            #Block 1
            nn.Sequential(
                    #First convolution layer
                nn.Conv1d(
                    in_channels=input_channels,
                    out_channels=self.feature_map_size,
                    kernel_size=7,
                    groups=1,
                    padding=3,
                    dtype=self.datatype
                ),
                
                nn.BatchNorm1d(
                    num_features=self.feature_map_size,
                    dtype=self.datatype
                ),

                nn.ReLU(),
                
                #Second convolution layer
                nn.Conv1d(
                    in_channels=self.feature_map_size,
                    out_channels=self.feature_map_size,
                    kernel_size=5,
                    groups=1,
                    padding=2,
                    dtype=self.datatype
                ),
                
                nn.BatchNorm1d(
                    num_features=self.feature_map_size,
                    dtype=self.datatype
                ),
                
                nn.ReLU(),
                
                nn.MaxPool1d(kernel_size=2)
            ),
            #Block 2
            nn.Sequential(
                    #First convolution layer
                nn.Conv1d(
                    in_channels=self.feature_map_size,
                    out_channels=self.feature_map_size,
                    kernel_size=7,
                    groups=1,
                    padding=3,
                    dtype=self.datatype
                ),
                
                nn.BatchNorm1d(
                    num_features=self.feature_map_size,
                    dtype=self.datatype
                ),

                nn.ReLU(),
                
                #Second convolution layer
                nn.Conv1d(
                    in_channels=self.feature_map_size,
                    out_channels=self.feature_map_size,
                    kernel_size=5,
                    groups=1,
                    padding=2,
                    dtype=self.datatype
                ),
                
                nn.BatchNorm1d(
                    num_features=self.feature_map_size,
                    dtype=self.datatype
                ),
                
                nn.ReLU(),
                
                nn.MaxPool1d(kernel_size=2)
            ),
            
            #Block 3
            nn.Sequential(
                    #First convolution layer
                nn.Conv1d(
                    in_channels=self.feature_map_size,
                    out_channels=self.feature_map_size,
                    kernel_size=7,
                    groups=1,
                    padding=3,
                    dtype=self.datatype
                ),
                
                nn.BatchNorm1d(
                    num_features=self.feature_map_size,
                    dtype=self.datatype
                ),

                nn.ReLU(),
                
                #Second convolution layer
                nn.Conv1d(
                    in_channels=self.feature_map_size,
                    out_channels=self.feature_map_size,
                    kernel_size=5,
                    groups=1,
                    padding=2,
                    dtype=self.datatype
                ),
                
                nn.BatchNorm1d(
                    num_features=self.feature_map_size,
                    dtype=self.datatype
                ),
                
                nn.ReLU(),
                
                nn.MaxPool1d(kernel_size=2)
            ),
            
            #Block 4
            nn.Sequential(
                    #First convolution layer
                nn.Conv1d(
                    in_channels=self.feature_map_size,
                    out_channels=self.feature_map_size,
                    kernel_size=7,
                    groups=1,
                    padding=3,
                    dtype=self.datatype
                ),
                
                nn.BatchNorm1d(
                    num_features=self.feature_map_size,
                    dtype=self.datatype
                ),

                nn.ReLU(),
                
                #Second convolution layer
                nn.Conv1d(
                    in_channels=self.feature_map_size,
                    out_channels=self.feature_map_size,
                    kernel_size=5,
                    groups=1,
                    padding=2,
                    dtype=self.datatype
                ),
                
                nn.BatchNorm1d(
                    num_features=self.feature_map_size,
                    dtype=self.datatype
                ),
                
                nn.ReLU(),
                
                nn.MaxPool1d(kernel_size=2)
            )
        )
        
        #Each max pooling layer divides the feature map size by 2, and since there are 4 blocks that contain pooling then the final, flattened input to the linear layers is
        # input_window * (input_window / 2^4 )
        self.fc_input_size = input_window * (input_window // (2 ** 4))
        
        self.classifer = nn.Sequential(
            #First linear lyaer
            nn.Linear(
                in_features=self.fc_input_size,
                out_features=256,
                dtype=self.datatype
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5),
        
            #Second linear lyaer
            nn.Linear(
                in_features=256,
                out_features=128,
                dtype=self.datatype
            ),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            
            #Since the labels are integers from 0-15, softmax will be applied within CrossEntropy(), this linear
            #mapping will suffice in mapping our output to one of the transmitters
            #Softmax classification layer
            nn.Linear(
                in_features=128,
                out_features=num_transmitters,
                dtype=self.datatype
            )
        )
        
    def forward(self, x):
        #Power normalization
        
        x = self.ConvolutionalLayers(x)
        
        x = x.view(x.size(0), -1)

        #Softmax classification layer
        x = self.classifer(x)#Logits for CrossEntropyLoss
        
        return x
        
        
        
    