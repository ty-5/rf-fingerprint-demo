import torch
import pandas as pd
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import configs

class OracleDataset(Dataset):
    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe
        
        #Map each transmitter ID to an numerical index ex ['3123D7B'] becomes (0, '3123D7B')
        self.labels = sorted(self.df['label'].unique())
        self.label_to_index = {label: idx for idx, label in enumerate(self.labels)}
    
    def __len__(self):
        return len(self.df)
    
    #getitem is necessary for PyTorch to iterate over our dataset and acquire data in the form of (data_value, label_id)
    def __getitem__(self, idx):
        row = self.df.iloc[idx] #iloc = index-location, it returns a row at that index
        
        #Convert the real and imaginary columns from the DF into tensors
        real = torch.tensor(row['real'], dtype=configs.datatype) #Experiment with torch.complex128 as specified in ORACLE dataset
        imag = torch.tensor(row['imag'], dtype=configs.datatype)
        
        #Stack the Interphase/Quadrature data tensors into one
        IQ_Data_tensor = torch.stack([real, imag]) #shape [2, WindowSize]
        
        #Normalize each window by the mean and std computed across all values, this is a standard scalar
        IQ_mean = IQ_Data_tensor.mean()
        IQ_std = IQ_Data_tensor.std()
        IQ_Data_tensor = (IQ_Data_tensor - IQ_mean) / (IQ_std + 1e-8) #Add epsilon value to avoid division by 0
         
        #Convert the label column to indices
        label_str = row['label']
        label_index = self.label_to_index[label_str]
        
        #Return a tuple of both value the IQ data tensor and the label that goes with it
        return IQ_Data_tensor, label_index
    
    
if __name__ == '__main__':
    #Unpickle the data: convert from byte stream to pandas dataframe
    df = pd.read_pickle(r"oracle_rf_baseline.pkl")
    dataset = OracleDataset(df)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for batch_x, batch_y in dataloader:
        #Should be in shape [batch_size, 2, window_size] 2 because we stacked two tensors, real and imaginary
        print("Batch Shape: ", batch_x.shape)
        #Labels are in shape [1, batch_size] which contain the indices that the labels are mapped to
        print("Labels: ", batch_y)
        break;    
        











