import torch
import torch.nn as nn
from training_data_loader import OracleDataset
from torch.utils.data import DataLoader, random_split
import pandas as pd
from CNN_Extended import CNNFingerprinter
import torch.optim as optim
from torchviz import make_dot
import time, sys
import configs
from sklearn.model_selection import train_test_split
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________           


#Hyperparameters:
BATCH_SIZE = 64 #ORACLE uses 1024, experiment with this
EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY=1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"

#DATA PROCESSING
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________           
#Load the data
df = pd.read_pickle(r"oracle_rf_ALL_DATA.pkl")

assert "label" in df.columns, "No column named 'Label'"


train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
validation_df, testing_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

training_dataset = OracleDataset(train_df)
validation_dataset = OracleDataset(validation_df)
testing_dataset = OracleDataset(testing_df)

print(f"Max label index: {max(training_dataset.label_to_index.values())}")


training_dataloader = DataLoader(training_dataset, batch_size=BATCH_SIZE, shuffle=True)
validation_dataloader = DataLoader(validation_dataset, batch_size=BATCH_SIZE, shuffle=False)
testing_dataloader = DataLoader(testing_dataset, batch_size=BATCH_SIZE, shuffle=False)


#FUNCTIONS
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________           
def train_model(data: DataLoader, model, optimizer, loss_function):
    model.train()
    current_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    #1. represents features or the IQ_Data_Tensor from training_data_loader.py
    #2. represents labels from training_data_loader.py
    for _, (signal_features, labels) in enumerate(data):
        
        signal_features = signal_features.to(device)
        labels = labels.to(device)    
        
        #Clear the gradients from the computational graph
        optimizer.zero_grad()
        
        #Forward propagation
        prediction = model(signal_features)
        
        #Compute loss
        loss = loss_function(prediction, labels)
        
        #Backward propagation
        loss.backward()
        
        #Gradient clipping will reduce any exploding gradients by capping the maximum size
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        #Update the weights
        optimizer.step()
        
        current_loss += loss.item()
        _, predicted_label = torch.max(prediction.detach(), 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted_label == labels).sum().item()
        
        
    return correct_predictions, total_predictions, current_loss


def test_model(data: DataLoader, model, loss_function):
    model.eval()
    current_validation_loss = 0
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        for (signal, label) in data:
            signal = signal.to(device)
            label = label.to(device)
            
            output = model(signal)
            current_validation_loss += loss_function(output, label).item()
            
            _, predicted_label = torch.max(output.detach(), 1)
            total_predictions += label.size(0)
            correct_predictions += (predicted_label == label).sum().item()
    
    return correct_predictions, total_predictions, current_validation_loss
       
#_____________________________________________________________________________________________________________________________________________________________________________________________________________________________________________________           
#MAIN FUNCTION
if __name__ == "__main__":
    print("Let's begin training...")
    model = CNNFingerprinter().to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min', #When the quantity stops decreasing
        factor=0.5, #Lower it by 50%
        patience=5, #If the loss does not improve for 5 epochs, then lower learning rate
    )
    #Early stopping patience
    patience = 10 #Number of epochs we will wait for the validation/testing loss to improve
    patience_counter = 0
    min_validation_loss = 1e9
    min_validation_loss = float(min_validation_loss)
    
    #Training loop
    for epoch in range(EPOCHS):
        start_time = time.time()
        
        #One epoch of training
        correct_predictions, total_predictions, current_loss = train_model(training_dataloader, model, optimizer, loss_function)
        
        #One epoch of testing - we'll test as we go to assess the testing loss
        validation_correct_predictions, validation_total_predictions, current_validation_loss = test_model(validation_dataloader, model, loss_function)
        
        #Track our learning data
        epoch_time = time.time() - start_time
        training_accuracy = 100 * (correct_predictions/total_predictions)
        print(f"Epoch = {epoch + 1}/{EPOCHS}, Training Loss = {(current_loss / len(validation_dataloader)):.6f}, Training Accuracy = {training_accuracy:.2f}%, Epoch time = {epoch_time:.2f} seconds")
        validation_accuracy = 100 * (validation_correct_predictions/validation_total_predictions)
        print(f"Validation Loss = {(current_validation_loss)/len(validation_dataloader):.6f}, Validation Accuracy = {validation_accuracy:.2f}%")
        
        #Use the scheduler to update the learning rate with respect to the validation/test loss
        scheduler.step(current_validation_loss)
        
        #Early stopping
        if(current_validation_loss < min_validation_loss):
            min_validation_loss = current_validation_loss
            patience_counter = 0
            torch.save(model.state_dict(), "RF_Model_Weights.pth")
            print("Validation loss improved, saving weights.")
        else:
            patience_counter += 1
            print(f"Validation Loss did not improve for {patience_counter}/{patience} epochs...")
            if (patience_counter >= patience):
                print(f"Early stopping at epoch: {epoch+1}")
                break
        
        #Output segmentation between epochs
        print("_" * 100)
    
    print("Training Complete, saving weights.")
    torch.save(model.state_dict(), "RF_Model_Weights.pth")
    
    
    
    

