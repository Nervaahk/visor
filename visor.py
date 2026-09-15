import torch 
import torch.nn as nn
import numpy as np
import matplotlib as plt
import torch.optim as optim
def F(input):
    return 3 + 1/input
limitpointa = 100000000
y = np.array([0.00001,0.0001,0.001,0.01,0.1,1,10,100,1000,10000,100000,1000000,10000000])
y = (y - np.mean(y)) / np.std(y)
y_tensor = torch.tensor(y, dtype=torch.float32).view(len(y),1,1)
f_y = F(y)
f_y_tensor = torch.tensor(f_y, dtype=torch.float32).view(len(f_y),1,1)
#limitpoint = 0
#limit_tensor = torch.tensor([limitpoint], dtype=torch.float32)
class limitModel(nn.Module):
    def __init__(self, hidden_size):
        super(limitModel,self).__init__()
        self.GRU = nn.GRU(input_size = 1, hidden_size = hidden_size, num_layers =2)
        self.Tanh = nn.Tanh()
        self.Linear  =nn.Linear(hidden_size, 1)
    def forward(self,x):
        out, _= self.GRU(x)
        out = self.Tanh(out)
        out = self.Linear(out)
    
        return out
hidden_size = 100
model = limitModel(hidden_size=hidden_size)
criterion = nn.MSELoss()
optimizer =optim.Adam(model.parameters(), lr= 0.01)
lambda_delta = 0.01
epoch = 0
max_epochs = 10000
delta_threshold = 0.0000001
while epoch <= max_epochs:
   optimizer.zero_grad()
   outputs = model(f_y_tensor)
   prediction = outputs.squeeze(-1)
   deltas = torch.abs(prediction[1:]-prediction[:-1])
   final_output = prediction[-1]

   
   loss = deltas[-10:].mean()


   loss.backward()
   optimizer.step()

   if epoch % 1000 == 0:
       print(f"Epoch:{epoch}, Loss: {loss}") 
   if loss.item() < delta_threshold:
        print("Training Complete.")
        break
   epoch += 1
model.eval()     
with torch.no_grad():
    prediction = model(f_y_tensor)
    print("Predicted Limit:", prediction[-1].item())
