import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from einops import rearrange, einsum

class MyModel(nn.Module):
    def __init__(self, num_bands, num_output_filters):
        super(MyModel, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(num_bands, num_output_filters, kernel_size=7, padding=1)
        self.conv2 = nn.Conv2d(num_output_filters, num_output_filters, kernel_size=5, padding=1)
        self.conv2 = nn.Conv2d(num_output_filters, num_output_filters, kernel_size=3, padding=1)
        
        # Linear transformation with trainable parameters
        self.linear_weights = nn.Parameter(torch.randn(num_output_filters))
        
    def forward(self, x):
        # Apply convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Perform matrix multiplication with trainable weights
        x = einsum('f, f n n -> n n', self.linear_weights, x)

        x = x.flatten()
        
        # Softmax activation
        x = F.softmax(x, dim=-1)
        
        return x

# Example usage
num_bands = 5
num_output_filters = 64
num_linear_params = 10  # Adjust as needed

model = MyModel(num_bands, num_output_filters)

# Assuming you have input_data and ground_truth_data defined similarly as before

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 100

for epoch in range(num_epochs):
    optimizer.zero_grad()
    outputs = model(input_data)
    
    # Compute loss
    loss = criterion(outputs, ground_truth_data)  # Adjust ground_truth_data as per your task
    
    # Backward pass and optimize
    loss.backward()
    optimizer.step()
    
    # Print progress
    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
