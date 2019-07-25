"""
Dates were of the format --> 30012018
HOH were of the format --> 1, 2, 3, ...
restaurant were in original format
Area was in the format of 10000 to 13800 converted to excel standardize

To get original arrangement back sort by 1) date 2) HOH 3) Area 4) restaurant id
"""
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import sys
import pyprind
import matplotlib.pyplot as plt

DATA_PATH = "time_series_data.csv"

data_frame = pd.read_csv(DATA_PATH)
data_array = np.array(data_frame)
train_data = torch.from_numpy(data_array)
input_indices = torch.tensor([0, 1, 2, 3])
truth_index = torch.tensor([4])
train_truth = np.array(data_frame["Tot_orders"])

# plot
plt.plot(train_truth[0:1000], label='sales of 1st 1000')
# plt.show()

#####################
# Set parameters
#####################

# Data params
# num_train = int((1 - test_size) * num_datapoints)
num_train = 1

# Network params
input_size = 20
# If `per_element` is True, then LSTM reads in one timestep at a time.
per_element = True
if per_element:
    lstm_input_size = 1
else:
    lstm_input_size = input_size
# size of hidden layers
h1 = 235  # TODO: Make this 235 and check results
output_dim = 1
num_layers = 4
learning_rate = 1e-3
num_epochs = 5
dtype = torch.float


#####################
# Generate data
#####################
# data = ARData(num_datapoints, num_prev=input_size, test_size=test_size, noise_var=noise_var,
#               coeffs=fixed_ar_coefficients[input_size])
# print(data.X_train.shape)

# make training and test sets in torch
# X_train = torch.from_numpy(data.X_train).type(torch.Tensor)
# X_test = torch.from_numpy(data.X_test).type(torch.Tensor)
# y_train = torch.from_numpy(data.y_train).type(torch.Tensor).view(-1)
# y_test = torch.from_numpy(data.y_test).type(torch.Tensor).view(-1)
#
# X_train = X_train.view([input_size, -1, 1])
# X_test = X_test.view([input_size, -1, 1])


#####################
# Build model
#####################

# Here we define our model as a class
class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, batch_size, output_dim=1,
                 num_layers=4):
        super(LSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.num_layers = num_layers

        # Define the LSTM layer
        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        # Define the output layer
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        # This is what we'll initialise our hidden state as
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_dim))

    def forward(self, input):
        # print("line 100", input)
        # Forward pass through LSTM layer
        # shape of lstm_out: [input_size, batch_size, hidden_dim]
        # shape of self.hidden: (a, b), where a and b both
        # have shape (num_layers, batch_size, hidden_dim).
        lstm_out, self.hidden = self.lstm(input.view(len(input), self.batch_size, -1))

        # Only take the output from the final timetep
        # Can pass on the entirety of lstm_out to the next layer if it is a seq2seq prediction
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred.view(-1)


model = LSTM(lstm_input_size, h1, batch_size=num_train, output_dim=output_dim, num_layers=num_layers)

loss_fn = torch.nn.MSELoss(size_average=False)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#####################
# Train model
#####################

hist = np.zeros(num_epochs)

for t in range(num_epochs):
    # Initialise hidden state
    # Don't do this if you want your LSTM to be stateful
    model.hidden = model.init_hidden()
    running_loss = 0.0
    # print("This is the epoch", t)
    bar = pyprind.ProgBar(len(train_data), stream=sys.stdout)
    for i, data in enumerate(train_data):
        # print("line 131", train_data[10000])
        input_tensor = torch.index_select(data, 0, input_indices).float()
        truth_tensor = torch.index_select(data, 0, truth_index).float()
        # Forward pass
        y_pred = model(input_tensor)

        loss = loss_fn(y_pred, truth_tensor)

        running_loss += loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimizer.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimizer.step()
        if i % 1000 == 0:
            print("progress = ", (i / len(train_data) * 100.0))
        bar.update()
    # if t % 100 == 0:
    print("Epoch ", t, "MSE: ", running_loss / len(train_data))
    hist[t] = running_loss / len(train_data)

#####################
# Plot preds and performance
#####################

plt.plot(y_pred.detach().numpy(), label="Preds")
plt.plot(train_truth, label="Data")
plt.legend()
plt.show()

plt.plot(hist, label="Training loss")
plt.legend()
plt.show()

# if __name__ == '__main__':
#
