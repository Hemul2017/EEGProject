


from rnn import Net
import torch
from torchsummary import summary

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = Net().to(device)
print(summary(model, input_size=(657, 64)))