import streamlit as st
from streamlit_option_menu import option_menu
from PIL import Image
import numpy as np
import pyperclip
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def preprocess_image(image):
    # Convert image to grayscale if it's not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
    # Resize and pad the image to 256x64
    (h, w) = image.shape
    final_img = np.ones([64, 256]) * 255  # blank white image
    
    # Crop or pad the image
    if w > 256:
        image = image[:, :256]
    if h > 64:
        image = image[:64, :]
    
    final_img[:h, :w] = image
    final_img = cv2.rotate(final_img, cv2.ROTATE_90_CLOCKWISE)
    
    # Convert to PyTorch tensor and normalize to [0, 1]
    final_img = torch.tensor(final_img, dtype=torch.float32) / 255.0
    final_img = final_img.unsqueeze(0).unsqueeze(0)  # Add batch and channel dimensions
    
    return final_img
class CNNtoRNN(nn.Module):
    def __init__(self, num_of_characters):
        super(CNNtoRNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # (3, 3) kernel, 'same' padding
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        # Max pooling layers
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)  # (2, 2) kernel
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)  # (2, 2) kernel
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 2))  # (1, 2) kernel
        
        # Dropout layers
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout(0.3)
        
        # Fully connected layer
        self.fc1 = nn.Linear(128 * 8, 64)  # input features = 128 * 8
        
        # Bidirectional LSTM layers
        self.lstm1 = nn.LSTM(64, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(512, 256, bidirectional=True, batch_first=True)
        
        # Final fully connected layer
        self.fc2 = nn.Linear(512, num_of_characters)
        
    def forward(self, x, check_shape= False):
        # Convolution + BatchNorm + ReLU + MaxPool
        x = self.maxpool1(F.relu(self.bn1(self.conv1(x))))
        if check_shape:
            print(f'After conv1: {x.shape}')
        x = self.maxpool2(F.relu(self.bn2(self.conv2(x))))
        if check_shape:
            print(f'After conv2: {x.shape}')     
        x = self.dropout1(x)
        x = self.maxpool3(F.relu(self.bn3(self.conv3(x))))
        if check_shape:
            print(f'After conv3: {x.shape}')
        x = self.dropout2(x)
        # Reshape from (batch, channels, height, width) to (batch, height, channels * width)
        batch, channels, height, width = x.size()
        x = x.permute(0, 2, 1, 3).contiguous().view(batch, height, channels * width)
        if check_shape:
            print(f'After reshape: {x.shape}')   
        # Fully connected layer
        x = F.relu(self.fc1(x))
        if check_shape:
            print(f'After fc1: {x.shape}')
        # Bidirectional LSTM layers
        x, _ = self.lstm1(x)
        if check_shape:
            print(f'After lstm1: {x.shape}')
        x, _ = self.lstm2(x)
        if check_shape:
            print(f'After lstm2: {x.shape}')
        # Output layer
        x = self.fc2(x)
        x = F.log_softmax(x, dim=2)
        if check_shape:
            print(f'After fc2 (log_softmax): {x.shape}')
        return x

# Example usage
num_of_characters = 31  # Number of possible characters
model = CNNtoRNN(num_of_characters)

# Create a random tensor with the shape of your input data
input_tensor = torch.randn(1, 1, 256, 64)  # (batch_size, channels, height, width)

# Forward pass to print the shape of each layer's output
output = model(input_tensor, check_shape= True)

model= torch.load("./models/model.pth", map_location=torch.device('cpu'))

criterion = nn.CTCLoss(blank=0,zero_infinity=True)  # 0 is the index for the CTC blank label
optimizer = optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 30
device = torch.device('cpu') # if torch.cuda.is_available() else 'cpu'
model.to(device)

# Decoding function
def greedy_decoder(output, labels):
    output = output.cpu().numpy()
    arg_maxes = np.argmax(output, axis=2)
    decodes = []
    for i in range(arg_maxes.shape[1]):
        args = arg_maxes[:, i]
        decode = []
        for j in range(args.shape[0]):
            index = args[j]
            if index != 0:
                decode.append(labels[index])
        decodes.append(decode)
    return decodes

labels = ['<BLANK>', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '-', "'", '`', ' ']

def extract_text(image):
    image = preprocess_image(image)
    with torch.no_grad():
        outputs = model(image)
        outputs = outputs.permute(1, 0, 2)  # Shape should be (seq_len, batch, num_of_characters)
        outputs = F.log_softmax(outputs, dim=2)
        decoded_output = greedy_decoder(outputs, labels)
        
    return decoded_output[0]



st.sidebar.title("Settings")

st.title('Handwritting recognition')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    image = np.array(image.convert('L'))
    extracted_text = extract_text(image)

    
    st.text_area("Extracted Text", extracted_text, height=150)
    
    # Copy text
    if st.button('Copy Text'):
        pyperclip.copy(extracted_text)
        st.success("Text copied successfully!")
else:
    st.header("Please upload an image.")
