#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
from PIL import Image


# In[2]:


import torch


# In[3]:


import torch.nn as nn
from torchvision import transforms


# In[4]:


class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 32 * 32, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# In[5]:


class_names = ['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'sheep', 'spider', 'squirrel']


# In[6]:


model = SimpleCNN(num_classes=len(class_names))
model.load_state_dict(torch.load("simple_cnn_animals.pth", map_location=torch.device("cpu")))
model.eval()


# In[7]:


transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])


# In[8]:


st.title("🐾 Hayvan Sınıflandırıcı AI")
st.write("Lütfen bir hayvan fotoğrafı yükleyin:")

uploaded_file = st.file_uploader("📷 Resim Seç", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)


# In[10]:


if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(img_tensor)
        _, predicted = torch.max(output, 1)
        predicted_class = class_names[predicted.item()]

    st.success(f"🧠 Tahmin: **{predicted_class.upper()}**")


# In[ ]:




