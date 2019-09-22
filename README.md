# GrayNet-Keras
* A versatile model for Deep learning application for CT images
* Keras implementation

## Publications
Ston AI paper [link](https://pubs.rsna.org/doi/10.1148/ryai.2019180066)   
C-MIMI 2019 presentation [link](https://cdn.ymaws.com/siim.org/resource/resmgr/mimi19/oral4/GrayNet_Kim.pdf)  

## How to install
```shell
## go to your project folder
$ cd <your project path>
$ git clone https://github.com/LMIC-MGH/GrayNet-Keras
```
## How to use
```python
from keras.layers import Input, GlobalAveragePooling2D, Dense
from keras.models import Model
from GrayNetKeras import DenseNet121_GrayNet

# Set your input
input_tensor = Input((256,256), name='input')

# Add densnet archtecture with pretrained weight of graynet
# last layer of graynet is global average pooling layer
gap = DenseNet121_GrayNet(input_tensor=input_tensor, weights='graynet')

# Set a fully connected layer for your model 
output = Dense(n_label=1, activation='sigmoid', name='fc')(gap) ## Your label

model = Model(inputs=input_tensor, outputs=output, name='main_model')
model.summary()

# ... compile your model and run!
# Please see example.ipynb for more example (Jupyter notebook)
# Model archtecture 
```