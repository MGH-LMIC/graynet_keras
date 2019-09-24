# GrayNet-Keras
* A versatile model for Deep learning application for CT images
* Keras implementation

## Publications
(Stone AI) Urinary Stone Detection on CT Images Using Deep Convolutional Neural Networks: Evaluation of Model Performance and Generalization, Radiology AI, 2019, [link](https://pubs.rsna.org/doi/10.1148/ryai.2019180066)     
(Organ segmentation) C-MIMI 2019 presentation [link](https://cdn.ymaws.com/siim.org/resource/resmgr/mimi19/oral4/GrayNet_Kim.pdf)  


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
from graynet_keras import DenseNet121_GrayNet

# Set your input
input_shape = (256,256,1)
input_tensor = Input(input_shape, name='input')

# Add densnet archtecture with pretrained weight of graynet
# last layer of graynet is global average pooling layer
model = Densenet121_GrayNet(input_tensor=input_tensor, weights='graynet', w_reg=None)

# Set a fully connected layer for your model 
output = model.output
output = Dense(units=1, activation='sigmoid', name='fc')(output) ## Your label

model = Model(inputs=input_tensor, outputs=output, name='main_model')
model.summary()

# ... compile your model and run!
# Please see example.ipynb for more example (Jupyter notebook)
# Model archtecture 
```

### How to excute Jupyter example codes
```shell
cp GrayNet_example.ipynb ../GrayNet_example.ipynb
```
Then, see & run the codes!
