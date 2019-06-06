from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.datasets import cifar10

(X_train, y_train0), (x_test, y_test) = cifar10.load_data()

from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(y_train0)
y_train=enc.transform(y_train0).toarray()


img = Input(shape = (32,32,3))
model = ResNet50(
weights = 'imagenet',
include_top = False, 
input_tensor = img, 
input_shape = None, 
pooling = 'avg'
)

final_layer = model.layers[-1].output
dense_layer_1 = Dense(128, activation = 'relu')(final_layer)
output_layer = Dense(10, activation = 'softmax')(dense_layer_1)
model = Model(inputs = img, outputs = output_layer)
model.compile(optimizer = 'adam',metrics = ['accuracy'], 
loss = 'categorical_crossentropy')

model.fit(
X_train, y_train, 
batch_size = 32, 
epochs = 50, 
validation_split = 0.2)

