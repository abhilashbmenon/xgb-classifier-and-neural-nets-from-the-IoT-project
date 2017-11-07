import numpy
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense


# load data
data = read_csv('iot_allow.csv', header=None)
dataset = data.values
# split data into X and y
X = dataset[:,0:5]
X = X.astype(str)
Y = dataset[:,5]

# encode string input values as integers
encoded_x = None
for i in range(0, X.shape[1]):
	label_encoder = LabelEncoder()
	feature = label_encoder.fit_transform(X[:,i])
	feature = feature.reshape(X.shape[0], 1)
	onehot_encoder = OneHotEncoder(sparse=False)
	feature = onehot_encoder.fit_transform(feature)
	if encoded_x is None:
		encoded_x = feature
	else:
		encoded_x = numpy.concatenate((encoded_x, feature), axis=1)
print("X shape: : ", encoded_x.shape)
# encode string class values as integers
label_encoder = LabelEncoder()
label_encoder = label_encoder.fit(Y)
label_encoded_y = label_encoder.transform(Y)
#Create model
model = Sequential()
model.add(Dense(28, input_dim=43, init='uniform', activation='relu'))
model.add(Dense(28, init='uniform', activation='relu'))
model.add(Dense(1, init='uniform', activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
history = model.fit(encoded_x, label_encoded_y, nb_epoch=150, batch_size=15, verbose=1)                    #CHANGED ON JULY 14th
#history = model.fit(encoded_x, label_encoded_y, validation_split=0.30, nb_epoch=10, batch_size=5)
#evaluate the model
scores = model.evaluate(encoded_x, label_encoded_y, verbose=1)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


#MISCELLANEOUS
# predictions = model.predict(X)
# rounded = [round(x[0]) for x in predictions]
# print(rounded)











