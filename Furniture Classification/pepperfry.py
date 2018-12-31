import numpy as np
import cv2
import os
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D as Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.utils import np_utils
from keras.metrics import categorical_accuracy
from sklearn.cross_validation import train_test_split
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt

def load_test_data(pth):
	X = []

	for img_name in os.listdir(pth):
		if(img_name[0] == '.') :
			continue
		img = cv2.imread(pth + '/' + img_name)
		X.append(proc_im(img))

	X = np.array(X)
	return X

def predict(model, img):
	x = proc_im(img)
	yprob = model.predict(np.array([x]))
	return np.argmax(yprob)

def class_number_to_label(num):
	mp = {
		0 : 'bed',
		1 : 'chair',
		2 : 'lamp',
		3 : 'shelf',
		4 : 'sofa',
		5 : 'stool',
		6 : 'table',
		7 : 'wardrobe'
	}
	return mp[num]
		

def proc_im(img):
	reszimg = cv2.resize(img, (64,64))
	reszimg = reszimg.astype('float32')
	return reszimg/255.0

def showim(img):
	plt.imshow(img)
	plt.show()


def perm(X, Y):
	size = X.shape[0]
	ysize = Y.shape[0]
	assert(size == ysize)
	p = np.random.permutation(size)
	return(X[p], Y[p])


def loadim(pth = '/home/s163/computer-networks/Furniture/'):
	# pth = '/home/s163/computer-networks/Furniture/'
	# folders = os.listdir(pth)
	mp = {
		'bed' : 0,
		'chair' : 1,
		'lamp' : 2,
		'shelf' : 3,
		'sofa' : 4,
		'stool' : 5,
		'table' : 6,
		'wardrobe' : 7
	}

	X = []
	Y = []

	for folder in os.listdir(pth):
		if(folder[0] == '.'):
			continue

		i = mp[folder[0:-4]]
		print("processing images for the class ", folder[0:-4], "with the label ", i)
		for img_name in os.listdir(pth + folder):
			if(img_name[0] == '.') :
				continue


			# print(pth + folder + '/' + img_name)
			img = cv2.imread(pth + folder + '/' + img_name)
			# print(img.shape)
			X.append(proc_im(img))
			Y.append(i)
			# print(pth + folder + '/' + img_name + " done")



	X = np.array(X)
	Y = to_categorical(Y)

	(X, Y) = perm(X, Y)
	np.save('X.npy', X)
	np.save('Y.npy', Y)



def make_model2(filter, shape_in, shape_out, eta):
	model = Sequential()

	model.add(Conv2D(32, kernel_size=filter, strides=(1, 1), activation='relu', input_shape=shape_in))
	model.add(Conv2D(64, filter, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.1))
	model.add(Conv2D(128, filter, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(256, filter, activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Flatten())
	model.add(Dropout(0.1))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.1))
	model.add(Dense(shape_out, activation='softmax'))        
	model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=eta), metrics=[categorical_accuracy])

	return model



def train(model, batch_sz):
	# X = np.load('X.npy')
	# Y = np.load('Y.npy')
	X_test = np.load('X_test.npy')
	X_validate = np.load('X_validate.npy')
	X_train = np.load('X_train.npy')
	Y_test = np.load('Y_test.npy')
	Y_validate = np.load('Y_validate.npy')
	Y_train = np.load('Y_train.npy')
	
	print(X_test.shape)
	print(X_validate.shape)
	print(X_train.shape)
	print(Y_test.shape)
	print(Y_validate.shape)
	print(Y_train.shape)


	# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5)
	# X_test, X_validate, Y_test, Y_validate = train_test_split(X_test, Y_test, test_size = 0.5)

	gen_spec = ImageDataGenerator(
		rotation_range=10,
		width_shift_range=0.1,
		height_shift_range=0.1,
		shear_range=0.2,
		zoom_range=0.1,
		horizontal_flip=True,
		    fill_mode='nearest')

	train_generator = gen_spec.flow(X_train, Y_train, batch_size = batch_sz)  
	validation_generator = gen_spec.flow(X_validate, Y_validate, batch_size = batch_sz)

	for i in range(4):
		model.fit_generator(train_generator,
			steps_per_epoch = X_train.shape[0] / batch_sz,
			epochs = 3,
			validation_data = validation_generator,
			validation_steps = X_validate.shape[0] / batch_sz)
		# model.save('model_' + str(i) + '.h5')
		score, acc = model.evaluate(X_test, Y_test, verbose=1)
		print('score: ', score, 'accuracy: ', acc)
		model.save('model.h5')

def gen_test_train_split():
	X = np.load('X.npy')
	Y = np.load('Y.npy')
	print(X.shape)

	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3)
	X_test, X_validate, Y_test, Y_validate = train_test_split(X_test, Y_test, test_size = 0.5)
	np.save('X_test.npy', X_test)
	np.save('Y_test.npy', Y_test)
	np.save('X_train.npy', X_train)
	np.save('Y_train.npy', Y_train)
	np.save('X_validate.npy', X_test)
	np.save('Y_validate.npy', Y_test)

def pred_class_number(model, X):
	# model = load_model('model')
	Yprob = model.predict(X)
	return np.argmax(Yprob, axis=1)

def get_confusion_matrix(model, X, Y):
	y_pred = pred_class_number(model, X)
	y_true = np.argmax(Y, axis=1)
	return confusion_matrix(y_true, y_pred)

def get_metrics(model, X, Y):
	pred_class = pred_class_number(model, X)
	y_pred = to_categorical(pred_class, 8)
	return precision_recall_fscore_support(Y, y_pred, average='weighted')

def print_metrics(model, X, Y):

	print("Printing the confusion matrix")
	print(get_confusion_matrix(model, X, Y))

	print("Accuracy: ", model.evaluate(X, Y, verbose=1)[1])
	prec, recall, fscore, supp = get_metrics(model, X, Y)

	print("Precision: ", prec)
	print("Recall: ", recall)
	print("F-measure: ", fscore)

def generate_and_train_model(dataset_path):
	shape_in = (64, 64, 3)
	shape_out = 8
	eta = 0.001
	batch_sz = 32
	kernel = (3, 3)	
	load_dim(dataset_path)
	gen_test_train_split()
	model = make_model2(kernel, shape_in, shape_out, eta)
	train(model, batch_sz)
	return model

# model = load_model('model.h5')
# X = np.load('../X.npy')
# Y = np.load('../Y.npy')
# for i in range(20):
# 	x = X[i]
# 	y = Y[i]
# 	ytrue = np.argmax(y)
# 	showim(x)
# 	ypred = pred_class_number(model, np.array([x]))
# 	# mp = {
# 	# 	'bed' : 0,
# 	# 	'chair' : 1,
# 	# 	'lamp' : 2,
# 	# 	'shelf' : 3,
# 	# 	'sofa' : 4,
# 	# 	'stool' : 5,
# 	# 	'table' : 6,
# 	# 	'wardrobe' : 7
# 	# }

# 	mp = {
# 		0 : 'bed',
# 		1 : 'chair',
# 		2 : 'lamp',
# 		3 : 'shelf',
# 		4 : 'sofa',
# 		5 : 'stool',
# 		6 : 'table',
# 		7 : 'wardrobe'
# 	}

# 	print("This is a", mp[ypred[0]])
# 	print("The data set says this a ", mp[ytrue])





# Note Y_test will be of dimensions sample_size x 8
# def test_model(model, X_test, Y_test)



# loadim()
# gen_test_train_split()
# model = make_model2(kernel, shape_in, shape_out, eta)
# model = load_model('model.h5')
# train(model, batch_sz)

# print(pred_class_number(model, X_test))
# print(get_metrics(model, X_test, Y_test))
# print_metrics(model, X_test, Y_test)

