import tensorflow as tf
import matplotlib.pyplot as plt


class myCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs={}):
    if(logs.get('accuracy')>0.99):
      print("\nReached 99% accuracy so cancelling training!")
      self.model.stop_training = True

mnist_data = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist_data.load_data()
#print(len(x_train))
#plt.imshow(x_train[0])
#plt.show()
x_train = x_train.reshape(60000, 28, 28, 1)
x_train = x_train/255
x_test = x_test.reshape(10000, 28, 28, 1)
x_test = x_test/255
#print(x_train[0])

callbacks = myCallback()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(128, (3,3), activation = "relu", input_shape = (28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation = "relu", input_shape = (28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation = "relu"),
    tf.keras.layers.Dense(10, activation = "softmax")

])

#model.summary()

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs = 10, callbacks=[callbacks])

model.save('number.h5')
print("Model saved")

"""test_loss, test_acc = model.evaluate(x_test, y_test)
print("\n Accuracy : ", test_acc)

predictions = model.predict(x_test)
print(predictions)"""




