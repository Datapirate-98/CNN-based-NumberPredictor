from tkinter import *
from tkinter.ttk import *
import PIL.ImageGrab
from PIL import Image
import tensorflow as tf
import numpy as np


model = tf.keras.models.load_model('number.h5')

operator = "Predicted Number: "
operator2 = ""

def Clear():
    cv.delete("all")
    global operator2
    text_input.set(operator2)
def Predict():
    file = 'D:/image.jpg'
    if file:
        # save the canvas in jpg format
        x = root.winfo_rootx() + cv.winfo_x()
        y = root.winfo_rooty() + cv.winfo_y()
        x1 = x + cv.winfo_width()
        y1 = y + cv.winfo_height()
        PIL.ImageGrab.grab().crop((x, y, x1, y1)).save(file)
        img = Image.open(file).convert("L")
        img = img.resize((28, 28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1, 28, 28, 1)
        pred = model.predict_classes(im2arr)
        x = pred[0]
        global operator
        operator = operator + str(x)
        text_input.set(operator)
        operator = operator = "Predicted Number: "



def paint(event):
    old_x = event.x
    old_y = event.y

    cv.create_line(old_x, old_y, event.x, event.y,
                   width=20, fill="white",
                   capstyle=ROUND, smooth=TRUE, splinesteps=36)

root = Tk()
root.geometry("800x800")
# create string variable
text_input = StringVar()
textdisplay = Entry(root,
                    textvariable=text_input,
                    justify='center')
textdisplay.place(relx = 0.24, rely = 0.05, height=40, width = 400)
btn1 = Button(root, text="Predict", command=lambda: Predict())
btn1.place(relx = 0.35, rely = 0.15, height = 40, width = 90)
btn2 = Button(root, text="Clear", command=lambda: Clear())
btn2.place(relx = 0.5, rely = 0.15, height = 40, width = 90)
cv = Canvas(root, width=790, height=600, bg="black", )
cv.bind('<B1-Motion>', paint)
cv.grid(row=8, column=0)

root.rowconfigure(0, weight=2)
root.columnconfigure(1, weight=2)

root.mainloop()