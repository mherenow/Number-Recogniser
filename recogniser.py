from keras.models import load_model
from tkinter import *
import tkinter as tk
import win32gui
from PIL import ImageGrab, Image
import numpy as np

model = load_model('Identifier.model')

def PredictDigit(img):
    
    img = img.resize((28,28))
    img = img.convert('L')
    img = np.array(img)

    img.reshape(1, 28, 28, 1)
    img = img/255.0

    res = model.predict([img])[0]
    return np.argmax(res), max(res)

class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)

        self.x = self.y = 0

        self.canvas = tk.Canvas(self, width =300, height=300, bg="white", cursor="cross")
        self.label = tk.Label(self, text="Wait...", font=("Legend", 48))
        self.classify_btn = tk.Button(self, text="Identify", command=lambda: self.identify_num)
        self.wipe_btn = tk.Button(self, text = "Clear", command = lambda:self.clear_scrn)

        self.canvas.grid(row=0, column=0, pady=2, sticky=W, )
        self.label.grid(row=0, column=0, padx=2, pady=2)
        self.classify_btn.grid(row=1, column=1, padx=2, pady=2)
        self.wipe_btn.grid(row=1, column=0, pady=2)

        self.canvas.bind("<B1-Motion>", lambda: self.draw)

        def clear_scrn(self):
            self.canvas.delete("all")
        
        def identify_num(self):
            loca = self.canvas.winfo_id()
            rect = win32gui.GetWindowRect(loca)
            im = ImageGrab.grab(rect)

            digit, acc = PredictDigit(im)
            self.label.configure(text= str(digit) + ', ' + str(int(acc*100) + '%'))

        def draw(self, event):
            self.x = event.x
            self.y = event.y
            r=8
            self.canvas.create_oval(self.x-r, self.y-r, self.x+r, self.y+r, fill = 'black')

app = App()
mainloop()
