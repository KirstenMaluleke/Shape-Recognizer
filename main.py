import pickle
import os.path
import tkinter.messagebox
from tkinter import *
from tkinter import simpledialog, filedialog
import PIL
from PIL import Image, ImageDraw
import cv2 as cv
import numpy as np

from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier

class ShapeRecognition:
    def __init__(self):
        self.shape1, self.shape2, self.shape3, self.shape4, self.shape5 = None, None, None, None, None
        self.shape1_counter, self.shape2_counter, self.shape3_counter, self.shape4_counter, self.shape5_counter = None, None, None, None, None
        self.project_name = None
        self.classify = None
        self.root = None
        self.img1 = None

        self.status_label = None
        self.canvas = None
        self.draw = None
        self.brush_width = 10
        self.class_prompt()
        self.init_ui()

    def class_prompt(self):
        msg = Tk()
        msg.withdraw()
        self.project_name = simpledialog.askstring("Project Name", "Enter your chosen project name below", parent = msg)
        if os.path.exists(self.project_name):
            with open(f"{self.project_name}/{self.project_name}_data.pickle", "rb") as f:
                data = pickle.load(f)
            self.shape1 = data['s1']
            self.shape2 = data['s2']
            self.shape3 = data['s3']
            self.shape4 = data['s4']
            self.shape5 = data['s5']
            self.shape1_counter = data['count1']
            self.shape2_counter = data['count2']
            self.shape3_counter = data['count3']
            self.shape4_counter = data['count4']
            self.shape5_counter = data['count5']
            self.classify = data['classify']
            self.project_name = data['proj_name']
        else:
            self.shape1 = simpledialog.askstring("Shape 1", "What is the name of the first shape?", parent=msg)
            self.shape2 = simpledialog.askstring("Shape 2", "What is the name of the second shape?", parent=msg)
            self.shape3 = simpledialog.askstring("Shape 3", "What is the name of the third shape?", parent=msg)
            self.shape4 = simpledialog.askstring("Shape 4", "What is the name of the fourth shape?", parent=msg)
            self.shape5 = simpledialog.askstring("Shape 5", "What is the name of the fifth shape?", parent=msg)
            self.shape1_counter = 1
            self.shape2_counter = 1
            self.shape3_counter = 1
            self.shape4_counter = 1
            self.shape5_counter = 1

            self.classify = LinearSVC()
            os.mkdir(self.project_name)
            os.chdir(self.project_name)
            os.mkdir(self.shape1)
            os.mkdir(self.shape2)
            os.mkdir(self.shape3)
            os.mkdir(self.shape4)
            os.mkdir(self.shape5)
            os.chdir("..")

    def init_ui(self):
        WIDTH = 500
        HEIGHT = 500
        WHITE = (250, 250, 250)
        self.root = Tk()
        self.root.title(f"Shape Recognizer - {self.project_name}")
        self.canvas = Canvas(self.root, width=WIDTH, height=HEIGHT, bg="white")
        self.canvas.pack(expand=YES, fill=BOTH)
        self.canvas.bind("<B1-Motion>", self.paint)
        self.img1 = PIL.Image.new("RGB", (WIDTH, HEIGHT), WHITE)
        self.draw = PIL.ImageDraw.Draw(self.img1)

        b_frame = tkinter.Frame(self.root)
        b_frame.pack(fill=X, side=BOTTOM)
        b_frame.columnconfigure(0, weight=1)
        b_frame.columnconfigure(1, weight=1)
        b_frame.columnconfigure(2, weight=1)
        b_frame.columnconfigure(3, weight=1)
        b_frame.columnconfigure(4, weight=1)

        shape1_button = Button(b_frame, text=self.shape1, command=lambda: self.save(1))
        shape1_button.grid(row=0, column=0, sticky=W+E)

        shape2_button = Button(b_frame, text=self.shape2, command=lambda: self.save(2))
        shape2_button.grid(row=0, column=1, sticky=W + E)

        shape3_button = Button(b_frame, text=self.shape3, command=lambda: self.save(3))
        shape3_button.grid(row=0, column=2, sticky=W + E)

        shape4_button = Button(b_frame, text=self.shape4, command=lambda: self.save(4))
        shape4_button.grid(row=0, column=3, sticky=W + E)

        shape5_button = Button(b_frame, text=self.shape5, command=lambda: self.save(5))
        shape5_button.grid(row=0, column=4, sticky=W + E)

        b_minus = Button(b_frame, text='- Brush', command=self.brush_minus)
        b_minus.grid(row=1, column=1, sticky=W + E)

        clear_btn = Button(b_frame, text='Clear', command=self.clear)
        clear_btn.grid(row=1, column=2, sticky=W + E)

        b_plus = Button(b_frame, text='+ Brush', command=self.brush_plus)
        b_plus.grid(row=1, column=0, sticky=W + E)

        train = Button(b_frame, text='Train the model', command=self.train_validate)
        train.grid(row=1, column=3, sticky=W + E)

        save_btn = Button(b_frame, text='Save the Model', command=self.save_model)
        save_btn.grid(row=1, column=4, sticky=W + E)

        load_btn = Button(b_frame, text='Load the Model', command=self.load_model)
        load_btn.grid(row=2, column=0, sticky=W + E)

        rotate_btn = Button(b_frame, text='Rotate Model', command=self.rotate_model)
        rotate_btn.grid(row=2, column=1, sticky=W + E)

        predict_btn = Button(b_frame, text='Predict the Shape', command=self.predict)
        predict_btn.grid(row=2, column=2, sticky=W + E)

        save_all_btn = Button(b_frame, text='Save All', command=self.save_all)
        save_all_btn.grid(row=2, column=3, sticky=W + E)

        self.status_label = Label(b_frame, text=f"Current Model: {type(self.classify).__name__}")
        self.status_label.config(font=("Arial", 10))
        self.status_label.grid(row=3, column=1, sticky=W + E)

        self.root.protocol("WM_WINDOW_DELETE", self.on_closing)
        self.root.attributes("-topmost", True)
        self.root.mainloop()

    def save(self, class_no):
        self.img1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.LANCZOS)

        if class_no == 1:
            img.save(f"{self.project_name}/{self.shape1}/{self.shape1_counter}.png", "PNG")
            self.shape1_counter += 1

        if class_no == 2:
            img.save(f"{self.project_name}/{self.shape2}/{self.shape2_counter}.png", "PNG")
            self.shape2_counter += 1

        if class_no == 3:
            img.save(f"{self.project_name}/{self.shape3}/{self.shape3_counter}.png", "PNG")
            self.shape3_counter += 1

        if class_no == 4:
            img.save(f"{self.project_name}/{self.shape4}/{self.shape4_counter}.png", "PNG")
            self.shape4_counter += 1

        if class_no == 5:
            img.save(f"{self.project_name}/{self.shape5}/{self.shape5_counter}.png", "PNG")
            self.shape5_counter += 1

        self.clear()

    def paint(self, event):
        x1, y1 = (event.x -1), (event.y -1)
        x2, y2 = (event.x +1), (event.y +1)
        self.canvas.create_rectangle(x1, y1, x2, y2, fill="black", width=self.brush_width)
        self.draw.rectangle([x1, y1, x2 + self.brush_width, y2 + self.brush_width], fill="black", width=self.brush_width)

    def brush_plus(self):
        self.brush_width += 1

    def brush_minus(self):
        if self.brush_width > 1:
            self.brush_width -= 1

    def clear(self):
        self.canvas.delete("all")
        self.draw.rectangle([0, 0, 1000, 1000], fill="white")

    def train_validate(self):
        image_ls = np.array([])
        shape_ls = np.array([])

        for i in range(1, (self.shape1_counter)//2):
            img = cv.imread(f"{self.project_name}/{self.shape1}/{i}.png")[:, :, 0]
            img = img.reshape(2500)
            image_ls = np.append(image_ls, [img])
            shape_ls = np.append(shape_ls, 1)

        for i in range(1, (self.shape2_counter)//2):
            img = cv.imread(f"{self.project_name}/{self.shape2}/{i}.png")[:, :, 0]
            img = img.reshape(2500)
            image_ls = np.append(image_ls, [img])
            shape_ls = np.append(shape_ls, 2)

        for i in range(1, (self.shape3_counter)//2):
            img = cv.imread(f"{self.project_name}/{self.shape3}/{i}.png")[:, :, 0]
            img = img.reshape(2500)
            image_ls = np.append(image_ls, [img])
            shape_ls = np.append(shape_ls, 3)

        for i in range(1, (self.shape4_counter)//2):
            img = cv.imread(f"{self.project_name}/{self.shape4}/{i}.png")[:, :, 0]
            img = img.reshape(2500)
            image_ls = np.append(image_ls, [img])
            shape_ls = np.append(shape_ls, 4)

        for i in range(1, self.shape5_counter // 2):
            img = cv.imread(f"{self.project_name}/{self.shape5}/{i}.png")[:, :, 0]
            img = img.reshape(2500)
            image_ls = np.append(image_ls, [img])
            shape_ls = np.append(shape_ls, 5)

        image_ls = image_ls.reshape(self.shape1_counter + self.shape2_counter + self.shape3_counter + self.shape4_counter + self.shape5_counter - 140, 2500)

        self.classify.fit(image_ls, shape_ls)

        acc_1 = 0
        acc_2 = 0
        acc_3 = 0
        acc_4 = 0
        acc_5 = 0
        correct_count1, correct_count2, correct_count3, correct_count4, correct_count5 = 0, 0, 0, 0, 0

        # Helper function to flatten the image
        def load_and_flatten_image(path):
            img = cv.imread(path, cv.IMREAD_GRAYSCALE)
            if img is not None:
                return img.flatten()
            else:
                return None

        # Predict for shape1
        for i in range(self.shape1_counter // 2, self.shape1_counter):
            img_flat = load_and_flatten_image(f"{self.project_name}/{self.shape1}/{i}.png")
            if img_flat is not None:
                prediction = self.classify.predict([img_flat])
                if prediction[0] == 1:
                    correct_count1 += 1
        acc_1 = correct_count1 / (self.shape1_counter // 2) if self.shape1_counter // 2 > 0 else 0

        # Predict for shape2
        for i in range(self.shape2_counter // 2, self.shape2_counter):
            img_flat = load_and_flatten_image(f"{self.project_name}/{self.shape2}/{i}.png")
            if img_flat is not None:
                prediction = self.classify.predict([img_flat])
                if prediction[0] == 2:
                    correct_count2 += 1
        acc_2 = correct_count2 / (self.shape2_counter // 2) if self.shape2_counter // 2 > 0 else 0

        # Predict for shape3
        for i in range(self.shape3_counter // 2, self.shape3_counter):
            img_flat = load_and_flatten_image(f"{self.project_name}/{self.shape3}/{i}.png")
            if img_flat is not None:
                prediction = self.classify.predict([img_flat])
                if prediction[0] == 3:
                    correct_count3 += 1
        acc_3 = correct_count3 / (self.shape3_counter // 2) if self.shape3_counter // 2 > 0 else 0

        # Predict for shape4
        for i in range(self.shape4_counter // 2, self.shape4_counter):
            img_flat = load_and_flatten_image(f"{self.project_name}/{self.shape4}/{i}.png")
            if img_flat is not None:
                prediction = self.classify.predict([img_flat])
                if prediction[0] == 4:
                    correct_count4 += 1
        acc_4 = correct_count4 / (self.shape4_counter // 2) if self.shape4_counter // 2 > 0 else 0

        # Predict for shape5
        for i in range(self.shape5_counter // 2, self.shape5_counter):
            img_flat = load_and_flatten_image(f"{self.project_name}/{self.shape5}/{i}.png")
            if img_flat is not None:
                prediction = self.classify.predict([img_flat])
                if prediction[0] == 5:
                    correct_count5 += 1
        acc_5 = correct_count5 / (self.shape5_counter // 2) if self.shape5_counter // 2 > 0 else 0

        # Calculate total accuracy
        acc_tot = (acc_1 + acc_2 + acc_3 + acc_4 + acc_5) / 5
        tkinter.messagebox.showinfo("Shape Recognizer", f"The model was successfully trained. The validation test had "
                                                        f"an accuracy of {acc_tot}", parent=self.root)

    def save_model(self):
        file_path = filedialog.asksaveasfilename(defaultextension="pickle")
        with open(file_path, "wb") as f:
            pickle.dump(self.classify, f)
        tkinter.messagebox.showinfo("Shape Recognizer", "Model was successfully saved", parent=self.root)

    def load_model(self):
        file_path = filedialog.askopenfilename()
        with open(file_path, "rb") as f:
            self.classify = pickle.load(f)
        tkinter.messagebox.showinfo("Shape Recognizer", "Model was successfully loaded", parent=self.root)

    def rotate_model(self):
        if isinstance(self.classify, LinearSVC):
            self.classify = KNeighborsClassifier()

        elif isinstance(self.classify, KNeighborsClassifier):
            self.classify = LogisticRegression()

        elif isinstance(self.classify, LogisticRegression):
            self.classify = DecisionTreeClassifier()

        elif isinstance(self.classify, DecisionTreeClassifier):
            self.classify = RandomForestClassifier()

        elif isinstance(self.classify, RandomForestClassifier):
            self.classify = GaussianNB

        elif isinstance(self.classify, GaussianNB):
            self.classify = LinearSVC
        self.status_label.config(text=f"Current Model: {type(self.classify).__name__}")



    def predict(self):
        self.img1.save("temp.png")
        img = PIL.Image.open("temp.png")
        img.thumbnail((50, 50), PIL.Image.LANCZOS)
        img.save("test_shape.png", "PNG")

        img = cv.imread("test_shape.png")[:, :, 0]
        img = img.reshape(2500)
        prediction = self.classify.predict([img])

        if prediction[0] == 1:
            tkinter.messagebox.showinfo("Shape Recognizer", f"The shape drawn is a {self.shape1}", parent=self.root)

        if prediction[0] == 2:
            tkinter.messagebox.showinfo("Shape Recognizer", f"The shape drawn is a {self.shape2}", parent=self.root)

        if prediction[0] == 3:
            tkinter.messagebox.showinfo("Shape Recognizer", f"The shape drawn is a {self.shape3}", parent=self.root)

        if prediction[0] == 4:
            tkinter.messagebox.showinfo("Shape Recognizer", f"The shape drawn is a {self.shape4}", parent=self.root)

        if prediction[0] == 5:
            tkinter.messagebox.showinfo("Shape Recognizer", f"The shape drawn is a {self.shape5}", parent=self.root)
    def save_all(self):
        data = {"s1": self.shape1, "s2": self.shape2, "s3": self.shape3, "s4": self.shape4, "s5": self.shape5,  "count1": self.shape1_counter, "count2": self.shape2_counter, "count3": self.shape3_counter, "count4": self.shape4_counter, "count5": self.shape5_counter, "classify": self.classify, "proj_name": self.project_name}
        with open(f"{self.project_name}/{self.project_name}_data.pickle", "wb") as f:
            pickle.dump(data, f)
        tkinter.messagebox.showinfo("Shape Recognizer", "The Project has been successfully saved", parent=self.root)
    def on_closing(self):
        response = tkinter.messagebox.askyesnocancel("Exit?", "Do you want to save your work?", parent=self.root)
        if response is not None:
            if response:
                self.save_all()
            self.root.destroy()
            exit()


ShapeRecognition()
