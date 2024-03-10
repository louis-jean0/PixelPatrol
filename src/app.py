import customtkinter as ctk
from tkinter import filedialog
from PIL import Image,ImageTk
import os
from detection import detection

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Détecteur de falsification d'images")
        self.geometry("800x600")
        self._set_appearance_mode("Dark")

        # Choisir une image
        self.btn_choisir_image = ctk.CTkButton(self,text="Choisir une image",command=self.choisir_image)
        self.btn_choisir_image.pack(padx=(0,0),pady=(0,0))

        # Zone d'affichage de l'image
        self.canvas_image = ctk.CTkCanvas(self,width=600,height=400)
        self.canvas_image.pack(padx=(0,0),pady=(0,0))

        # Bouton pour lancer la détection
        self.btn_detection = ctk.CTkButton(self,text="Lancer la détection",command=self.detecter)
        self.btn_detection.pack(padx=(0,0),pady=(0,0))

        self.image_path = None

    def choisir_image(self):
        self.image_path = filedialog.askopenfilename()
        self.afficher_image(self.image_path)

    def afficher_image(self,path):
        image = Image.open(path)
        image.thumbnail((600,400))
        self.image_tk = ImageTk.PhotoImage(image)
        self.canvas_image.create_image(300,200,image=self.image_tk)

    def detecter(self):
        if self.image_path:
            image_detectee_path = detection(self.image_path)
            self.afficher_image(image_detectee_path)

if __name__ == "__main__":
    app = Application()
    app.mainloop()