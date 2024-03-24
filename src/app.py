import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from detection import detection_kmeans, copy_move_detection

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Détecteur de falsification d'images")
        self.geometry("1800x900")
        ctk.set_appearance_mode("Dark")
        self.image_path = None

        # Conteneur global à gauche pour les boutons
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.pack(side='left', padx=(20, 20), pady=(20, 20), fill='y')

        # Choisir une image
        self.btn_choisir_image = ctk.CTkButton(self.buttons_frame, text="Choisir une image", command=self.choisir_image, hover_color="darkgrey")
        self.btn_choisir_image.pack(pady=(10, 10))

        # Bouton pour lancer la détection
        self.btn_sift = ctk.CTkButton(self.buttons_frame, text="Lancer la détection", command=self.detecter_sift, hover_color="darkgrey")
        self.btn_sift.pack(pady=(10, 10))

        # Conteneur global à droite pour les canvas
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.pack(side='left', padx=(20, 20), pady=(20, 20), fill='both', expand=True)

        # Canvas pour l'image originale
        self.canvas_image = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)
        self.canvas_image.pack(side='left', padx=(20, 20), pady=(20, 20))

        # Canvas pour l'image avec correspondances
        self.canvas_image_detectee = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)
        self.canvas_image_detectee.pack(side='left', padx=(20, 20), pady=(20, 20))

        # Canvas pour l'image avec masques
        self.canvas_masque = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)
        self.canvas_masque.pack(side='left', padx=(20, 20), pady=(20, 20))

    # Choisir une image parmi l'explorateur de fichiers
    def choisir_image(self):
        self.image_path = filedialog.askopenfilename()
        self.afficher_image(self.image_path, self.canvas_image)

    # Afficher une image sur un canvas
    def afficher_image(self, path, canvas):
        image = Image.open(path)
        image.thumbnail((canvas.winfo_width(), canvas.winfo_height()))
        # Convertir l'image PIL en format utilisable par Tkinter
        image_tk = ImageTk.PhotoImage(image)
        x = (canvas.winfo_width() - image_tk.width()) / 2
        y = (canvas.winfo_height() - image_tk.height()) / 2
        canvas.delete("all")
        canvas.create_image(x, y, anchor='nw', image=image_tk)
        canvas.image_tk = image_tk

    def detecter(self):
        if self.image_path:
            image_detectee_path = detection_kmeans(self.image_path,self.taille_bloc.get(),self.nb_clusters.get())
            self.afficher_image(image_detectee_path, self.canvas_image_detectee)

    def detecter_sift(self):
        if self.image_path:
            image_correspondances_path,image_masque_path = copy_move_detection(self.image_path)
            self.afficher_image(image_correspondances_path,self.canvas_image_detectee)
            self.afficher_image(image_masque_path,self.canvas_masque)

if __name__ == "__main__":
    app = Application()
    app.mainloop()