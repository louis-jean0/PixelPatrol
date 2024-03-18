import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from detection import detection_kmeans, sift

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Détecteur de falsification d'images")
        self.geometry("1200x650")  # Ajusté pour plus d'espace pour les sliders
        self.taille_bloc = tk.IntVar(value=4)
        self.nb_clusters = tk.IntVar(value=10)
        self.image_path = None

        # Choisir une image
        self.btn_choisir_image = ctk.CTkButton(self,text="Choisir une image",command=self.choisir_image,hover_color="darkgrey")
        self.btn_choisir_image.pack(pady=(20, 10),padx=(20, 20),side='left')

        # Zone d'affichage de l'image originale
        self.canvas_image = ctk.CTkCanvas(self,width=400,height=400)
        self.canvas_image.pack(pady=(0, 0),padx=(20, 20),side='left')

        # Zone d'affichage de l'image détectée
        self.canvas_image_detectee = ctk.CTkCanvas(self,width=400, height=400)
        self.canvas_image_detectee.pack(pady=(0, 0), padx=(20, 20), side='left')

        # Taille des blocs
        self.label_taille_bloc = ctk.CTkLabel(self,text="Taille des blocs : 16")
        self.label_taille_bloc.pack(pady=(20, 2))
        self.slider_taille_bloc = ctk.CTkSlider(self,from_=2,to=7,variable=self.taille_bloc,command=self.update_taille_bloc)
        self.slider_taille_bloc.pack(pady=(0, 20))

        # Nombre de clusters
        self.label_nb_clusters = ctk.CTkLabel(self, text="Nombre de clusters : 10")
        self.label_nb_clusters.pack(pady=(0, 2))
        self.slider_nb_clusters = ctk.CTkSlider(self,from_=2,to=20,variable=self.nb_clusters,command=self.update_nb_clusters)
        self.slider_nb_clusters.pack(pady=(0, 20))

        # Conteneur pour les boutons
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.pack(pady=(10, 20), side='top')

        # Bouton pour lancer la détection
        self.btn_detection = ctk.CTkButton(self.buttons_frame, text="Lancer la détection", command=self.detecter, hover_color="darkgrey")
        self.btn_detection.pack(pady=(10, 5), side='top')

        # Bouton pour lancer l'affichage des points d'intêret
        self.btn_sift = ctk.CTkButton(self.buttons_frame, text="Lancer SIFT", command=self.detecter_sift, hover_color="darkgrey")
        self.btn_sift.pack(pady=(5, 10), side='bottom')

    def choisir_image(self):
        self.image_path = filedialog.askopenfilename()
        self.afficher_image(self.image_path, self.canvas_image)

    def afficher_image(self, path, canvas):
        image = Image.open(path)
        image.thumbnail((512, 512))
        image_tk = ImageTk.PhotoImage(image)
        canvas.create_image(300, 200, image=image_tk)
        canvas.image_tk = image_tk

    def detecter(self):
        if self.image_path:
            image_detectee_path = detection_kmeans(self.image_path,self.taille_bloc.get(),self.nb_clusters.get())
            self.afficher_image(image_detectee_path, self.canvas_image_detectee)
    
    def update_taille_bloc(self, value):
        self.taille_bloc = 2 ** int(value)
        self.label_taille_bloc.configure(text=f"Taille des blocs: {self.taille_bloc}")

    def update_nb_clusters(self, value):
        self.nb_clusters = int(value)
        self.label_nb_clusters.configure(text=f"Nombre de clusters: {int(value)}")

    def detecter_sift(self):
        if self.image_path:
            image_detectee_path = sift(self.image_path) 
            self.afficher_image(image_detectee_path, self.canvas_image_detectee)

if __name__ == "__main__":
    app = Application()
    app.mainloop()