import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from detection import copy_move_detection, detection_dct

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Détecteur de falsification d'images")
        self.geometry("1600x900")
        ctk.set_appearance_mode("Dark")
        self.image_path = None
        self.taille_bloc = tk.IntVar(value=8)
        self.btn_dct = None

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

        # Canvas pour l'image avec correspondances
        self.canvas_image_detectee = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)
        
        # Canvas pour l'image avec masques
        self.canvas_masque = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)

        # Ajout d'un menu déroulant pour le choix du mode de détection
        self.mode_detection_var = tk.StringVar()
        self.mode_detection_var.set("Choisir le mode de détection")
        self.modes_detection = ["DCT", "SIFT"]
        self.menu_mode_detection = ctk.CTkOptionMenu(self.buttons_frame, variable=self.mode_detection_var, values=self.modes_detection, command=self.mode_selectionne)
        self.menu_mode_detection.pack(pady=(10, 10))

        # Taille des blocs
        self.label_taille_bloc = ctk.CTkLabel(self.buttons_frame,text="Taille des blocs : 8")
        self.slider_taille_bloc = ctk.CTkSlider(self.buttons_frame,from_=3,to=5,command=self.update_taille_bloc)

    def mode_selectionne(self, mode):
        # Ajuster l'UI en fonction du mode de détection choisi
        self.canvas_image.pack_forget()
        self.canvas_image_detectee.pack_forget()
        self.canvas_masque.pack_forget()
        if mode == "DCT":
            self.canvas_image.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_image_detectee.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.label_taille_bloc.pack(pady=(20, 2))
            self.slider_taille_bloc.pack(pady=(0, 20))
            self.btn_sift.pack_forget()
            if(self.btn_dct == None):
                self.btn_dct = ctk.CTkButton(self.buttons_frame, text="Lancer la détection", command=self.detecter_dct, hover_color="darkgrey")
                self.btn_dct.pack(pady=(10, 10))
            
        elif mode == "SIFT":
            self.canvas_image.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_image_detectee.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_masque.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.btn_dct.pack_forget()
            self.btn_dct = None
            self.slider_taille_bloc.pack_forget()
            self.label_taille_bloc.pack_forget()
            self.btn_sift.pack(pady=(10, 10))
    
    def update_taille_bloc(self, value):
        nouvelle_taille_bloc = 2 ** int(value)
        self.taille_bloc.set(nouvelle_taille_bloc)
        self.label_taille_bloc.configure(text=f"Taille des blocs: {self.taille_bloc.get()}")
        print(self.taille_bloc.get())

    # Choisir une image parmi l'explorateur de fichiers
    def choisir_image(self):
        dossier_data = os.path.join(os.path.dirname(__file__),"../data")
        self.image_path = filedialog.askopenfilename(initialdir=dossier_data)
        self.afficher_image(self.image_path, self.canvas_image)

    # Afficher une image sur un canvas
    def afficher_image(self, path, canvas):
        image = Image.open(path)
        image.thumbnail((canvas.winfo_width(), canvas.winfo_height()))
        image_tk = ImageTk.PhotoImage(image)
        x = (canvas.winfo_width() - image_tk.width()) / 2
        y = (canvas.winfo_height() - image_tk.height()) / 2
        canvas.delete("all")
        canvas.create_image(x, y, anchor='nw', image=image_tk)
        canvas.image_tk = image_tk

    def detecter_dct(self):
        if self.image_path:
            image_detectee_path = detection_dct(self.image_path,self.taille_bloc.get())
            self.afficher_image(image_detectee_path, self.canvas_image_detectee)

    def detecter_sift(self):
        if self.image_path:
            image_correspondances_path,image_masque_path = copy_move_detection(self.image_path)
            self.afficher_image(image_correspondances_path,self.canvas_image_detectee)
            self.afficher_image(image_masque_path,self.canvas_masque)

if __name__ == "__main__":
    app = Application()
    app.mainloop()