import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
from detection import copy_move_detection, detection_dct, creer_masque_dct, detection_svm, calculer_metriques

class Application(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Pixel Patrol")
        self.geometry("2048x1152")
        ctk.set_appearance_mode("Dark")
        self.image_path = None
        self.taille_bloc = tk.IntVar(value=4)
        self.btn_dct = None

        # Conteneur global à gauche pour les boutons
        self.buttons_frame = ctk.CTkFrame(self)
        self.buttons_frame.pack(side='left', padx=(20, 20), pady=(20, 20), fill='both')

        # Conteneur global pour les métriques
        self.metrics_frame = ctk.CTkFrame(self)
        self.metrics_frame.pack(side='bottom', padx=(20, 20), pady=(20, 20), fill='both')

        # Labels pour les métriques
        self.label_accuracy = ctk.CTkLabel(self.metrics_frame, text="Précision : N/A")
        
        self.label_recall = ctk.CTkLabel(self.metrics_frame, text="Rappel : N/A")

        self.label_f1_score = ctk.CTkLabel(self.metrics_frame, text="F1-score : N/A")

        self.label_jaccard = ctk.CTkLabel(self.metrics_frame, text="Indice de Jaccard : N/A")

        self.label_prediction_svm = ctk.CTkLabel(self.metrics_frame, text="Prédiction du SVM : N/A")

        # Choisir une image
        self.btn_choisir_image = ctk.CTkButton(self.buttons_frame, text="Choisir une image", command=self.choisir_image, hover_color="darkgrey")
        self.btn_choisir_image.pack(pady=(10, 10))

        # Conteneur global à droite pour les canvas
        self.canvas_frame = ctk.CTkFrame(self)
        self.canvas_frame.pack(side='left', padx=(20, 20), pady=(20, 20), fill='both', expand=True)

        # Canvas pour l'image originale
        self.canvas_image = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)

        # Canvas pour l'image avec correspondances
        self.canvas_image_detectee = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)
        
        # Canvas pour l'image avec masques
        self.canvas_masque = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)

        # Cavas pour l'image du masque de vérité
        self.canvas_masque_verite = ctk.CTkCanvas(self.canvas_frame, width=400, height=400)

        # Ajout d'un menu déroulant pour le choix du mode de détection
        self.mode_detection_var = tk.StringVar()
        self.mode_detection_var.set("Choisir le mode de détection")
        self.modes_detection = ["DCT", "SIFT", "SVM"]
        self.menu_mode_detection = ctk.CTkOptionMenu(self.buttons_frame, variable=self.mode_detection_var, values=self.modes_detection, command=self.mode_selectionne)
        self.menu_mode_detection.pack(pady=(10, 10))

        # SIFT
        self.btn_sift = ctk.CTkButton(self.buttons_frame, text="Lancer la détection", command=self.detecter_sift, hover_color="darkgrey")

        #DCT
        self.btn_dct = ctk.CTkButton(self.buttons_frame, text="Lancer la détection", command=self.detecter_dct, hover_color="darkgrey")

        # SVM
        self.button_classification = ctk.CTkButton(self.buttons_frame, text="Lancer la classification", command=self.detecter_svm, hover_color="darkgrey")

        # Taille des blocs
        self.label_taille_bloc = ctk.CTkLabel(self.buttons_frame,text="Taille des blocs : 4")
        self.slider_taille_bloc = ctk.CTkSlider(self.buttons_frame,from_=2,to=5,command=self.update_taille_bloc)

    def mode_selectionne(self, mode):
        # Ajuster l'UI en fonction du mode de détection choisi
        self.canvas_image.pack_forget()
        self.canvas_image_detectee.pack_forget()
        self.canvas_masque.pack_forget()
        self.canvas_masque_verite.pack_forget()
        self.label_accuracy.pack_forget()
        self.label_recall.pack_forget()
        self.label_f1_score.pack_forget()
        self.label_jaccard.pack_forget()
        self.btn_dct.pack_forget()
        self.btn_sift.pack_forget()
        self.button_classification.pack_forget()
        self.slider_taille_bloc.pack_forget()
        self.label_taille_bloc.pack_forget()

        if mode == "DCT":
            self.canvas_image.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_image.create_text(200, 20, text="Image falsifiée", font=("Arial", 12), fill="white")
            self.canvas_image_detectee.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_image_detectee.create_text(200, 20, text="Image détectée", font=("Arial", 12), fill="white")
            self.canvas_masque.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_masque.create_text(200, 20, text="Masque prédit", font=("Arial", 12), fill="white")
            self.canvas_masque_verite.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_masque_verite.create_text(200, 20, text="Masque de vérité", font=("Arial", 12), fill="white")
            self.label_taille_bloc.pack(pady=(20, 2))
            self.slider_taille_bloc.pack(pady=(0, 20))
            self.btn_sift.pack_forget()
            self.button_classification.pack_forget()
            self.btn_dct.pack(pady=(10, 10))
            
        elif mode == "SIFT":
            self.canvas_image.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_image.create_text(200, 20, text="Image falsifiée", font=("Arial", 12), fill="white")
            self.canvas_image_detectee.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_image_detectee.create_text(200, 20, text="Image des points clés SIFT", font=("Arial", 12), fill="white")
            self.canvas_masque.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_masque.create_text(200, 20, text="Masque prédit", font=("Arial", 12), fill="white")
            self.canvas_masque_verite.pack(side='left', padx=(20, 20), pady=(20, 20))
            self.canvas_masque_verite.create_text(200, 20, text="Masque de vérité", font=("Arial", 12), fill="white")
            self.btn_dct.pack_forget()
            self.slider_taille_bloc.pack_forget()
            self.label_taille_bloc.pack_forget()
            self.button_classification.pack_forget()
            self.btn_sift.pack(pady=(10, 10))

        elif mode == "SVM":
            self.canvas_image_detectee.pack_forget()
            self.canvas_masque.pack_forget()
            self.canvas_masque_verite.pack_forget()
            self.btn_dct.pack_forget()
            self.slider_taille_bloc.pack_forget()
            self.label_taille_bloc.pack_forget()
            self.btn_sift.pack_forget()
            self.canvas_image.pack(side='top', expand=True)
            self.canvas_image.create_text(200, 20, text="Image", font=("Arial", 12), fill="white")
            self.button_classification.pack(pady=(10, 10))

    def update_taille_bloc(self, value):
        nouvelle_taille_bloc = 2 ** int(value)
        self.taille_bloc.set(nouvelle_taille_bloc)
        self.label_taille_bloc.configure(text=f"Taille des blocs: {self.taille_bloc.get()}")

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
            image_masque_path = creer_masque_dct(image_detectee_path)
            self.afficher_image(image_detectee_path, self.canvas_image_detectee)
            self.afficher_image(image_masque_path, self.canvas_masque)
            nom_base = os.path.basename(self.image_path)
            dossier_data_masks = "../data/splicing/masks"
            nom_masque_verite = nom_base.replace('.jpg','.png')
            chemin_masque_verite = os.path.join(dossier_data_masks, nom_masque_verite)
            if os.path.exists(chemin_masque_verite):
                self.afficher_image(chemin_masque_verite, self.canvas_masque_verite)
                rappel,precision,f1,jaccard = calculer_metriques(chemin_masque_verite, image_masque_path)
                self.label_accuracy.pack(pady=(10, 5))
                self.label_accuracy.configure(text="Précision : {:.1f}%".format(precision*100))
                self.label_recall.pack(pady=(5, 5))
                self.label_recall.configure(text="Rappel : {:.1f}%".format(rappel*100))
                self.label_f1_score.pack(pady=(5, 5))
                self.label_f1_score.configure(text="F1-score : {:.1f}%".format(f1*100))
                self.label_jaccard.pack(pady=(5, 10))
                self.label_jaccard.configure(text="Indice de Jaccard : {:.1f}%".format(jaccard*100))
            else:
                print(f"Le masque de vérité correspondant à {nom_base} n'existe pas.")
                self.canvas_masque_verite.delete("all")

    def detecter_sift(self):
        if self.image_path:
            # Détection avec la méthode SIFT et affichage de l'image résultante
            image_correspondances_path, image_masque_path = copy_move_detection(self.image_path)
            self.afficher_image(image_correspondances_path, self.canvas_image_detectee)
            self.afficher_image(image_masque_path, self.canvas_masque)
            nom_base = os.path.basename(self.image_path)
            if "_F.png" in nom_base:
                nom_masque_verite = nom_base.replace('_F.png', '_M.png')
                dossier_data_masks = "../data/copy-move/masks"
                chemin_masque_verite = os.path.join(dossier_data_masks, nom_masque_verite)
                # Vérifier si le masque de vérité existe
                if os.path.exists(chemin_masque_verite):
                    # Afficher le masque de vérité
                    self.afficher_image(chemin_masque_verite, self.canvas_masque_verite)
                    rappel,precision,f1,jaccard = calculer_metriques(chemin_masque_verite, image_masque_path)
                    self.label_accuracy.pack(pady=(10, 5))
                    self.label_accuracy.configure(text="Précision : {:.1f}%".format(precision*100))
                    self.label_recall.pack(pady=(5, 5))
                    self.label_recall.configure(text="Rappel : {:.1f}%".format(rappel*100))
                    self.label_f1_score.pack(pady=(5, 5))
                    self.label_f1_score.configure(text="F1-score : {:.1f}%".format(f1*100))
                    self.label_jaccard.pack(pady=(5, 10))
                    self.label_jaccard.configure(text="Indice de Jaccard : {:.1f}%".format(jaccard*100))
                else:
                    print(f"Le masque de vérité correspondant à {nom_base} n'existe pas.")
                    self.canvas_masque_verite.delete("all")
            else:
                print(f"Le masque de vérité correspondant à {nom_base} n'existe pas.")
                self.canvas_masque_verite.delete("all")
    
    def detecter_svm(self):
        if self.image_path:
            prediction = detection_svm(self.image_path)
            resultat = "authentique" if prediction == 0 else "falsifiée"
            self.label_prediction_svm.configure(text=f"Prédiction du SVM : image {resultat}")
            self.label_prediction_svm.pack(pady=(10,10))   

if __name__ == "__main__":
    app = Application()
    app.mainloop()