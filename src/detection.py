from PIL import Image
import numpy as np
import math

def detection(image_path): # Cette méthode découpe l'image en petits blocs et compare leurs caractéristiques (moyenne, écart-type) pour déterminer si il y a eu copy-move
    image = Image.open(image_path)
    if(image.mode != "L"):
        image = image.convert("L") # On passe l'image en niveaux de gris
    image_np = np.array(image) # On transforme l'image en un tableau np pour pouvoir manipuler les pixels
    largeur,hauteur = image.size
    taille_bloc = 16
    blocs = [(x, y, image_np[y:y+taille_bloc, x:x+taille_bloc]) for y in range(0, hauteur, taille_bloc) for x in range(0, largeur, taille_bloc)] # On découpe en blocs carrés de taille taille_bloc
    blocs_suspects = set()
    moyenne_globale = image_np.mean()
    ecart_type_global = image_np.std()
    for i in range(len(blocs)):
        for j in range(i+1,len(blocs)):
            bloc_i = blocs[i][2]
            bloc_j = blocs[j][2]
            distance = math.sqrt((blocs[i][0] - blocs[j][0])**2 + (blocs[i][1] - blocs[j][1])**2) # On calcule la distance entre deux blocs (car les blocs adjacents posent des problèmes de faux positif)
            if(bloc_i.shape == bloc_j.shape and distance < 64): # Si les deux blocs sont de même taille et si ils sont séparés par au moins deux blocs
                diff_moy = abs(bloc_i.mean() - bloc_j.mean())
                diff_ecart_type = abs(bloc_i.std() - bloc_j.std())
                if(diff_moy < moyenne_globale / 1000 and diff_ecart_type < ecart_type_global / 1000): # On met des conditions sur les caractéristiques
                    blocs_suspects.add(i)
                    blocs_suspects.add(j)
    image_reconstruite = Image.new("L",(largeur,hauteur))
    for index, (x,y,bloc) in enumerate(blocs):
        if index in blocs_suspects:
            print(x,y)
            bloc.fill(0) # On remplace les pixels des blocs suspects par du noir
        bloc_image = Image.fromarray(bloc)
        image_reconstruite.paste(bloc_image,(x,y))
    image_reconstruite_path = "test.pgm"
    image_reconstruite.save(image_reconstruite_path)
    return image_reconstruite_path