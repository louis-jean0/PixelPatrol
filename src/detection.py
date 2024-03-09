from PIL import Image
import numpy as np

def detection(image): # Cette méthode découpe l'image en petits blocs et compare leurs caractéristiques (moyenne, écart-type) pour déterminer si il y a eu copy-move
    if(image.mode != "L"):
        image = image.convert("L") # On passe l'image en niveaux de gris
    image = np.array(image) # On transforme l'image en un tableau np pour pouvoir manipuler les pixels
    blocs = [image[x:x+16,y:y+16] for x in range(0,image.shape[0],16) for y in range(0,image.shape[1],16)] # On découpe en blocs
    moyennes_blocs = [] # Pour stocker la moyenne de chaque bloc
    ecart_types_blocs = [] # Pour stocker l'écart-type de chaque bloc
    for bloc in blocs:
        moyennes_blocs.append(bloc.mean())
        ecart_types_blocs.append(bloc.std())
    blocs_suspects = set()
    for i in range(0,len(blocs)):
        for j in range(i+1,len(blocs)):
            if(i == len(blocs) - 1): # Pour éviter le dépassement quand on arrive au dernier bloc
                break 
            diff_moy = abs(moyennes_blocs[i] - moyennes_blocs[j])
            diff_ecart_type = abs(ecart_types_blocs[i] - ecart_types_blocs[j])
            if(diff_moy < 0.1 and diff_ecart_type < 0.001):
                blocs_suspects.add(i)
                blocs_suspects.add(j)
    print(blocs_suspects)
    for suspect in blocs_suspects:
        blocs[suspect].fill(0)
    image_reconstruite = Image.new(mode="L",size=(image.shape[0],image.shape[1]))
    for bloc in blocs:
        image_reconstruite.paste(Image.fromarray(np.uint8(bloc)).convert("L"))
    return image_reconstruite

def main():

    image = Image.open("../data/001_F.png")
    image_sus = detection(image)
    image_sus.save("test.pgm")

main()
