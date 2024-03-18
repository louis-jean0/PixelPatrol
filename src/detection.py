from PIL import Image,ImageDraw
import numpy as np
import math
import cv2
from sklearn.cluster import KMeans
from skimage.segmentation import slic
from skimage import io

"""
def detection(image_path): # Cette méthode découpe l'image en petits blocs et compare leurs caracteristiques (moyenne, écart-type) pour déterminer si il y a eu copy-move
    image = Image.open(image_path)
    if(image.mode != "L"):
        image = image.convert("L") # On passe l'image en niveaux de gris
    image_np = np.array(image) # On transforme l'image en un tableau np pour pouvoir manipuler les pixels
    largeur,hauteur = image.size
    taille_bloc = 32 # On choisit la taille des blocs (taille_bloc x taille_bloc)
    blocs = [(x, y, image_np[y:y+taille_bloc, x:x+taille_bloc]) for y in range(0, hauteur, taille_bloc) for x in range(0, largeur, taille_bloc)] # On découpe en blocs carrés de taille taille_bloc
    blocs_suspects = set()
    moyenne_globale = image_np.mean() # On calcule la moyenne de l'image de base
    ecart_type_global = image_np.std() # On calcule l'écart-type de l'image de base
    for i in range(len(blocs)):
        for j in range(len(blocs)):
            if(i == j):
                continue
            bloc_i = blocs[i][2] # On accède aux pixels du bloc i
            bloc_j = blocs[j][2] # On accède aux pixels du bloc j
            distance = math.sqrt((blocs[i][0] - blocs[j][0])**2 + (blocs[i][1] - blocs[j][1])**2) # On calcule la distance entre deux blocs (car les blocs adjacents posent des problèmes de faux positif)
            if(bloc_i.shape == bloc_j.shape and distance > math.log2(taille_bloc)*taille_bloc): # Si les deux blocs sont de même taille et si ils sont assez distants (j'ai mis log2 temporairement pour que ce soit adaptatif à la taille des blocs)
                diff_moy = abs(bloc_i.mean() - bloc_j.mean()) # On calcule les écarts de moyennes et d'écart-types
                diff_ecart_type = abs(bloc_i.std() - bloc_j.std())
                if(diff_moy < moyenne_globale / 100 and diff_ecart_type < ecart_type_global / 1000): # On met des conditions sur les caracteristiques
                    blocs_suspects.add(i) # On ajoute le bloc i aux blocs_suspects
                    blocs_suspects.add(j) # On ajoute le bloc j aux blocs_suspects
    image_reconstruite = Image.new("L",(largeur,hauteur)) # On créé l'image reconstruite, c'est-à-dire l'image qui montrera les zones détectées
    for index,(x,y,bloc) in enumerate(blocs):
        if index in blocs_suspects:
            print(x,y)
            bloc.fill(0) # On remplace les pixels des blocs suspects par du noir
        bloc_image = Image.fromarray(bloc)
        image_reconstruite.paste(bloc_image,(x,y))
    image_reconstruite_path = "test.pgm" # On choisit le chemin de l'image reconstruite
    image_reconstruite.save(image_reconstruite_path) # On enregistre l'image reconstruite
    return image_reconstruite_path # On retourne le chemin de l'image reconstruite (car c'est comme cela que j'ai écrit l'application qui va l'utiliser)
"""

def kmeans(caracteristiques, k, max_iters=10):
    np.random.seed(42) # Pour pouvoir reproduire les résultats (s'affranchir du random pour les premiers tests)
    centres = caracteristiques[np.random.choice(caracteristiques.shape[0],k,replace=False),:]
    for _ in range(max_iters):
        distances = np.linalg.norm(caracteristiques - centres[:,np.newaxis],axis=2)
        clusters_plus_proches = np.argmin(distances,axis=0)
        nouveaux_centres = np.array([caracteristiques[clusters_plus_proches == j].mean(axis=0) for j in range(k)])
        if np.all(centres == nouveaux_centres):
            break
        centres = nouveaux_centres
    return clusters_plus_proches

def histogramme_ndg(image):
    histogramme,_ = np.histogram(image,bins=256,range=(0,256))
    histogramme_normalise = histogramme / np.sum(histogramme)
    return histogramme_normalise

def histogramme_couleur(image):
    histogramme_couleur = [histogramme_ndg(image[:,:,i]) for i in range(3)]
    return histogramme_couleur

def local_binary_pattern(image):
    largeur,hauteur = image.shape
    lbp_image = np.zeros_like(image)
    for y in range(1,hauteur-1):
        for x in range(1,largeur-1):
            centre = image[y,x]
            binary = ''
            for ny,nx in [(y-1,x-1),(y-1,x),(y-1,x+1),(y,x+1),(y+1,x+1),(y+1,x),(y+1,x-1),(y,x-1)]:
                binary += '1' if image[ny,nx] >= centre else '0'
            lbp_image[y,x] = int(binary,2)
    return lbp_image

def lbp_couleur(image):
    lbp_image = np.zeros_like(image)
    for i in range(0,3):
        canal = image[:,:,i]
        lbp_canal = local_binary_pattern(canal)
        lbp_image[:,:,i] = lbp_canal
    return lbp_image

def dct_2d(image):
    M,N = image.shape
    dct = np.zeros((M,N))
    for u in range(M):
        for v in range(N):
            sum_uv = 0
            for x in range(M):
                for y in range(N):
                    cos_x_u = np.cos((2*x+1) * u * np.pi / (2*M))
                    cos_y_v = np.cos((2*y+1) * v * np.pi / (2*N))
                    sum_uv += image[x,y] * cos_x_u * cos_y_v
            alpha_u = np.sqrt(1/M) if u == 0 else np.sqrt(2/M)
            alpha_v = np.sqrt(1/N) if v == 0 else np.sqrt(2/N)
            dct[u,v] = alpha_u * alpha_v * sum_uv
    return dct

def detection_kmeans(image_path, taille_bloc=16, k=5):
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    image_ndg = image.convert("L")
    image_np_ndg = np.array(image_ndg)
    image_np = np.array(image)
    largeur, hauteur = image.size
    caracteristiques = []
    blocs_positions = []
    for y in range(0, hauteur, taille_bloc):
        for x in range(0, largeur, taille_bloc):
            bloc = image_np[y:y+taille_bloc,x:x+taille_bloc,:]
            bloc_ndg = image_np_ndg[y:y+taille_bloc,x:x+taille_bloc]
            moyennes = [bloc[:, :, i].mean() for i in range(3)]  # Moyenne pour R, G, B
            ecarts_types = [bloc[:, :, i].std() for i in range(3)]  # Écart-type pour R, G, B
            histogrammes = np.concatenate(histogramme_couleur(bloc))  # Histogrammes pour R, G, B
            lbp = lbp_couleur(bloc)
            histogrammes_lbp = np.concatenate(histogramme_couleur(lbp))
            #dct = dct_2d(bloc_ndg)
            #dct = dct[:8, :8].flatten()
            vecteur_caracteristiques = np.concatenate((moyennes,ecarts_types,histogrammes,histogrammes_lbp))#,dct))
            caracteristiques.append(vecteur_caracteristiques)
            blocs_positions.append((x, y))
    caracteristiques = np.array(caracteristiques)
    labels = kmeans(caracteristiques, k, 100)
    _, counts = np.unique(labels, return_counts=True)
    cluster_suspect = np.argmin(counts)
    print(cluster_suspect)
    blocs_confirmes = []
    for index, (x, y) in enumerate(blocs_positions):
        if labels[index] != cluster_suspect:
            continue
        compteur_voisins = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue  # Skip le bloc lui-même
                voisin_x, voisin_y = x + dx * taille_bloc, y + dy * taille_bloc
                voisin_index = next((i for i, pos in enumerate(blocs_positions) if pos == (voisin_x, voisin_y)), None)
                if voisin_index is not None and labels[voisin_index] == cluster_suspect:
                    compteur_voisins += 1
        if compteur_voisins >= 2:
            blocs_confirmes.append(index)
    image_reconstruite = Image.new("RGB", (largeur, hauteur))
    for index, (x, y) in enumerate(blocs_positions):
        bloc = image_np[y:y+taille_bloc, x:x+taille_bloc, :]
        if index in blocs_confirmes:
            bloc = np.zeros_like(bloc)
        bloc_image = Image.fromarray(bloc, "RGB")
        image_reconstruite.paste(bloc_image, (x, y))
    image_reconstruite_path = "test_color.png"
    image_reconstruite.save(image_reconstruite_path)
    return image_reconstruite_path

def sift(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints,descriptors = sift.detectAndCompute(image, None)
    image_keypoints = cv2.drawKeypoints(image,keypoints,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    image_keypoints = cv2.cvtColor(image_keypoints,cv2.COLOR_BGR2RGB)
    image_keypoints_path = "image_keypoints.png"
    cv2.imwrite(image_keypoints_path,image_keypoints)
    return image_keypoints_path