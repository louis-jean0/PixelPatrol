import os
from PIL import Image
import numpy as np
import cv2
from skimage.measure import ransac
from skimage.transform import AffineTransform
from skimage.feature import local_binary_pattern
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, accuracy_score, f1_score, jaccard_score
from sklearn.preprocessing import StandardScaler
from joblib import load

# Détection de falsification par copy-move
def copy_move_detection(image_path):
    # Chargement de l'image et conversion en niveaux de gris
    image = cv2.imread(image_path)
    image_ndg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extraction des points clés et descripteurs avec SIFT
    detecteur_sift = cv2.SIFT_create()

    (points_cles, descripteurs) = detecteur_sift.detectAndCompute(image_ndg, None)

    descripteurs = np.asarray(descripteurs)
    produits_points = np.dot(descripteurs, descripteurs.transpose())
    normes = np.tile(np.sqrt(np.diag(produits_points)), (descripteurs.shape[1], 1)).transpose()
    descripteurs_normalises = descripteurs / (normes + np.finfo(float).eps)
    produits_points = np.dot(descripteurs_normalises, descripteurs_normalises.transpose())

    points_apparies_1 = []
    points_apparies_2 = []

    statut_points_apparies = [False] * len(points_cles)

    # Recherche de correspondances pour chaque point clé
    for i in range(len(points_cles)):
        produits_points[produits_points > 1] = 1
        angles = np.sort(np.arccos(produits_points[i, :]))
        indices_tries = np.argsort(np.arccos(produits_points[i, :]))

        # Évaluation des conditions pour une correspondance valide
        if angles[0] < 0.01 and \
           angles[1] / (angles[2] + np.finfo(float).eps) < 0.55 and \
           not statut_points_apparies[indices_tries[1]]:
           
            # Mise à jour du statut de correspondance
            statut_points_apparies[indices_tries[1]] = True
            statut_points_apparies[i] = True

            # Obtention des coordonnées des points clés et évaluation de leur distance
            point_actuel = points_cles[i].pt
            meilleur_correspondant = points_cles[indices_tries[1]].pt
            distance_entre_points = np.linalg.norm(np.array(point_actuel) - np.array(meilleur_correspondant))
            
            if distance_entre_points > 10:
                points_apparies_1.append(point_actuel)
                points_apparies_2.append(meilleur_correspondant)

    # Dessin des correspondances sur une copie de l'image
    image_correspondances = np.copy(image_ndg)
    image_correspondances = cv2.cvtColor(image_correspondances, cv2.COLOR_GRAY2BGR)
        
    for i in range(len(points_apparies_1)):
        cv2.circle(image_correspondances, (int(points_apparies_1[i][0]), int(points_apparies_1[i][1])), 5, (255, 0, 0), 1)
        cv2.circle(image_correspondances, (int(points_apparies_2[i][0]), int(points_apparies_2[i][1])), 5, (0, 255, 0), 1)
        cv2.line(image_correspondances, (int(points_apparies_1[i][0]), int(points_apparies_1[i][1])), (int(points_apparies_2[i][0]), int(points_apparies_2[i][1])), (0, 0, 255), 1)
    
    image_correspondances_path = "image_correspondance.png"
    cv2.imwrite(image_correspondances_path, image_correspondances)

    pts_src = np.float32(points_apparies_1)
    pts_dst = np.float32(points_apparies_2)

    if(len(pts_src) < 3 or len(pts_dst) < 3):
        print("Pas assez de points pour RANSAC")
        return None, None
    
    # Utilisation de RANSAC pour estimer la transformation
    _, inliers = ransac((pts_src, pts_dst), AffineTransform, min_samples=3, residual_threshold=2, max_trials=1000)
    
    # Création d'un masque noir de la même taille que l'image originale
    masque_falsifications = np.zeros_like(image)
    
    # Pour chaque paire d'inliers, copier les zones de l'image originale vers le masque noir
    taille_rect = 10  # Taille du demi-côté du rectangle
    for i in range(len(pts_src)):
        if inliers[i]: 
            pt_src = pts_src[i]
            pt_dst = pts_dst[i]
            
            # Déterminer les coordonnées du rectangle pour la source et la destination
            x1_src, y1_src = int(max(pt_src[0] - taille_rect, 0)), int(max(pt_src[1] - taille_rect, 0))
            x2_src, y2_src = int(min(pt_src[0] + taille_rect, image.shape[1])), int(min(pt_src[1] + taille_rect, image.shape[0]))
            
            x1_dst, y1_dst = int(max(pt_dst[0] - taille_rect, 0)), int(max(pt_dst[1] - taille_rect, 0))
            x2_dst, y2_dst = int(min(pt_dst[0] + taille_rect, image.shape[1])), int(min(pt_dst[1] + taille_rect, image.shape[0]))
            
            # Copier les régions de l'image originale vers le masque
            masque_falsifications[y1_src:y2_src, x1_src:x2_src] = image[y1_src:y2_src, x1_src:x2_src]
            masque_falsifications[y1_dst:y2_dst, x1_dst:x2_dst] = image[y1_dst:y2_dst, x1_dst:x2_dst]
    
    # Sauvegarde du masque
    image_masque_path = "image_masque_sift.png"
    cv2.imwrite(image_masque_path,masque_falsifications)

    return image_correspondances_path,image_masque_path

def calculer_metriques(masque_verite_path, masque_prediction_path):
    masque_verite = Image.open(masque_verite_path).convert('L')
    masque_verite = np.array(masque_verite)
    masque_verite_binaire = binariser_masque(masque_verite)
    
    masque_falsifie = Image.open(masque_prediction_path).convert('L')
    masque_falsifie = np.array(masque_falsifie)
    masque_falsifie_binaire = binariser_masque(masque_falsifie)
    
    masque_verite_1d = masque_verite_binaire.flatten()
    masque_falsifie_1d = masque_falsifie_binaire.flatten()
    
    rappel = recall_score(masque_verite_1d, masque_falsifie_1d, zero_division=1)
    precision = accuracy_score(masque_verite_1d, masque_falsifie_1d)
    f1 = f1_score(masque_verite_1d, masque_falsifie_1d, zero_division=1)
    jaccard = jaccard_score(masque_verite_1d, masque_falsifie_1d,zero_division=1) 

    return rappel,precision,f1,jaccard

def dct2d(block):
    return cv2.dct(np.float32(block))

# Détection de falsification par splicing/removal
def detection_dct(image_path, taille_bloc):
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    img_ycrcb = cv2.cvtColor(image_np, cv2.COLOR_RGB2YCrCb)
    
    img_detection = np.zeros_like(img_ycrcb[:, :, 0], dtype=np.float32)
    
    hauteur, largeur, _ = img_ycrcb.shape
    nb_bloc_hauteur = hauteur // taille_bloc
    nb_bloc_largeur = largeur // taille_bloc

    high_freqs = []

    for i in range(nb_bloc_hauteur):
        for j in range(nb_bloc_largeur):
            bloc = img_ycrcb[i*taille_bloc:(i+1)*taille_bloc, j*taille_bloc:(j+1)*taille_bloc, 0]

            dct_result = dct2d(bloc)
            high_freqs.append(dct_result[taille_bloc//2:, taille_bloc//2:].flatten())

            dct_haute_freq = dct_result[taille_bloc//2:, taille_bloc//2:] # On filtre les hautes fréquences

            # Ajuster la taille pour correspondre aux blocs
            resize = cv2.resize(np.abs(dct_haute_freq), (taille_bloc, taille_bloc))

            img_detection[i*taille_bloc:(i+1)*taille_bloc, j*taille_bloc:(j+1)*taille_bloc] += resize

    high_freqs = np.concatenate(high_freqs)
    mean = np.mean(high_freqs)
    std = np.std(high_freqs)

    #seuil = mean + 2 * std

    """
    match taille_bloc:
        case 4:
            seuil = 0.8

        case 8:
            seuil = 1.0
        
        case 16:
            seuil = 1.2

        case 32:
            seuil = 1.6
        case _:
            seuil = mean + 2 * std
    """

    seuil = 0.8

    img_detection = img_detection > seuil
    
    masque = image_np.copy()
    masque[img_detection] = [255, 0, 0]
    
    masque_path = "image_detection_dct.png"
    Image.fromarray(masque).save(masque_path)
    
    return masque_path

def creer_masque_dct(image_detection_dct_path):
    image_detection_dct = Image.open(image_detection_dct_path)
    image_detection_dct = np.array(image_detection_dct)
    masque = (image_detection_dct[:, :, 0] == 255) & (image_detection_dct[:, :, 1] == 0) & (image_detection_dct[:, :, 2] == 0)
    masque = masque.astype(np.uint8)
    kernel = np.ones((3,3),np.uint8)
    masque_erode = cv2.erode(masque, kernel, iterations=1)
    masque_dilate = cv2.dilate(masque_erode, kernel, iterations=2)
    masque_fermeture = cv2.morphologyEx(masque_dilate, cv2.MORPH_CLOSE, kernel)
    masque_final_path = "masque_dct.png"
    Image.fromarray(masque_fermeture * 255).save(masque_final_path)
    return masque_final_path

def tracer_histogramme(caracteristiques, titre='Histogramme des caracteristiques', nb_bins=50):
    plt.figure(figsize=(10, 6))
    plt.hist(caracteristiques, bins=nb_bins, color='blue', alpha=0.7)
    plt.title(titre)
    plt.xlabel('Valeur des caracteristiques')
    plt.ylabel('Fréquence')
    plt.show()

def marquer_falsifications(image_path, suspects, taille_bloc, pas):
    image = cv2.imread(image_path)
    hauteur, largeur, _ = image.shape
    
    for idx, suspect in enumerate(suspects):
        if suspect:  # Si le bloc est marqué comme suspect
            i = idx * pas // largeur * taille_bloc
            j = (idx * pas) % largeur
            cv2.rectangle(image, (j, i), (j + taille_bloc, i + taille_bloc), (0, 0, 255), 2)
    
    cv2.imwrite("sus.png", image)

def resize_and_pad_image(image, target_width=384, target_height=256):
    h, w = image.shape[:2]
    
    if w <= target_width and h <= target_height:
        return pad_image(image, target_width, target_height)
    
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return pad_image(resized_image, target_width, target_height)

def pad_image(image, target_width, target_height):
    h, w = image.shape[:2]
    top = (target_height - h) // 2
    bottom = target_height - h - top
    left = (target_width - w) // 2
    right = target_width - w - left
    return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

def extraction_caracteristiques(image_path, taille_bloc=16, pas=16, n_points=8, rayon=1):
    image = cv2.imread(image_path)
    if image is None:
        return None 

    image = resize_and_pad_image(image)

    # Convert to YCbCr color space
    ycbcr_image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
    cr_cb = ycbcr_image[:, :, 1:]
    hauteur, largeur, _ = cr_cb.shape
    blocs = []

    # Generate blocks
    for i in range(0, hauteur - taille_bloc + 1, pas):
        for j in range(0, largeur - taille_bloc + 1, pas):
            blocs.append(cr_cb[i:i + taille_bloc, j:j + taille_bloc])

    blocs = np.array(blocs)
    n_blocs = blocs.shape[0]
    caracteristiques = []

    # Process each block
    for bloc in blocs:
        for k in range(2):
            lbp = local_binary_pattern(bloc[:, :, k], n_points, rayon, method='uniform')
            dct = cv2.dct(np.float32(lbp))
            caracteristiques.append(np.std(dct))

    # Normalize features
    if len(caracteristiques) == 0:
        return None
    caracteristiques = np.array(caracteristiques).reshape(-1, 1)
    return np.array(caracteristiques).flatten()
    
def binariser_masque(masque, seuil=0):
    return (masque > seuil).astype(np.uint8)

def classifier_dossier(dossier, model, scaler):
    noms_images = [os.path.join(dossier, f) for f in os.listdir(dossier) if os.path.isfile(os.path.join(dossier, f))]
    total_images = len(noms_images)
    bonnes_reponses = 0
    
    for image_path in noms_images:
        caracteristiques = preparer_image(image_path)
        
        if caracteristiques is None or len(caracteristiques) == 0:
            print(f"Impossible d'extraire les caractéristiques de l'image {image_path}")
            continue
        caracteristiques = scaler.transform([caracteristiques])
        prediction = model.predict(caracteristiques)
        if prediction[0] == 1:  # 1 signifie "falsifiée" dans votre cas
            bonnes_reponses += 1
    
    return bonnes_reponses, total_images

def preparer_image(image_path):
    return extraction_caracteristiques(image_path)

def classifier_image(image_path, model, scaler):
    caracteristiques = extraction_caracteristiques(image_path)
    caracteristiques = scaler.transform([caracteristiques])
    prediction = model.predict(caracteristiques)
    return prediction[0]

def detection_svm(image_path):
    svm = load("svm.joblib")
    scaler = load("scaler.joblib")
    prediction = classifier_image(image_path,svm,scaler)
    return prediction

def calculer_metriques_moyennes_sift():
    dossier_images = "../data/copy-move/images"
    dossier_masques = "../data/copy-move/masks"
    dataset = []
    for image_filename in os.listdir(dossier_images):
        if image_filename.endswith("_F.png"):
            image_path = os.path.join(dossier_images, image_filename)
            mask_filename = image_filename.replace("_F.png", "_M.png")
            mask_path = os.path.join(dossier_masques, mask_filename)
            if os.path.exists(mask_path):
                dataset.append((image_path, mask_path))
            else:
                print(f"Le masque correspondant à {image_filename} n'existe pas.")
    print(f"Total de paires image-masque trouvées : {len(dataset)}")
    #print(dataset)
    precisions, rappels, f1_scores, jaccards = [], [], [], []
    for image_path, truth_mask_path in dataset:
        predicted_mask = copy_move_detection(image_path)[1]
        if(predicted_mask == None):
            continue
        rappel, precision, f1, jaccard = calculer_metriques(truth_mask_path, predicted_mask)
        
        precisions.append(precision)
        rappels.append(rappel)
        f1_scores.append(f1)
        jaccards.append(jaccard)

    precision_moyenne = np.mean(precisions)
    rappel_moyen = np.mean(rappels)
    f1_moyen = np.mean(f1_scores)
    jaccard_moyen = np.mean(jaccards)

    return rappel_moyen, precision_moyenne, f1_moyen, jaccard_moyen

def calculer_metriques_moyennes_dct(taille_bloc):
    dossier_images = "../data/splicing/images"
    dossier_masques = "../data/splicing/masks"
    dataset = []
    for image_filename in os.listdir(dossier_images):
            image_path = os.path.join(dossier_images, image_filename)
            mask_filename = image_filename.replace(".jpg",".png")
            mask_path = os.path.join(dossier_masques, mask_filename)
            if os.path.exists(mask_path):
                dataset.append((image_path, mask_path))
            else:
                print(f"Le masque correspondant à {image_filename} n'existe pas.")
    print(f"Total de paires image-masque trouvées : {len(dataset)}")
    #print(dataset)
    precisions, rappels, f1_scores, jaccards = [], [], [], []
    for image_path, truth_mask_path in dataset[:5]:
        image_detection_path = detection_dct(image_path, taille_bloc)
        creer_masque_dct(image_detection_path)
        rappel, precision, f1, jaccard = calculer_metriques(truth_mask_path, "masque_dct.png")
        
        precisions.append(precision)
        rappels.append(rappel)
        f1_scores.append(f1)
        jaccards.append(jaccard)

    precision_moyenne = np.mean(precisions)
    rappel_moyen = np.mean(rappels)
    f1_moyen = np.mean(f1_scores)
    jaccard_moyen = np.mean(jaccards)

    return rappel_moyen, precision_moyenne, f1_moyen, jaccard_moyen

if __name__ == "__main__":
    """
    svm = load("svm.joblib")
    scaler = load("scaler.joblib")
    dossier = '../CASIA2.0/Au'
    bonnes_reponses, total_images = classifier_dossier(dossier, svm, scaler)
    taux_de_bonnes_reponses = bonnes_reponses / total_images if total_images > 0 else 0
    print(f"Taux de bonnes réponses : {taux_de_bonnes_reponses:.2f} ({bonnes_reponses}/{total_images})")
    """
    rap, pre, f1, jac = calculer_metriques_moyennes_dct(4)
    print("Rappel : " + str(rap),"Precision : " + str(pre), "F1 : " + str(f1), "Jaccard :" + str(jac))


# Ancien code, peut-être encore utile

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
"""