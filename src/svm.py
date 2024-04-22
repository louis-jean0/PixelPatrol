import os
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from detection import extraction_caracteristiques
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from joblib import dump

def lire_liste(path):
    with open(path, 'r') as fichier:
        noms_images = fichier.read().splitlines()
    return noms_images

def charger_images(dossier, noms_images, label):
    features = []
    labels = []
    for nom_image in noms_images:
        image_path = os.path.join(dossier, nom_image)
        image_caracteristiques = extraction_caracteristiques(image_path)
        if image_caracteristiques is not None:
            features.append(image_caracteristiques)
            labels.append(label)
        else:
            print(f"Erreur ou données manquantes pour l'image {nom_image}")
    return features, labels

def sous_echantillonner_liste(source_path, target_path, nombre_cible):
    # Lecture des noms de fichiers
    with open(source_path, 'r') as fichier:
        noms_images = fichier.read().splitlines()
    
    # Mélanger la liste des noms d'images
    random.shuffle(noms_images)
    
    # Réduire la liste au nombre cible
    noms_images_reduits = noms_images[:nombre_cible]
    
    # Sauvegarder la nouvelle liste dans un fichier
    with open(target_path, 'w') as fichier:
        for nom in noms_images_reduits:
            fichier.write(nom + '\n')

# Chemins vers les dossiers contenant les images authentiques et falsifiées
dossier_authentiques = "../CASIA2.0/Au"
dossier_falsifiees = "../CASIA2.0/Tp"

nombre_images_falsifiees = 5123
sous_echantillonner_liste("../CASIA2.0/au_list.txt", "../CASIA2.0/au_list_reduced.txt", nombre_images_falsifiees)

# Lecture des noms de fichiers
noms_authentiques = lire_liste("../CASIA2.0/au_list_reduced.txt")
noms_falsifiees = lire_liste("../CASIA2.0/tp_list.txt")

print(len(noms_falsifiees))
print(len(noms_authentiques))

random.shuffle(noms_authentiques)
random.shuffle(noms_falsifiees)

# Chargement des caractéristiques
features_authentiques, labels_authentiques = charger_images(dossier_authentiques, noms_authentiques, 0)
features_falsifiees, labels_falsifiees = charger_images(dossier_falsifiees, noms_falsifiees, 1)

# Concaténation des caractéristiques et des labels
features = np.vstack([features_authentiques, features_falsifiees])
scaler = StandardScaler()
scaler.fit(features)
features = scaler.transform(features)
labels = np.array(labels_authentiques + labels_falsifiees)

# Division en données d'entraînement et de test
X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Vérification de la présence de NaN et nettoyage
if np.isnan(X_train).any():
    print("Des NaN sont présents dans les données.")
    mask = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[mask]
    Y_train = Y_train[mask]


# Entraînement du SVM
"""
param_grid = {
    'C': [2**i for i in range(-5, 5)],
    'gamma': [2**i for i in range(-5, 5)]
}
svm = SVC(kernel="rbf")
grid_search = GridSearchCV(svm, param_grid, cv=3, scoring='accuracy', verbose=2, n_jobs=-1)
grid_search.fit(X_train, Y_train)
print("Meilleurs paramètres:", grid_search.best_params_)
print("Meilleure précision obtenue:", grid_search.best_score_)
best_model = grid_search.best_estimator_
"""
svm = SVC(kernel="rbf", gamma="scale")
svm.fit(X_train, Y_train)

# Prédiction et évaluation
Y_pred = svm.predict(X_test)
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))

# Sauvegarde du modèle
dump(svm, "svm.joblib")
dump(scaler, "scaler.joblib")