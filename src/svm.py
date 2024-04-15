import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from detection import extraction_caracteristiques
from sklearn.metrics import classification_report, confusion_matrix
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
        features.append(image_caracteristiques)
        labels.append(label)
    return features,labels

dossier_authentiques = "../CASIA2.0/Au"
dossier_falsifiees = "../CASIA2.0/Tp"

noms_authentiques = lire_liste("../CASIA2.0/au_list.txt")
noms_falsifiees = lire_liste("../CASIA2.0/tp_list.txt")

features_authentiques, labels_authentiques = charger_images(dossier_authentiques,noms_authentiques,0)
features_falsifiees, labels_falsifiees = charger_images(dossier_falsifiees,noms_falsifiees,1)

features = np.array(features_authentiques + features_falsifiees)
labels = np.array(labels_authentiques + labels_falsifiees)

X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
svm = SVC(kernel="rbf", gamma="scale")
svm.fit(X_train, Y_train)

Y_pred = svm.predict(X_test)
print(confusion_matrix(Y_test,Y_pred))
print(classification_report(Y_test,Y_pred))
dump(svm, "svm.joblib")