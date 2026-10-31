# Customer_Churn_Prediction
# Prédiction du Churn Client (Télécoms)

Projet de Machine Learning supervisé visant à prédire si un client va quitter son opérateur, afin de permettre des actions de rétention ciblées.

## Objectif

Le churn (départ des clients) coûte cher aux entreprises de services : acquérir un nouveau client est plus coûteux que conserver un client existant. L'objectif de ce projet est de :

- identifier les clients à risque de départ ;
- comprendre les facteurs qui expliquent ces départs ;
- fournir un modèle exploitable par une équipe marketing ou relation client.

## Dataset

- **Source :** Telco Customer Churn (Kaggle, BlastChar)
- **Taille :** 7 043 clients, 21 variables
- **Variable cible :** `Churn` (Yes = parti, No = resté)
- **Taux de churn :** 26,5 % (1 869 clients partis, 5 174 restés)
- **Variables :** profil du client (ancienneté, personnes à charge...), services souscrits (internet, téléphone, sécurité en ligne, streaming...), contrat, mode de facturation et de paiement, montants facturés.

Le jeu de données est déséquilibré : environ 3 clients sur 4 restent. Cela impacte le choix des métriques d'évaluation.

## Démarche

1. **Exploration** : dimensions, types de variables, répartition de la cible.
2. **Nettoyage** : `TotalCharges` était lue comme du texte. Après conversion, 11 valeurs manquantes sont apparues. Elles correspondent à des clients avec une ancienneté de 0 mois (nouveaux clients, pas encore facturés) et ont été remplacées par 0. L'identifiant `customerID` a été supprimé.
3. **Encodage** : transformation des variables catégorielles en variables 0/1 (`get_dummies`), passant de 19 à 30 variables explicatives.
4. **Séparation train / test** : 80 % / 20 %, stratifiée pour conserver 26,5 % de churn dans chaque ensemble.
5. **Standardisation** de `tenure`, `MonthlyCharges` et `TotalCharges`, calculée sur le jeu d'entraînement uniquement pour éviter toute fuite de données (data leakage).
6. **Comparaison de 3 modèles** : régression logistique, arbre de décision, Random Forest.
7. **Évaluation** : précision, rappel, F1-score, matrice de confusion, validation croisée.
8. **Amélioration** : pondération des classes, ajustement du seuil de décision, optimisation par GridSearchCV.
9. **Interprétation** des coefficients du modèle final.

## Résultats

### Comparaison des modèles (accuracy sur le jeu de test)

| Modèle | Accuracy test | Accuracy train |
|---|---|---|
| Baseline (prédire que personne ne part) | 73,5 % | — |
| Régression logistique | 80,6 % | — |
| Arbre de décision | 72,5 % | 99,8 % |
| Random Forest | 79,0 % | 99,8 % |

L'arbre de décision et le Random Forest montrent un fort **surapprentissage** (près de 100 % sur le train contre 72,5 % et 79,0 % sur le test). La régression logistique, plus simple, généralise mieux.

### Pourquoi l'accuracy ne suffit pas

La baseline, qui ne détecte aucun départ, atteint déjà 73,5 %. Le rappel sur la classe « Part » est donc la métrique la plus importante : il mesure la part des clients réellement partants que le modèle parvient à détecter.

| Modèle | Précision (Part) | Rappel (Part) | F1 (Part) |
|---|---|---|---|
| Régression logistique | 0,66 | 0,56 | 0,60 |
| Random Forest | 0,63 | 0,51 | 0,56 |
| Régression logistique pondérée | 0,50 | 0,78 | 0,61 |
| **Modèle final (GridSearchCV)** | **0,51** | **0,78** | **0,62** |

### Effet du seuil de décision (régression logistique)

| Seuil | Précision | Rappel | F1 |
|---|---|---|---|
| 0,5 | 0,659 | 0,559 | 0,605 |
| 0,4 | 0,570 | 0,666 | 0,614 |
| 0,3 | 0,520 | 0,757 | 0,617 |
| 0,2 | 0,467 | 0,856 | 0,604 |

Plus le seuil baisse, plus le modèle détecte de clients partants, mais au prix de davantage de fausses alertes.

### Modèle final

- **Paramètres retenus** : régression logistique, `C = 0,1`, `class_weight = 'balanced'`
- **F1-score en validation croisée** : 0,634
- **Sur le jeu de test (1 409 clients, dont 374 partants)** :

| Métrique (classe « Part ») | Valeur |
|---|---|
| Rappel | 78,3 % |
| Précision | 51,0 % |
| F1-score | 61,8 % |
| Accuracy globale | 74 % |

Le modèle détecte **293 des 374 clients partants**. En contrepartie, environ un client sur deux signalé comme « à risque » ne partirait finalement pas.

### Choix assumé : rappel contre précision

L'accuracy globale du modèle final (74 %) est proche de la baseline, et inférieure à celle de la régression logistique standard (80,6 %). Ce choix est volontaire : en rétention client, rater un départ coûte généralement plus cher qu'envoyer une offre à un client qui serait resté. Le seuil peut être ajusté selon le coût réel d'une action de rétention.

## Facteurs de départ identifiés

D'après les coefficients du modèle final :

**Augmentent le risque de départ**
- Service internet en fibre optique
- Paiement par chèque électronique
- Facturation dématérialisée (paperless billing)
- Options de streaming (TV, films)

**Réduisent le risque de départ**
- Contrat de 2 ans, puis contrat d'1 an (par rapport au contrat mensuel)
- Ancienneté élevée (`tenure`)
- Services de sécurité en ligne et de support technique
- Présence de personnes à charge

Ces résultats montrent des **associations statistiques** et non des relations de cause à effet. Ils suggèrent par exemple de proposer des contrats longs aux nouveaux clients et de mettre en avant les services de support.

## Limites et pistes d'amélioration

- Le modèle repose sur une seule période et un seul jeu de données.
- Le modèle final est volontairement orienté vers le rappel, au détriment de la précision.
- Pistes : tester des modèles de boosting (XGBoost, LightGBM), traiter le déséquilibre par suréchantillonnage (SMOTE), intégrer un coût de rétention pour choisir le seuil optimal, déployer le modèle via une API.

## Outils

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Joblib, Jupyter Notebook

## Fichiers du dépôt

- `Churn_Prediction.ipynb` : notebook complet (analyse, modélisation, évaluation)
- `modele_churn.pkl` : modèle final sauvegardé
- `README.md` : présentation du projet

## Auteur

**ALHADJ OUSMANE ABDELSALAM**
Étudiant en Management et Gouvernance des Systèmes d'Information, ENSA Khouribga
