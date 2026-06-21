# IntroML_Project
Prévision d'un type de culture par des données climatiques et chimiques.
Ce projet vise à classifier 22 types de cultures agricoles à partir de variables agronomiques (N, P, K, température, humidité, pH et précipitations). Le dataset est parfaitement équilibré avec 100 échantillons par classe, ce qui garantit des métriques fiables et sans biais de distribution.

Une analyse exploratoire avancée, notamment via t-SNE 2D, révèle une structure fortement non linéaire des données : chaque culture forme un cluster compact et bien séparé. Ces visualisations montrent que les variables agronomiques contiennent une information discriminante très élevée, justifiant l’utilisation d’un classifieur non linéaire.

Un pipeline complet StandardScaler → PCA → SVM a été optimisé avec GridSearchCV. Le meilleur modèle obtenu est un SVM à noyau RBF, avec PCA = 7 composantes, C = 1, gamma = 1. Ce modèle atteint une accuracy remarquable de ≈ 99%, confirmée par les matrices de confusion quasi parfaites.

L’ensemble des analyses démontre que la combinaison PCA + SVM-RBF est très performante, stable et robuste pour la recommandation de cultures. Le modèle généralise efficacement et sépare presque parfaitement les 22 classes.
