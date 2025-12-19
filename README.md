# 🧬 Resource-Constrained Neural Architecture Search (NAS) for LLMs

[](https://www.python.org/)
[](https://pytorch.org/)
[](https://huggingface.co/)
[](https://www.google.com/search?q=)

> **Projet de Recherche - Algorithmes d'IA**
>
> *Comment explorer l'espace des architectures Transformers avec un budget de calcul strictement limité (GPU étudiant / Laptop) ?*

-----

## 📑 Table des Matières

1.  [Contexte et Problématique](https://www.google.com/search?q=%23-contexte-et-probl%C3%A9matique)
2.  [Méthodologie Technique](https://www.google.com/search?q=%23-m%C3%A9thodologie-technique)
3.  [Installation et Usage](https://www.google.com/search?q=%23-installation-et-usage)
4.  [Résultats Expérimentaux](https://www.google.com/search?q=%23-r%C3%A9sultats-exp%C3%A9rimentaux)
5.  [Analyse Critique & Key Insights](https://www.google.com/search?q=%23-analyse-critique--key-insights)
6.  [Bibliographie](https://www.google.com/search?q=%23-bibliographie)
7.  [Auteurs](https://www.google.com/search?q=%23-auteurs)

-----

## 🎯 Contexte et Problématique

L'entraînement des modèles de langage (LLMs) et la recherche de leur architecture optimale (**NAS**) nécessitent généralement des milliers d'heures de GPU (ex: *NASNet*, *AmoebaNet*). Cette barrière technologique limite la recherche aux grandes entreprises (Google, OpenAI).

**Notre objectif :** Démocratiser le NAS en implémentant une stratégie d'**optimisation sous contrainte de ressources**. Nous cherchons à identifier les meilleures architectures de type **DistilBERT** en utilisant des "Proxy Tasks" (tâches intermédiaires) ultra-rapides, simulant un environnement à budget computationnel faible.

-----

## 🛠 Méthodologie Technique

Nous avons développé un **Algorithme Génétique** (Evolutionary Algorithm) capable de faire évoluer une population de Transformers.

### 1\. Espace de Recherche (Search Space)

Le génome de nos modèles est défini par les hyperparamètres suivants :

  * `num_hidden_layers`: Profondeur du réseau [2, 4, 6]
  * `hidden_size`: Largeur des couches [256, 512, 768]
  * `num_attention_heads`: Nombre de têtes d'attention [4, 8, 12]

> **Sécurité Mathématique :** Une fonction `_ensure_validity()` garantit que chaque architecture générée respecte la contrainte $d_{model} \% n_{heads} == 0$.

### 2\. Le "Budget Hack" : Proxy Task Evaluation

Au lieu d'un entraînement complet, nous utilisons une stratégie d'**Estimation Basse Fidélité** (inspirée par *DistilBERT* et *LEMONADE*) :

  * **Dataset :** GLUE/SST-2 (sous-échantillonné à 1000 exemples).
  * **Entraînement :** Les modèles sont entraînés *from scratch* (poids aléatoires) pendant seulement **50 à 400 steps**.
  * **Hypothèse :** La vitesse d'apprentissage (Learning Speed) dans les premiers instants est corrélée à la performance finale.

### 3\. Moteur Évolutif

  * **Sélection :** Ranking basé sur l'accuracy de validation.
  * **Reproduction :** Stratégie d'élitisme (Top 50% conservé) + Mutations aléatoires sur les enfants.

-----

## 💻 Installation et Usage

### Pré-requis

  * Python 3.10 ou 3.11 (Recommandé pour compatibilité PyTorch)
  * Bibliothèques : `transformers`, `datasets`, `torch`, `scikit-learn`, `matplotlib`

<!-- end list -->

```bash
# Cloner le repo
git clone https://github.com/Leandredt/AI-algorithms-project.git
cd nas-distilbert-project

# Installer les dépendances
pip install -r requirements.txt
# Note : Si vous avez des erreurs Numpy, utilisez : pip install "numpy<2.0"
```

### Lancer l'expérience

Ouvrez le notebook `AI_Algorithms_project.ipynb` ou exécutez le script principal.
Vous pouvez choisir entre deux modes :

  * `mode="TOY"` : Simulation mathématique instantanée (pour tester l'algo).
  * `mode="REAL"` : Entraînement réel des réseaux de neurones.

<!-- end list -->

```python
# Exemple d'appel dans le code
best_model, history = run_evolution(generations=4, population_size=5, mode="REAL")
```

-----

## 📊 Résultats Expérimentaux

Nous avons mené deux expériences majeures pour valider notre approche.

### Expérience A : Budget Ultra-Faible (50 Steps)

  * **Observation :** Les **petits modèles** (9M paramètres) ont dominé (52.2% acc) tandis que les gros modèles (46M) ont échoué (47.0% acc).
  * **Interprétation :** Les gros modèles souffrent d'inertie. Ils n'ont pas eu assez de pas de gradient pour s'adapter ("Warm-up phase").

### Expérience B : Budget Moyen (400 Steps)

  * **Observation :** Avec un budget plus raisonnable, la tendance s'inverse. Le modèle à **48M paramètres** atteint **72.2%** d'accuracy.
  * **Convergence :** L'algorithme a rapidement convergé (dès la Génération 2) vers les modèles à plus forte capacité, saturant l'espace de recherche.

| Génération | Modèle (Params) | Accuracy (400 steps) |
| :--- | :--- | :--- |
| **Gen 1 (Random)** | 30M - 40M (Mixte) | 68.6% - 70.8% |
| **Gen 2 (Optimized)** | 48.07M (Large) | **72.2%** |
| **Gen 3 (Converged)** | 48.07M (Large) | 71.8% (Stable) |

-----

## 🧠 Analyse Critique & Key Insights

Ce projet a permis de mettre en lumière des phénomènes cruciaux pour le NAS :

### 1\. Le Paradoxe du Sous-Apprentissage ("Under-training Paradox")

Nos résultats démontrent que la performance d'une architecture est relative au budget d'entraînement.

> *Sur un sprint (50 steps), une Twingo bat une Ferrari qui n'a pas le temps de passer la seconde.*
> *Sur une course (400 steps), la puissance brute l'emporte.*

### 2\. Calibrage de la Proxy Task

Pour que le NAS soit efficace industriellement, le "Proxy" doit maintenir la **corrélation de rang** (Rank Correlation). Si le budget est trop faible, le classement est inversé (les mauvais modèles paraissent bons). Nous avons identifié que \~400 steps est le seuil minimal pour notre espace de recherche.

### 3\. Efficacité de l'Évolution

L'algorithme génétique a prouvé sa capacité à naviguer le **Front de Pareto**. Il a su :

1.  Éliminer les modèles "moyens" inefficaces.
2.  Identifier et propager les "gènes" performants (ex: hidden\_size=768) à travers les générations.

-----

## 📚 Bibliographie

Ce travail s'appuie sur l'analyse critique des papiers suivants :

1.  **NASNet** (Zoph et al.) - *Reinforcement Learning for NAS.*
2.  **AmoebaNet** (Real et al.) - *Regularized Evolution for Image Classifier Architecture Search.*
3.  **LEMONADE** (Elsken et al.) - *Multi-objective Evolutionary Algorithms.*
4.  **DistilBERT** (Sanh et al.) - *Distilling the knowledge in a neural network.*

-----

## 👥 Auteurs

**Groupe :**

  * Léandre DURAND-TERRASSON (Implémentation Technique)
  * Marwan HEMANI (Analyse & Recherche)
  * Geoffroy-Junior GANKOUE-DZON (Analyse & Recherche)

-----




