# 🧮 FORMULE MATHÉMATIQUE DU SCORE D'ANOMALIE

## 📐 LA FORMULE EXACTE

### **Pour le One-Class SVM (ton modèle)**

Le score d'anomalie est calculé par la fonction **`decision_function()`** du SVM :

```
score(x) = w^T · φ(x) - ρ
```

**Où :**
- `x` = Tes données (les 350 features après feature engineering)
- `w` = Vecteur de poids du modèle (normal à l'hyperplan)
- `φ(x)` = Transformation kernel (RBF dans notre cas)
- `ρ` (rho) = Décalage de l'hyperplan (offset)
- `^T` = Transposée du vecteur

---

## 🎯 EXPLICATION SIMPLE

### **Version simplifiée :**
```
Score = Distance signée de x à l'hyperplan de décision

Si Score > 0  → x est du côté "NORMAL" ✅
Si Score < 0  → x est du côté "ANOMALIE" ❌
Si Score = 0  → x est exactement sur la frontière
```

---

## 🔬 FORMULE DÉTAILLÉE AVEC RBF KERNEL

Notre modèle utilise un **kernel RBF (Radial Basis Function)**, donc :

### **Étape 1 : Kernel RBF**
```
φ(x) = Kernel RBF

K(x, x_i) = exp(-γ · ||x - x_i||²)
```

**Où :**
- `x` = Nouvelle observation (tes données)
- `x_i` = Vecteurs de support du modèle (exemples d'entraînement)
- `γ` (gamma) = Paramètre du kernel (contrôle la "portée")
- `||x - x_i||²` = Distance euclidienne au carré

### **Étape 2 : Calcul du score**
```
score(x) = Σ α_i · K(x, x_i) - ρ
           i=1 à n_support_vectors
```

**Où :**
- `α_i` = Coefficients des vecteurs de support (appris pendant l'entraînement)
- `K(x, x_i)` = Similarité kernel entre x et chaque vecteur de support
- `ρ` = Seuil de décision (offset)
- La somme est sur tous les **vecteurs de support** du modèle

---

## 💻 CALCUL DANS LE CODE

### **En Python avec scikit-learn :**

```python
from sklearn.svm import OneClassSVM
import numpy as np

# 1. Modèle entraîné (déjà fait)
model = OneClassSVM(kernel='rbf', gamma=0.01, nu=0.1)
model.fit(X_train)  # X_train = données d'entraînement normales

# 2. Prédiction binaire
prediction = model.predict(X_new)
# Retourne : +1 (normal) ou -1 (anomalie)

# 3. CALCUL DU SCORE D'ANOMALIE
anomaly_score = model.decision_function(X_new)
# Retourne : score réel (ex: -0.75)
```

### **Ce qui se passe en interne :**

```python
# Pseudo-code simplifié
def decision_function(X_new):
    score = 0
    
    # Pour chaque vecteur de support
    for i in range(n_support_vectors):
        x_i = support_vectors[i]
        alpha_i = dual_coef[i]
        
        # Calcul du kernel RBF
        distance_squared = np.sum((X_new - x_i) ** 2)
        kernel_value = np.exp(-gamma * distance_squared)
        
        # Accumulation
        score += alpha_i * kernel_value
    
    # Soustraction du seuil
    score -= rho
    
    return score
```

---

## 🧪 EXEMPLE DE CALCUL AVEC TES DONNÉES

### **Tes données :**
```json
{
  "CPU": 0.95,
  "Mémoire": 0.92,
  "Pods": 0.88
}
```

### **Étape 1 : Feature Engineering**
```python
# 3 métriques → 350 features
features = [
    0.95,  # cpu_ratio
    0.92,  # mem_ratio
    0.88,  # pod_ratio
    0.85,  # cpu_mean_3h
    0.08,  # cpu_std_6h
    ...    # 345 autres features
]  # Total : 350 valeurs
```

### **Étape 2 : Normalisation**
```python
# Avec le scaler entraîné
X_scaled = (features - mean) / std
# Exemple : X_scaled = [2.5, 2.1, 1.8, 1.9, 0.3, ...]
```

### **Étape 3 : Calcul du kernel avec chaque vecteur de support**

Supposons que le modèle a **500 vecteurs de support** :

```python
gamma = 0.01  # Paramètre du modèle

score = 0
for i in range(500):
    # Distance euclidienne au carré
    distance² = Σ(X_scaled[j] - support_vector_i[j])²
    # Par exemple : distance² = 25.3
    
    # Kernel RBF
    K = exp(-0.01 × 25.3) = exp(-0.253) = 0.776
    
    # Coefficient alpha
    alpha_i = 0.0015  # (exemple)
    
    # Accumulation
    score += 0.0015 × 0.776 = 0.001164
```

Après avoir sommé les 500 vecteurs de support :
```python
total_sum = 0.523  # (exemple)
```

### **Étape 4 : Soustraction du seuil rho**
```python
rho = 1.273  # Valeur apprise pendant l'entraînement

score_final = 0.523 - 1.273 = -0.75
```

**Résultat : `-0.75` 🎯**

---

## 📊 PARAMÈTRES DU MODÈLE

Ces valeurs sont **apprises automatiquement** pendant l'entraînement :

| Paramètre | Description | Valeur typique |
|-----------|-------------|----------------|
| **γ (gamma)** | Portée du kernel RBF | 0.001 - 0.1 |
| **ν (nu)** | Fraction d'anomalies attendues | 0.05 - 0.15 |
| **ρ (rho)** | Seuil de décision | Appris automatiquement |
| **α_i** | Coefficients des vecteurs support | Appris automatiquement |
| **Support vectors** | Exemples clés de l'entraînement | 100 - 1000+ |

### **Dans ton modèle :**
```python
# Paramètres définis
gamma = 'scale'  # Automatique : 1 / (n_features × variance)
nu = 0.1         # 10% d'anomalies attendues

# Paramètres appris
rho = model.offset_[0]              # Ex: 1.273
dual_coef = model.dual_coef_        # Ex: array de 500 valeurs
support_vectors = model.support_vectors_  # Ex: 500 vecteurs de 350 features
```

---

## 🔢 FORMULE COMPLÈTE DÉVELOPPÉE

### **Version mathématique complète :**

$$
f(x) = \sum_{i=1}^{n_{SV}} \alpha_i \cdot \exp\left(-\gamma \sum_{j=1}^{350} (x_j - x_{i,j})^2\right) - \rho
$$

**Où :**
- $f(x)$ = Score d'anomalie (ton -0.75)
- $n_{SV}$ = Nombre de vecteurs de support (ex: 500)
- $\alpha_i$ = Coefficient du i-ème vecteur de support
- $\gamma$ = Paramètre du kernel RBF
- $x_j$ = j-ème feature de ta nouvelle observation (350 features)
- $x_{i,j}$ = j-ème feature du i-ème vecteur de support
- $\rho$ = Seuil de décision (offset)

### **Interprétation :**
```
Si f(x) > 0  →  NORMAL   (prédiction = +1)
Si f(x) < 0  →  ANOMALIE (prédiction = -1)
```

---

## 🎨 VISUALISATION DU CALCUL

```
Tes 350 features (après normalisation)
        │
        ↓
┌───────────────────────────────────────┐
│   Pour chaque vecteur de support:    │
│                                       │
│   1. Calcule distance euclidienne    │
│   2. Applique kernel RBF             │
│   3. Multiplie par coefficient α_i   │
│   4. Additionne                      │
└───────────────────────────────────────┘
        │
        ↓
    Somme totale = 0.523
        │
        ↓
    Soustrais ρ (rho) = 1.273
        │
        ↓
    Score final = -0.75 ✅
        │
        ↓
┌───────────────────────────────────────┐
│  Score < 0 → ANOMALIE DÉTECTÉE! 🔴   │
│  Magnitude élevée → CRITIQUE          │
└───────────────────────────────────────┘
```

---

## 🧠 POURQUOI CETTE FORMULE EST INTELLIGENTE ?

### **1. Le kernel RBF capture la similarité**
```
Distance petite → Kernel proche de 1 → Similaire ✅
Distance grande → Kernel proche de 0 → Différent ❌
```

### **2. Les vecteurs de support sont les "exemples clés"**
Ce sont les observations d'entraînement les plus **représentatives** du comportement normal.

### **3. Les coefficients α_i pondèrent l'importance**
Certains vecteurs de support comptent plus que d'autres.

### **4. Le seuil ρ définit la frontière**
C'est la "barre" entre normal et anormal, optimisée pendant l'entraînement.

---

## 📝 EXEMPLE NUMÉRIQUE COMPLET

### **Configuration du modèle :**
```
γ (gamma) = 0.01
ν (nu) = 0.1
ρ (rho) = 1.273
Nombre de support vectors = 500
```

### **Tes features normalisées (extrait) :**
```
X_scaled = [2.5, 2.1, 1.8, 1.9, 0.3, ..., 1.2]  # 350 valeurs
```

### **Calcul pour le premier vecteur de support :**
```
SV₁ = [0.1, 0.2, 0.15, 0.18, 0.05, ..., 0.12]
α₁ = 0.0015

Distance² = (2.5-0.1)² + (2.1-0.2)² + ... + (1.2-0.12)²
         = 5.76 + 3.61 + ... + 1.17
         = 25.3

Kernel = exp(-0.01 × 25.3) = exp(-0.253) = 0.776

Contribution = 0.0015 × 0.776 = 0.001164
```

### **Répéter pour les 500 vecteurs de support :**
```
Somme totale = 0.523
```

### **Score final :**
```
score = 0.523 - 1.273 = -0.75 ✅
```

---

## 🔍 VÉRIFICATION DANS LE CODE

### **Accéder aux paramètres du modèle :**

```python
import pickle

# Charger le modèle
with open('models/one_class_svm_final.pkl', 'rb') as f:
    model = pickle.load(f)

# Afficher les paramètres
print("Paramètres du modèle:")
print(f"Gamma: {model.gamma}")
print(f"Nu: {model.nu}")
print(f"Rho (offset): {model.offset_[0]}")
print(f"Nombre de support vectors: {len(model.support_vectors_)}")
print(f"Shape support vectors: {model.support_vectors_.shape}")
print(f"Coefficients duaux: {model.dual_coef_.shape}")

# Exemple de calcul manuel
X_new = [[...]]  # Tes 350 features normalisées

# Méthode 1 : Directe
score = model.decision_function(X_new)[0]
print(f"\nScore d'anomalie: {score:.4f}")

# Méthode 2 : Calcul manuel (pour comprendre)
from scipy.spatial.distance import cdist

# Distance aux support vectors
distances = cdist(X_new, model.support_vectors_, metric='euclidean')
distances_squared = distances ** 2

# Kernel RBF
kernel_values = np.exp(-model.gamma * distances_squared)

# Score
score_manual = (kernel_values @ model.dual_coef_.T).ravel() - model.offset_
print(f"Score calculé manuellement: {score_manual[0]:.4f}")
```

---

## 📚 RÉSUMÉ DE LA FORMULE

### **Version ultra-simplifiée :**
```
Score = Similarité_aux_exemples_normaux - Seuil

Si Score > 0 → Ressemble aux exemples normaux ✅
Si Score < 0 → Ne ressemble PAS aux normaux ❌
```

### **Version mathématique :**
```
score(x) = Σ[αᵢ · K(x, xᵢ)] - ρ

Avec K(x, xᵢ) = exp(-γ||x - xᵢ||²)
```

### **Pour ton cas (-0.75) :**
```
Tes données (95%, 92%, 88%)
    ↓ Feature engineering
350 features
    ↓ Normalisation
Features scaled
    ↓ Kernel RBF avec 500 support vectors
Similarité totale = 0.523
    ↓ Soustraire seuil ρ = 1.273
Score = -0.75
    ↓
ANOMALIE CRITIQUE 🔴
```

---

## 🎯 CONCLUSION

**OUI, il existe une formule mathématique précise !**

C'est une combinaison de :
1. **Distances** (entre tes données et les exemples d'entraînement)
2. **Kernel RBF** (transformation non-linéaire)
3. **Coefficients appris** (α_i et ρ)

Le modèle **ne fait PAS de magie** - c'est du **calcul mathématique pur** basé sur des formules bien définies ! 🧮

---

**Tu comprends maintenant la formule complète ! 🎉**
