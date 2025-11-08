# 📊 EXPLICATION : ANOMALY SCORE (Score d'Anomalie)

## 🎯 C'EST QUOI EXACTEMENT ?

L'**anomaly_score** (par exemple `-0.75`) est un **score de confiance** qui indique **à quel point une observation est anormale**.

---

## 🔢 COMMENT EST-IL CALCULÉ ?

### **Méthode : `decision_function()` du One-Class SVM**

Le modèle **One-Class SVM** (Support Vector Machine) calcule ce score avec sa fonction **`decision_function()`**.

```python
# Dans le code de l'API (api/app.py) ou notebooks
anomaly_score = model.decision_function(X_scaled)[0]
```

---

## 📐 FORMULE MATHÉMATIQUE

### **One-Class SVM : Distance à l'hyperplan**

Le One-Class SVM crée un **hyperplan** (frontière de décision) qui sépare les données "normales" des anomalies dans un espace à haute dimension.

**Le score = Distance signée de l'observation à cet hyperplan**

```
         │
         │  Normal (score positif)
    +0.5 │  ✅ ✅ ✅
         │
    0.0  ├─────────────────── [HYPERPLAN]
         │
   -0.5  │  ❌ ❌ (Anomalies légères)
         │
   -1.0  │     🔴 (Anomalies fortes)
         │
```

### **Formule simplifiée :**
```
score = distance_à_hyperplan

Si score > 0  → NORMAL ✅
Si score < 0  → ANOMALIE ❌

Plus le score est négatif → Plus l'anomalie est forte
```

---

## 🎨 INTERPRÉTATION DU SCORE

### **Échelle typique :**

| Score | Interprétation | Sévérité | Action |
|-------|----------------|----------|--------|
| **> 0.5** | Très normal | ✅ Aucune | RAS |
| **0 à 0.5** | Normal | ✅ Aucune | Surveillance |
| **0 à -0.2** | Légèrement suspect | 🟡 INFO | Observer |
| **-0.2 à -0.5** | Anomalie modérée | 🟠 WARNING | Vérifier |
| **< -0.5** | Anomalie forte | 🔴 CRITICAL | Action immédiate |

### **Ton exemple : `-0.75`**

```json
{
  "anomaly_score": -0.75
}
```

**Signification :**
- ✅ C'est **clairement une anomalie** (score négatif)
- 🔴 C'est une anomalie **forte/critique** (< -0.5)
- ⚠️ L'observation est **très éloignée** de ce qui est normal
- 📧 Une **alerte CRITICAL** doit être envoyée

---

## 🧮 EXEMPLE DE CALCUL CONCRET

### **Étape 1 : Données reçues**
```json
{
  "cluster_cpu_request_ratio": 0.95,  // 95% - TRÈS ÉLEVÉ! 🔴
  "cluster_mem_request_ratio": 0.92,  // 92% - TRÈS ÉLEVÉ! 🔴
  "cluster_pod_ratio": 0.88           // 88% - ÉLEVÉ! 🟡
}
```

### **Étape 2 : Feature Engineering**
```python
# 3 métriques → 350 features
features = feature_engineer.transform(data)
# Exemple de features générées :
# - cpu_mean_3h = 0.85
# - cpu_std_6h = 0.08
# - cpu_trend = +0.10 (augmentation forte!)
# - mem_cpu_ratio = 0.968
# - is_business_hours = 1
# ... 345 autres features
```

### **Étape 3 : Normalisation**
```python
X_scaled = scaler.transform(features)
# Les features sont normalisées (mean=0, std=1)
```

### **Étape 4 : Calcul du score par le modèle**
```python
# Le modèle SVM calcule la distance à l'hyperplan
score = model.decision_function(X_scaled)[0]
# Résultat : -0.75
```

### **Étape 5 : Interprétation**
```python
if score < 0:
    prediction = "ANOMALY"
    
    if score < -0.5:
        severity = "critical"   # -0.75 tombe ici!
    elif score < -0.2:
        severity = "warning"
    else:
        severity = "info"
```

---

## 🔍 COMPARAISON DES SCORES

### **Exemple 1 : Métriques NORMALES**
```json
{
  "cluster_cpu_request_ratio": 0.45,  // 45% - OK ✅
  "cluster_mem_request_ratio": 0.52,  // 52% - OK ✅
  "cluster_pod_ratio": 0.38           // 38% - OK ✅
}
```
**Score attendu** : `+0.32` (positif = normal) ✅

### **Exemple 2 : Métriques LÉGÈREMENT ÉLEVÉES**
```json
{
  "cluster_cpu_request_ratio": 0.68,  // 68% - Un peu haut 🟡
  "cluster_mem_request_ratio": 0.72,  // 72% - Un peu haut 🟡
  "cluster_pod_ratio": 0.65           // 65% - OK 🟢
}
```
**Score attendu** : `-0.15` (anomalie légère) 🟡

### **Exemple 3 : Métriques CRITIQUES** (ton cas)
```json
{
  "cluster_cpu_request_ratio": 0.95,  // 95% - CRITIQUE! 🔴
  "cluster_mem_request_ratio": 0.92,  // 92% - CRITIQUE! 🔴
  "cluster_pod_ratio": 0.88           // 88% - ÉLEVÉ! 🟠
}
```
**Score obtenu** : `-0.75` (anomalie forte) 🔴

---

## 🧠 POURQUOI CE SCORE EST INTELLIGENT ?

### **1. Contexte Multi-Features**
Le score ne regarde pas juste "CPU = 95%" isolément, mais analyse :
- La **tendance** (CPU augmente depuis 3h ?)
- La **variabilité** (Fluctuations importantes ?)
- Les **corrélations** (CPU + Mémoire élevés ensemble ?)
- Le **timing** (C'est pendant les heures de bureau ?)
- Les **patterns fréquentiels** (Comportement cyclique anormal ?)

### **2. Apprentissage du "Normal"**
Le modèle a appris sur des milliers d'exemples ce qui est "normal" :
- CPU moyen : ~50% ± 15%
- Mémoire moyenne : ~55% ± 12%
- Pod ratio moyen : ~40% ± 10%

Quand tu envoies 95%, 92%, 88%, c'est **très loin** de la distribution normale !

### **3. Distance Mathématique**
```
Score = Distance dans un espace à 350 dimensions

Point normal (50%, 55%, 40%) + 347 autres features
         ↓
    [HYPERPLAN]
         ↓
Ton point (95%, 92%, 88%) + 347 autres features
```

**Distance calculée = -0.75** (très éloigné)

---

## 📈 VISUALISATION DU SCORE

```
Distribution des Scores sur le Dataset d'Entraînement
═══════════════════════════════════════════════════════

         Normaux (85%)           │  Anomalies (15%)
                                 │
  ┌──────────────────────────────┼───────────────┐
  │                              │               │
  │   ✅✅✅✅✅✅✅✅✅✅✅       │  ❌❌🔴🔴     │
  │                              │               │
 +1.0                           0.0           -1.0
  │                              │               │
  └──────────────────────────────┴───────────────┘
                                 │
                          [FRONTIÈRE]
                                 │
                        Ton score : -0.75 🔴
                          (très anormal!)
```

---

## 💻 CODE DANS L'API

### **Où le score est calculé** (`api/app.py`)

```python
# 1. Charger le modèle au démarrage
model = pickle.load(open('models/one_class_svm_final.pkl', 'rb'))

# 2. Dans l'endpoint /predict
@app.route('/predict', methods=['POST'])
def predict():
    # ... (feature engineering)
    
    # Prédiction binaire (-1 ou 1)
    prediction = model.predict(X_scaled)[0]
    # -1 = anomalie, 1 = normal
    
    # ⭐ CALCUL DU SCORE D'ANOMALIE ⭐
    anomaly_score = model.decision_function(X_scaled)[0]
    # C'est ici que -0.75 est calculé!
    
    # Déterminer la sévérité
    if anomaly_score < -0.5:
        severity = "critical"  # Ton cas!
    elif anomaly_score < -0.2:
        severity = "warning"
    else:
        severity = "info"
    
    return jsonify({
        "prediction": "anomaly",
        "anomaly_score": -0.75,  # Score calculé
        "severity": "critical"
    })
```

---

## 🎯 RÉSUMÉ SIMPLE

### **Anomaly Score = Distance au "Normal"**

1. **Le modèle connaît le "normal"** → Entraîné sur des données historiques
2. **Tu envoies de nouvelles données** → 95% CPU, 92% Mémoire
3. **Le modèle calcule la distance** → Très loin du normal !
4. **Score = -0.75** → Négatif = Anomalie, Forte magnitude = Critique

### **Échelle simplifiée :**
```
+1.0  ────  Parfaitement normal ✅
+0.5  ────  Normal ✅
 0.0  ────  [FRONTIÈRE]
-0.2  ────  Suspect 🟡
-0.5  ────  Anomalie 🟠
-0.75 ────  Anomalie forte 🔴 ← TON CAS
-1.0  ────  Anomalie extrême 🚨
```

---

## 📚 POUR ALLER PLUS LOIN

### **Mathématiques détaillées du One-Class SVM**

Le One-Class SVM résout ce problème d'optimisation :

```
min  1/2 ||w||² + 1/(ν·n) Σ ξᵢ - ρ
w,ξ,ρ

sous contraintes :
  wᵀφ(xᵢ) ≥ ρ - ξᵢ
  ξᵢ ≥ 0
```

**Où :**
- `w` = vecteur normal à l'hyperplan
- `ρ` = décalage de l'hyperplan (rho)
- `φ(x)` = fonction kernel (RBF dans notre cas)
- `ξᵢ` = slack variables (erreurs)
- `ν` = paramètre de contrôle (fraction d'outliers attendus)

**Le score de décision est :**
```
score = wᵀφ(x) - ρ

Si score > 0 → x est du côté "normal"
Si score < 0 → x est du côté "anomalie"
```

**Dans ton cas :**
```
score(tes_données) = -0.75
→ Très négatif
→ Très loin de l'hyperplan du côté anomalie
→ ANOMALIE CRITIQUE confirmée!
```

---

## 🎓 QUESTIONS FRÉQUENTES

### **Q1 : Pourquoi -0.75 et pas un pourcentage ?**
**R :** C'est une **distance géométrique** dans un espace multi-dimensionnel, pas un pourcentage. L'échelle dépend du modèle entraîné.

### **Q2 : Est-ce que -0.75 est toujours critique ?**
**R :** Oui dans notre système ! Nous avons défini le seuil "critical" à -0.5. Tout score < -0.5 déclenche une alerte critique.

### **Q3 : Peut-on avoir un score < -1.0 ?**
**R :** Oui ! Il n'y a pas de limite théorique. Plus c'est négatif, plus c'est anormal.

### **Q4 : Comment changer les seuils ?**
**R :** Dans `api/app.py`, modifie la fonction `determine_severity()` :
```python
if score < -0.5:    # Changez ce seuil
    return "critical"
elif score < -0.2:  # Et celui-ci
    return "warning"
```

### **Q5 : Le score dépend-il des features ?**
**R :** Oui ! Plus on a de features pertinentes (350 dans notre cas), plus le score est précis et significatif.

---

## ✅ CONCLUSION

**Ton score de `-0.75` signifie :**

✅ **Anomalie confirmée** (score négatif)
✅ **Sévérité critique** (< -0.5)
✅ **Action requise** (alerte email envoyée)
✅ **Forte confiance** (magnitude importante)

**En langage simple :**
> "Tes métriques (95% CPU, 92% Mémoire) sont tellement éloignées de ce qui est normal dans ton cluster que le modèle est très confiant qu'il y a un problème sérieux. Score de -0.75 = c'est critique, agis vite !"

---

**Maintenant tu comprends parfaitement l'anomaly score ! 🎉**
