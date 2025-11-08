# 📚 INDEX DE LA DOCUMENTATION

**Guide central pour naviguer dans toute la documentation du projet**

---

## 🎯 PAR OÙ COMMENCER ?

### **Si tu veux DÉMARRER RAPIDEMENT (5 min) :**
👉 **[QUICKSTART_GUIDE.md](QUICKSTART_GUIDE.md)**
- 1 seule commande à lancer
- Tests rapides inclus
- Parfait pour une première utilisation

### **Si tu veux COMPRENDRE LE PROJET :**
👉 **Lis mes explications ci-dessus dans le chat**
- Architecture complète expliquée
- Comment fonctionne le ML
- Flow de détection d'anomalie
- Feature engineering détaillé

### **Si tu veux TOUT SAVOIR sur l'exécution :**
👉 **[COMMANDES_EXECUTION.md](COMMANDES_EXECUTION.md)**
- Guide complet des commandes
- Méthode Docker complète
- Troubleshooting détaillé
- Tous les exemples de tests

### **Si tu cherches UNE COMMANDE PRÉCISE :**
👉 **[COMMANDES_EXECUTION.md](COMMANDES_EXECUTION.md)**
- Toutes les commandes essentielles
- Exemples de données
- Endpoints API
- Aide rapide

---

## 📖 DOCUMENTATION COMPLÈTE

### **1. Documentation Générale**

| Fichier | Description | Quand l'utiliser |
|---------|-------------|------------------|
| **README.md** | Vue d'ensemble du projet | Pour comprendre l'objectif global |
| **AZURE_MIGRATION_REPORT.md** | Rapport migration Azure | Pour voir les changements AWS→Azure |
| **INDEX_DOCUMENTATION.md** | Ce fichier ! | Point d'entrée de la doc |

### **2. Guides d'Exécution**

| Fichier | Niveau | Temps lecture | Utilisation |
|---------|--------|---------------|-------------|
| **QUICKSTART_GUIDE.md** | 🟢 Débutant | 5 min | Démarrage rapide |
| **COMMANDES_EXECUTION.md** | 🟡 Intermédiaire | 15 min | Guide complet des commandes |

### **3. Scripts de Test**

| Fichier | Type | Description |
|---------|------|-------------|
| **test_api.py** | Python | Tests complets de l'API |
| **test_alerts.py** | Python | Tests des alertes Alertmanager |
| **test_batch_predict.py** | Python | Tests de prédiction par batch |
| **demo_batch_process.py** | Python | Démo interactive du batch process |

### **4. Notebooks Explicatifs**

| Notebook | Phase CRISP-DM | Contenu |
|----------|----------------|---------|
| **01_business_understanding.ipynb** | Phase 1 | Contexte et objectifs business |
| **02_data_understanding.ipynb** | Phase 2 | Exploration des données |
| **03_data_preparation.ipynb** | Phase 3 | Nettoyage et préparation |
| **04_modeling.ipynb** | Phase 4 | Entraînement des modèles ML |
| **05_evaluation.ipynb** | Phase 5 | Évaluation des performances |
| **06_deployment.ipynb** | Phase 6 | Déploiement en production |

### **5. Configuration**

| Fichier | Technologie | Usage |
|---------|-------------|-------|
| **docker-compose.yml** | Docker | Orchestration des containers |
| **Dockerfile** | Docker | Image de l'API |
| **alertmanager.yml** | Alertmanager | Configuration emails |
| **requirements.txt** | Python | Dépendances |

---

## 🗺️ PARCOURS D'APPRENTISSAGE RECOMMANDÉ

### **Niveau 1 : Débutant - "Je découvre le projet"**

1. ✅ Lis le **README.md** (5 min)
2. ✅ Lis mes **explications dans le chat** (10 min)
3. ✅ Lance avec **QUICKSTART_GUIDE.md** (5 min)
4. ✅ Teste avec `.\run_and_test.ps1`

**🎯 Objectif :** Comprendre l'objectif et faire tourner le système

---

### **Niveau 2 : Intermédiaire - "Je veux maîtriser l'utilisation"**

1. ✅ Étudie **COMMANDES_EXECUTION.md** section par section
2. ✅ Lis **TEST_DATA_DOCUMENTATION.md** pour comprendre les données
3. ✅ Explore **notebooks/01** et **notebooks/02**
4. ✅ Teste différents scénarios manuellement

**🎯 Objectif :** Maîtriser toutes les fonctionnalités

---

### **Niveau 3 : Avancé - "Je veux comprendre le code"**

1. ✅ Étudie **notebooks/04_modeling.ipynb** (ML)
2. ✅ Analyse **src/feature_engineering.py**
3. ✅ Décortique **api/app.py**
4. ✅ Comprends **alertmanager.yml**
5. ✅ Explore **docker-compose.yml**

**🎯 Objectif :** Comprendre l'architecture technique

---

### **Niveau 4 : Expert - "Je veux personnaliser"**

1. ✅ Modifie les seuils d'anomalie
2. ✅ Personnalise les emails
3. ✅ Ajoute de nouveaux endpoints
4. ✅ Optimise les features
5. ✅ Déploie en production Azure

**🎯 Objectif :** Adapter le système à tes besoins

---

## 🔍 RECHERCHE PAR BESOIN

### **"Je veux lancer le projet"**
→ `docker-compose up -d --build`

### **"Je veux tester l'API"**
→ `python test_api.py`

### **"Je veux tester les alertes"**
→ `python test_alerts.py`

### **"Je veux comprendre le ML"**
→ Lis mes explications + `notebooks/04_modeling.ipynb`

### **"Je veux comprendre les données de test"**
→ Lis `TEST_DATA_DOCUMENTATION.md`

### **"Je veux voir les logs"**
→ `docker logs aws-anomaly-detection-api --follow`

### **"Je veux changer les emails"**
→ Édite `alertmanager.yml` ligne `to: '...'`

### **"L'API ne marche pas"**
→ **GUIDE_EXECUTION.md** section "Résolution de problèmes"

### **"Je veux comprendre les features"**
→ `src/feature_engineering.py` + mes explications

### **"Je veux tester une anomalie"**
→ **QUICKSTART_GUIDE.md** section "Test 2"

---

## 📊 STRUCTURE DU PROJET

```
aws_anomaly_detection_project/
│
├── 📚 DOCUMENTATION/
│   ├── README.md                    ← Vue d'ensemble
│   ├── INDEX_DOCUMENTATION.md       ← Ce fichier
│   ├── QUICKSTART_GUIDE.md          ← Démarrage rapide
│   ├── GUIDE_EXECUTION.md           ← Guide complet
│   ├── COMMANDES_REFERENCE.md       ← Référence
│   └── DELIVERY_SUMMARY.md          ← Synthèse livraison
│
├── 🤖 SCRIPTS/
│   ├── run_and_test.ps1             ← Script auto PowerShell
│   └── test_api.py                  ← Tests Python
│
├── 📓 NOTEBOOKS/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   ├── 05_evaluation.ipynb
│   └── 06_deployment.ipynb
│
├── 💻 CODE SOURCE/
│   ├── api/app.py                   ← API Flask
│   ├── src/feature_engineering.py  ← Features
│   ├── src/data_loader.py          ← Chargement données
│   └── src/utils.py                ← Utilitaires
│
├── 🧠 MODÈLES ML/
│   ├── one_class_svm_final.pkl     ← Modèle principal
│   ├── scaler.pkl                   ← Normalisation
│   ├── feature_names.pkl            ← 350 features
│   └── final_model_config.json     ← Config
│
├── 📁 DONNÉES/
│   ├── cluster_cpu_request_ratio.json
│   ├── cluster_mem_request_ratio.json
│   └── cluster_pod_ratio.json
│
└── 🐳 CONFIGURATION/
    ├── docker-compose.yml           ← Orchestration
    ├── Dockerfile                   ← Image API
    ├── alertmanager.yml             ← Emails
    └── requirements.txt             ← Dépendances
```

---

## 🎓 QUESTIONS FRÉQUENTES

### **Q : Par où commencer ?**
**R :** Lance `.\run_and_test.ps1` et lis **QUICKSTART_GUIDE.md**

### **Q : Comment tester sans Docker ?**
**R :** Voir **GUIDE_EXECUTION.md** section "MÉTHODE 2"

### **Q : Où sont les explications techniques ?**
**R :** Mes messages précédents + `notebooks/04_modeling.ipynb`

### **Q : Comment débugger ?**
**R :** `docker logs aws-anomaly-detection-api --tail 100`

### **Q : Puis-je modifier les seuils ?**
**R :** Oui, dans `api/app.py` fonction `determine_severity()`

### **Q : Comment ajouter des destinataires d'email ?**
**R :** Édite `alertmanager.yml` ligne 29 et 47

### **Q : C'est quoi CRISP-DM ?**
**R :** Méthodologie standard pour projets ML (voir notebooks/)

---

## 🚀 COMMANDES ESSENTIELLES

```powershell
# Démarrer
.\run_and_test.ps1

# Tester
python test_api.py

# Logs
docker logs aws-anomaly-detection-api --follow

# Arrêter
docker-compose down
```

---

## 📞 AIDE & SUPPORT

1. **Documentation** : Consulte les guides listés ci-dessus
2. **Logs** : `docker logs <container> --tail 50`
3. **Troubleshooting** : **GUIDE_EXECUTION.md** section dédiée
4. **Référence** : **COMMANDES_REFERENCE.md**

---

## ✅ CHECKLIST RAPIDE

- [ ] Docker Desktop installé et lancé
- [ ] PowerShell ouvert dans le projet
- [ ] Lu **QUICKSTART_GUIDE.md**
- [ ] Lancé `.\run_and_test.ps1`
- [ ] Vérifié http://localhost:5000/health
- [ ] Testé avec `python test_api.py`
- [ ] Reçu l'email d'alerte test
- [ ] Consulté http://localhost:9093

---

## 🎯 PROCHAINES ÉTAPES

1. **Maintenant** : Lance `.\run_and_test.ps1`
2. **Ensuite** : Explore les notebooks pour comprendre le ML
3. **Puis** : Personnalise selon tes besoins
4. **Enfin** : Déploie en production !

---

## 🎉 RÉCAPITULATIF

✅ **4 guides créés** pour couvrir tous les besoins
✅ **1 script automatique** pour lancer en 1 commande
✅ **Documentation complète** avec explications détaillées
✅ **Notebooks CRISP-DM** pour comprendre le ML
✅ **Troubleshooting complet** pour résoudre les problèmes

**TU AS TOUT CE QU'IL FAUT POUR RÉUSSIR ! 🚀**

---

**Dernière mise à jour :** 8 Novembre 2025
**Version :** 1.0.0
