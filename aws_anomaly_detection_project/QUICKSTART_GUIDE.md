# ⚡ QUICK START - GUIDE RAPIDE

**Lancer et tester le projet en 5 minutes chrono !** ⏱️

---

## 🚀 MÉTHODE ULTRA-RAPIDE (Recommandée)

### **1 seule commande à lancer ! 🎯**

```powershell
# Ouvre PowerShell dans le dossier du projet
cd C:\Users\acer\Desktop\yassmine\aws_anomaly_detection_project

# Lance le script automatique
.\run_and_test.ps1
```

**C'EST TOUT ! 🎉**

Le script fait automatiquement :
- ✅ Vérifie Docker
- ✅ Lance les containers
- ✅ Attend le démarrage
- ✅ Teste l'API
- ✅ Envoie une anomalie test
- ✅ Tu reçois un email !

---

## 📝 MÉTHODE MANUELLE (Étape par étape)

### **Étape 1 : Ouvrir PowerShell**
```powershell
cd C:\Users\acer\Desktop\yassmine\aws_anomaly_detection_project
```

### **Étape 2 : Lancer Docker**
```powershell
docker-compose up -d --build
```
⏱️ Attends 30 secondes

### **Étape 3 : Vérifier**
```powershell
# Doit afficher 2 containers
docker ps
```

### **Étape 4 : Tester**
```powershell
# Test simple
curl http://localhost:5000/health

# OU test complet
python test_api.py
```

---

## 🧪 TESTER UNE ANOMALIE RAPIDEMENT

```powershell
# Copie-colle cette commande
$body = '{"cluster_cpu_request_ratio": 0.95, "cluster_mem_request_ratio": 0.92, "cluster_pod_ratio": 0.88}'
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -Body $body -ContentType "application/json" | ConvertTo-Json
```

**📧 Tu dois recevoir un EMAIL d'alerte !**

---

## 🌐 INTERFACES WEB

Ouvre dans ton navigateur :

| Interface | URL | Description |
|-----------|-----|-------------|
| **API** | http://localhost:5000/health | État de l'API |
| **Alertmanager** | http://localhost:9093 | Interface alertes |

---

## 📊 EXEMPLES DE TESTS

### **Test 1 : Métriques NORMALES** ✅
```powershell
$normal = '{"cluster_cpu_request_ratio": 0.45, "cluster_mem_request_ratio": 0.52, "cluster_pod_ratio": 0.38}'
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -Body $normal -ContentType "application/json"
```
**Résultat attendu :** `"prediction": "normal"` - Pas d'email

### **Test 2 : ANOMALIE CRITIQUE** 🔴
```powershell
$anomaly = '{"cluster_cpu_request_ratio": 0.95, "cluster_mem_request_ratio": 0.92, "cluster_pod_ratio": 0.88}'
Invoke-RestMethod -Uri "http://localhost:5000/predict" -Method Post -Body $anomaly -ContentType "application/json"
```
**Résultat attendu :** `"prediction": "anomaly"` - EMAIL ENVOYÉ !

### **Test 3 : Obtenir infos du modèle** ℹ️
```powershell
Invoke-RestMethod -Uri "http://localhost:5000/model/info"
```

---

## 🛑 ARRÊTER LE SYSTÈME

```powershell
# Arrêter proprement
docker-compose down

# OU tout supprimer (volumes inclus)
docker-compose down -v
```

---

## 📋 VÉRIFICATIONS RAPIDES

### ✅ Tout fonctionne si :
- [ ] `docker ps` montre 2 containers
- [ ] http://localhost:5000/health retourne `{"status": "healthy"}`
- [ ] http://localhost:9093 affiche l'interface
- [ ] Test d'anomalie envoie un email

### ❌ Problèmes courants :

**Docker ne démarre pas ?**
```powershell
# Lance Docker Desktop manuellement
# Attends qu'il soit complètement démarré (icône en bas à droite)
```

**Port 5000 occupé ?**
```powershell
# Trouve le processus
netstat -ano | findstr :5000
# Tue-le
taskkill /PID <numero_pid> /F
```

**Pas d'email reçu ?**
1. Vérifie les SPAM
2. Vérifie Alertmanager: `docker logs alertmanager`
3. Test manuel: `Invoke-RestMethod -Uri "http://localhost:5000/alert/test" -Method Post`

---

## 🔍 LOGS UTILES

```powershell
# Logs API en temps réel
docker logs aws-anomaly-detection-api --follow

# Logs Alertmanager
docker logs alertmanager --follow

# Dernières 50 lignes
docker logs aws-anomaly-detection-api --tail 50
```

---

## 📚 DOCUMENTATION COMPLÈTE

Pour plus de détails, consulte :
- **`GUIDE_EXECUTION.md`** → Guide complet avec troubleshooting
- **`README.md`** → Vue d'ensemble du projet
- **`DELIVERY_SUMMARY.md`** → Synthèse de livraison

---

## 🎯 WORKFLOW TYPIQUE

```
1. Lancer:    .\run_and_test.ps1
              ⬇️
2. Attendre:  30 secondes
              ⬇️
3. Tester:    python test_api.py
              ⬇️
4. Utiliser:  Envoyer des prédictions
              ⬇️
5. Arrêter:   docker-compose down
```

---

## 💡 ASTUCES PRO

**Redémarrage rapide :**
```powershell
docker-compose restart
```

**Voir l'utilisation ressources :**
```powershell
docker stats
```

**Mode verbose (debug) :**
```powershell
docker-compose logs -f
```

**Test batch (plusieurs prédictions) :**
```powershell
python test_api.py  # Contient déjà des tests batch
```

---

## 🎊 TU ES PRÊT !

**Commande magique à retenir :**
```powershell
.\run_and_test.ps1
```

Cette seule commande fait TOUT ! 🚀

---

**Questions ? Problèmes ?**
→ Consulte `GUIDE_EXECUTION.md` pour le troubleshooting détaillé
