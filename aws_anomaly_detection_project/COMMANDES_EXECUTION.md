# 🚀 COMMANDES D'EXÉCUTION - AZURE ANOMALY DETECTION

## 📍 SE POSITIONNER DANS LE PROJET

```powershell
cd C:\Users\acer\Desktop\yassmine\aws_anomaly_detection_project
```

---

## 1️⃣ DÉMARRER LE PROJET

### Lancer tous les services (API + Alertmanager)
```powershell
docker-compose up -d
```
✅ Lance en arrière-plan  
✅ Attend 30 secondes que les services démarrent

### Lancer avec reconstruction complète
```powershell
docker-compose up -d --build
```
✅ Reconstruit les images Docker avant de lancer

---

## 2️⃣ VÉRIFIER LE STATUT

### Voir les containers en cours
```powershell
docker ps
```
✅ Affiche: aws-anomaly-detection-api + alertmanager

### Voir le statut détaillé
```powershell
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

---

## 3️⃣ TESTER L'API

### Health check
```powershell
curl http://localhost:5000/health
```
✅ Doit retourner: `{"status": "healthy"}`

### Info du modèle
```powershell
curl http://localhost:5000/model_info
```
✅ Retourne: détails du modèle One-Class SVM

### Service info
```powershell
curl http://localhost:5000/
```
✅ Retourne: liste des endpoints disponibles

---

## 4️⃣ TESTER LES ALERTES

### Test complet des alertes Azure
```powershell
python test_alerts.py
```
✅ Teste 5 étapes:
- Alertmanager status
- Envoi d'alerte test
- Alertes actives
- Prédiction avec anomalie
- Vérification alerte Azure reçue

### Résultat attendu
```
Tests passed: 5/5
🎉 ALL TESTS PASSED!
```

---

## 5️⃣ TESTER BATCH PREDICTION

### Test avec données réelles (150 échantillons)
```powershell
python test_batch_predict.py
```
✅ Utilise les données des fichiers JSON  
✅ Teste le feature engineering  
✅ Affiche les anomalies détectées

### Démonstration du processus complet
```powershell
python demo_batch_process.py
```
✅ Montre étape par étape comment ça fonctionne  
✅ Affiche 10 échantillons en détail

---

## 6️⃣ VOIR LES LOGS

### Logs de l'API (dernières 50 lignes)
```powershell
docker logs aws-anomaly-detection-api --tail 50
```

### Logs en temps réel (suivre)
```powershell
docker logs aws-anomaly-detection-api --follow
```
Appuyez sur `Ctrl+C` pour arrêter

### Logs Alertmanager
```powershell
docker logs alertmanager --tail 50
```

---

## 7️⃣ INTERFACES WEB

### Ouvrir l'API dans le navigateur
```powershell
start http://localhost:5000
```

### Ouvrir Alertmanager UI
```powershell
start http://localhost:9093
```

---

## 8️⃣ ARRÊTER LE PROJET

### Arrêter les containers (conserver les données)
```powershell
docker-compose stop
```

### Arrêter et supprimer les containers
```powershell
docker-compose down
```

### Arrêter et supprimer TOUT (containers + volumes + réseaux)
```powershell
docker-compose down -v
```
⚠️ Supprime aussi les données d'Alertmanager

---

## 9️⃣ REDÉMARRER LE PROJET

### Après un arrêt
```powershell
docker-compose start
```

### Reconstruction complète
```powershell
docker-compose down -v
docker-compose up -d --build
```

---

## 🔟 COMMANDES UTILES

### Voir l'utilisation des ressources
```powershell
docker stats
```

### Accéder au shell du container API
```powershell
docker exec -it aws-anomaly-detection-api /bin/bash
```

### Nettoyer les images Docker inutilisées
```powershell
docker system prune -a
```

---

## 🧪 SÉQUENCE DE TEST COMPLÈTE

```powershell
# 1. Se positionner
cd C:\Users\acer\Desktop\yassmine\aws_anomaly_detection_project

# 2. Nettoyer et reconstruire
docker-compose down -v
docker-compose up -d --build

# 3. Attendre le démarrage (30 secondes)
Start-Sleep -Seconds 30

# 4. Vérifier le statut
docker ps

# 5. Tester l'API
curl http://localhost:5000/health

# 6. Tester les alertes
python test_alerts.py

# 7. Tester le batch
python test_batch_predict.py

# 8. Voir les logs
docker logs aws-anomaly-detection-api --tail 50
```

---

## 📊 VÉRIFICATIONS RAPIDES

### Vérifier que tout fonctionne
```powershell
# API healthy?
$health = Invoke-RestMethod http://localhost:5000/health
Write-Host "API Status: $($health.status)"

# Alertmanager running?
$am = Invoke-RestMethod http://localhost:9093/api/v2/status
Write-Host "Alertmanager Version: $($am.versionInfo.version)"

# Containers up?
docker ps --format "{{.Names}}: {{.Status}}"
```

### Résultat attendu
```
API Status: healthy
Alertmanager Version: 0.26.0
aws-anomaly-detection-api: Up (healthy)
alertmanager: Up (healthy)
```

---

## 🌐 URLS D'ACCÈS

- **API**: http://localhost:5000
- **API Health**: http://localhost:5000/health
- **API Model Info**: http://localhost:5000/model_info
- **Alertmanager UI**: http://localhost:9093
- **Alertmanager API**: http://localhost:9093/api/v2/alerts

---

## 📧 EMAILS CONFIGURÉS

Les alertes sont envoyées automatiquement à:
- mohamedamine.chaabani@esprit.tn
- aminchaabeni2000@gmail.com

---

## 🔧 DÉPANNAGE

### Container ne démarre pas
```powershell
# Voir les erreurs
docker logs aws-anomaly-detection-api

# Reconstruire
docker-compose up -d --build --force-recreate
```

### Port déjà utilisé
```powershell
# Trouver le processus
netstat -ano | findstr :5000

# Arrêter le processus
taskkill /PID <PID> /F
```

### Réinitialisation complète
```powershell
docker-compose down -v
docker system prune -a
docker-compose up -d --build
```

---

## 📚 DOCUMENTATION

- **Guide rapide**: QUICKSTART_GUIDE.md
- **Guide complet**: GUIDE_EXECUTION.md
- **Commandes**: COMMANDES_REFERENCE.md
- **Données de test**: TEST_DATA_DOCUMENTATION.md
- **Migration Azure**: AZURE_MIGRATION_REPORT.md
- **Index**: INDEX_DOCUMENTATION.md

---

## ✅ CHECKLIST DE DÉMARRAGE

- [ ] Se positionner dans le projet
- [ ] Lancer `docker-compose up -d`
- [ ] Attendre 30 secondes
- [ ] Vérifier `docker ps`
- [ ] Tester `curl http://localhost:5000/health`
- [ ] Exécuter `python test_alerts.py`
- [ ] Ouvrir http://localhost:9093

**Si tous les tests passent → Système opérationnel ! ✅**
