"""
DÉMONSTRATION COMPLÈTE - TEST BATCH PREDICTION
Montre exactement comment les tests batch fonctionnent
"""

import json
import requests
from datetime import datetime
from pathlib import Path

API_URL = "http://localhost:5000"

def demonstration_complete():
    """Démonstration complète du processus de test batch"""
    
    print("\n" + "="*70)
    print("  🔬 DÉMONSTRATION TEST BATCH - PROCESSUS COMPLET")
    print("="*70 + "\n")
    
    # ========================================================================
    # ÉTAPE 1: CHARGEMENT DES DONNÉES JSON
    # ========================================================================
    print("📂 ÉTAPE 1: Chargement des fichiers JSON")
    print("-" * 70)
    
    # Charger CPU
    cpu_file = Path(__file__).parent / 'data' / 'cluster_cpu_request_ratio.json'
    with open(cpu_file, 'r') as f:
        cpu_json = json.load(f)
    cpu_values = cpu_json['data']['result'][0]['values']
    print(f"✅ CPU data chargée: {len(cpu_values)} points")
    print(f"   Format: [[timestamp, valeur], ...]")
    print(f"   Exemple: {cpu_values[0]}")
    
    # Charger Memory
    mem_file = Path(__file__).parent / 'data' / 'cluster_mem_request_ratio.json'
    with open(mem_file, 'r') as f:
        mem_json = json.load(f)
    mem_values = mem_json['data']['result'][0]['values']
    print(f"✅ Memory data chargée: {len(mem_values)} points")
    
    # Charger Pods
    pod_file = Path(__file__).parent / 'data' / 'cluster_pod_ratio.json'
    with open(pod_file, 'r') as f:
        pod_json = json.load(f)
    pod_values = pod_json['data']['result'][0]['values']
    print(f"✅ Pods data chargée: {len(pod_values)} points\n")
    
    # ========================================================================
    # ÉTAPE 2: FUSION DES DONNÉES PAR TIMESTAMP
    # ========================================================================
    print("🔗 ÉTAPE 2: Fusion des 3 métriques par timestamp")
    print("-" * 70)
    
    # Créer des dictionnaires pour accès rapide
    cpu_dict = {int(ts): float(val) for ts, val in cpu_values}
    mem_dict = {int(ts): float(val) for ts, val in mem_values}
    pod_dict = {int(ts): float(val) for ts, val in pod_values}
    
    # Trouver timestamps communs
    cpu_times = set(cpu_dict.keys())
    mem_times = set(mem_dict.keys())
    pod_times = set(pod_dict.keys())
    common_times = sorted(cpu_times & mem_times & pod_times)
    
    print(f"✅ Timestamps communs trouvés: {len(common_times)}")
    print(f"   CPU timestamps: {len(cpu_times)}")
    print(f"   Memory timestamps: {len(mem_times)}")
    print(f"   Pods timestamps: {len(pod_times)}")
    print(f"   Intersection: {len(common_times)}\n")
    
    # ========================================================================
    # ÉTAPE 3: CONSTRUCTION DU BATCH
    # ========================================================================
    print("📦 ÉTAPE 3: Construction du batch (premiers 10 échantillons)")
    print("-" * 70)
    
    batch = []
    for ts in common_times[:10]:  # Prendre 10 pour la démo
        sample = {
            "timestamp": datetime.fromtimestamp(ts).isoformat(),
            "cluster_cpu_request_ratio": cpu_dict[ts],
            "cluster_mem_request_ratio": mem_dict[ts],
            "cluster_pod_ratio": pod_dict[ts]
        }
        batch.append(sample)
    
    print(f"✅ Batch créé avec {len(batch)} échantillons")
    print("\n📋 Exemple des 3 premiers échantillons:\n")
    
    for i, sample in enumerate(batch[:3], 1):
        print(f"  Échantillon {i}:")
        print(f"    Timestamp: {sample['timestamp']}")
        print(f"    CPU: {sample['cluster_cpu_request_ratio']:.4f} ({sample['cluster_cpu_request_ratio']*100:.1f}%)")
        print(f"    Memory: {sample['cluster_mem_request_ratio']:.4f} ({sample['cluster_mem_request_ratio']*100:.1f}%)")
        print(f"    Pods: {sample['cluster_pod_ratio']:.4f} ({sample['cluster_pod_ratio']*100:.1f}%)")
        print()
    
    # ========================================================================
    # ÉTAPE 4: STRUCTURE DE LA REQUÊTE
    # ========================================================================
    print("📨 ÉTAPE 4: Structure de la requête HTTP")
    print("-" * 70)
    
    request_payload = {
        "samples": batch
    }
    
    print("✅ Format de la requête:")
    print(f"   URL: POST {API_URL}/batch_predict")
    print(f"   Content-Type: application/json")
    print(f"   Body:")
    print(f"   {{")
    print(f"     \"samples\": [")
    print(f"       {{")
    print(f"         \"timestamp\": \"2025-10-27T12:30:00\",")
    print(f"         \"cluster_cpu_request_ratio\": 0.6154,")
    print(f"         \"cluster_mem_request_ratio\": 0.6461,")
    print(f"         \"cluster_pod_ratio\": 0.1933")
    print(f"       }},")
    print(f"       ... ({len(batch)} échantillons total)")
    print(f"     ]")
    print(f"   }}\n")
    
    # ========================================================================
    # ÉTAPE 5: ENVOI DE LA REQUÊTE
    # ========================================================================
    print("🚀 ÉTAPE 5: Envoi de la requête à l'API")
    print("-" * 70)
    
    print(f"Envoi de {len(batch)} échantillons à l'API...")
    
    try:
        response = requests.post(
            f"{API_URL}/batch_predict",
            json=request_payload,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        print(f"✅ Requête envoyée")
        print(f"   Status Code: {response.status_code}")
        print(f"   Response Time: {response.elapsed.total_seconds():.2f}s\n")
        
        # ====================================================================
        # ÉTAPE 6: TRAITEMENT DE LA RÉPONSE
        # ====================================================================
        print("📥 ÉTAPE 6: Réception et analyse de la réponse")
        print("-" * 70)
        
        if response.status_code == 200:
            result = response.json()
            
            predictions = result.get('predictions', [])
            summary = result.get('summary', {})
            
            print("✅ Prédiction réussie!")
            print(f"\n📊 Résumé:")
            print(f"   Total échantillons: {summary.get('total', 0)}")
            print(f"   Échantillons valides: {summary.get('valid', 0)}")
            print(f"   Anomalies détectées: {summary.get('anomalies', 0)}")
            print(f"   Normaux: {summary.get('normal', 0)}")
            print(f"   Taux d'anomalies: {summary.get('anomaly_rate', 0)}%")
            
            print(f"\n🔍 Détails des prédictions (premiers 5):\n")
            for i, pred in enumerate(predictions[:5], 1):
                status = "🚨 ANOMALY" if pred['is_anomaly'] else "✅ NORMAL"
                print(f"   {i}. {status} - Timestamp: {pred['timestamp']}")
            
            if len(predictions) > 5:
                print(f"   ... et {len(predictions)-5} autres prédictions")
            
        else:
            print(f"❌ Erreur: {response.status_code}")
            print(f"   Message: {response.text[:200]}")
            
    except Exception as e:
        print(f"❌ Erreur lors de l'envoi: {str(e)}")
    
    # ========================================================================
    # ÉTAPE 7: CE QUI SE PASSE DANS L'API
    # ========================================================================
    print("\n⚙️  ÉTAPE 7: Ce qui se passe dans l'API")
    print("-" * 70)
    print("""
    Lorsque l'API reçoit le batch:
    
    1. Validation des données
       ✓ Vérifie que 'samples' existe
       ✓ Vérifie les champs requis (cpu, mem, pods)
    
    2. Feature Engineering
       ✓ Crée un DataFrame avec TOUS les échantillons
       ✓ Génère 104 features:
         - Rolling windows (mean, std, min, max, median, skew, kurt)
         - Lag features (12, 24, 48, 96 périodes)
         - Features temporelles (hour, day, weekend, etc.)
         - Features statistiques avancées
    
    3. Prédiction
       ✓ Normalise les features avec StandardScaler
       ✓ Applique le modèle One-Class SVM
       ✓ Résultat: 1 (normal) ou -1 (anomalie) par échantillon
    
    4. Génération des alertes
       ✓ Pour chaque anomalie détectée:
         → Crée une alerte AzureClusterAnomaly
         → Envoie à Alertmanager (http://alertmanager:9093)
         → Severity: warning (≤5 anomalies) ou critical (>5)
    
    5. Retour de la réponse
       ✓ Liste des prédictions avec timestamps
       ✓ Résumé statistique
       ✓ Status 200 si succès
    """)
    
    print("\n" + "="*70)
    print("  ✅ DÉMONSTRATION TERMINÉE")
    print("="*70 + "\n")
    
    print("📝 Résumé du flux:")
    print("   JSON files → Fusion → Batch → API → Feature Engineering")
    print("                                   ↓")
    print("   Email ← Alertmanager ← Alerts ← Prédictions ← Model\n")

if __name__ == "__main__":
    demonstration_complete()
