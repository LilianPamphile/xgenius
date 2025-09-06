# ⚽ Soccer Match Prediction Pipeline

**Projet personnel - 2025**  
Pipeline complet et automatisé de **prédiction quotidienne du profil offensif ou défensif** des matchs de football.

---

## 🚀 Objectif du projet

Prédire le **nombre de buts attendus** par match et classer les rencontres en :

- **Ouvertes** (offensives, ≥ 2.5 buts probables)  
- **Fermées** (défensives, ≤ 2.5 buts)  
- **Neutres** (équilibrées)  

Les prédictions combinent :  
- modèles ML empilés (CatBoost, LightGBM, HGB)  
- intervalles conformaux calibrés  
- un modèle heuristique (forme, xG, tirs, discipline)  
- ajustements par ligue  

---

## 🧱 Architecture du pipeline

### 1. Collecte
- API : [API-Football](https://rapidapi.com/api-sports/api/api-football)  
- Stockage : PostgreSQL (Railway)

### 2. Enrichissement
- Forme récente (5 derniers matchs)  
- Stats globales par équipe (xG, tirs, possession, clean sheets)  
- Variables croisées + macros ligue (moyenne buts 60j, avantage domicile)  

### 3. Modélisation
- **Stacking** : CatBoost + HistGradientBoosting + LightGBM  
- **Conformal quantiles** (p25–p75, coverage calibré)  
- **Classif Over2.5 calibrée** (isotonic)  
- **Score heuristique** (proxy basé sur stats brutes)  

### 4. Prédictions & diffusion
- Classement **Over / Under / Opps**  
- Export **CSV** : `suivi_predictions/historique_predictions.csv`  
- Envoi automatique sur **Telegram** 📲  
- (Optionnel) Publication vidéo sur **TikTok** 🎥  

---

## 🧠 Modèles ML

| Modèle                 | Rôle                                     |
|------------------------|------------------------------------------|
| **CatBoost (Optuna)**  | Learner principal                        |
| **HistGradientBoosting** | Learner secondaire                     |
| **LightGBM (quantile)** | Intervalles de confiance                |
| **Ridge Stacking**      | Meta-modèle final                       |
| **Classif GBC calibrée** | Prédiction Over 2.5 (proba calibrée)   |
| **CatBoost heuristique** | Score basé sur signaux explicites      |

---

## 📊 Scores & indicateurs

- **Prédiction buts attendus** (continu)  
- **Intervalle conformal** (incertitude)  
- **Probabilité Over 2.5** calibrée  
- **META score** (0–100) = indice global de match ouvert  
- **Drivers** (ex. `xG↑`, `tirs cadrés↑`, `défenses friables`, `solidité↑`)  

---

## ⚙️ Lancement du projet

### 🔧 Prérequis
- Python 3.10+  
- PostgreSQL (Railway recommandé)  
- Variables d’environnement :
  - `RAPIDAPI_KEY`  
  - `TOKEN_HUB` (push GitHub)  
  - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`  
  - (optionnel) `TIKTOK_CLIENT_KEY`, `TIKTOK_CLIENT_SECRET`, `TIKTOK_ACCESS_TOKEN`  

### 📦 Installation
```bash
git clone https://github.com/LilianPamphile/xgenius.git
cd xgenius
pip install -r requirements.txt
```

▶️ Exécution
```bash
python train_model.py   # Entraînement + push artefacts
python main.py          # Prédictions quotidiennes + export + Telegram
```
📂 Sorties
📊 Telegram Bot → résumé quotidien des matchs

💾 CSV → suivi_predictions/historique_predictions.csv
📁 Models & artefacts → model_files/, artifacts/

👤 Auteur
Lilian Pamphile
📧 lilian.pamphile.bts@gmail.com
