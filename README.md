# ⚽ Soccer Match Prediction Pipeline

**Projet personnel - 2025**  
Pipeline complet et automatisé de **prédiction quotidienne du profil offensif** des matchs de football européens.

---

## 🚀 Objectif du projet

Ce projet a pour but de prédire le **nombre total de buts** attendus pour chaque match de football (compétitions européennes majeures), en combinant :
- des modèles de machine learning avancés (CatBoost, LightGBM, etc.)
- des données enrichies (forme, xG, statistiques défensives...)
- un score heuristique & prédictif : **GMOS**

---

## 🧱 Architecture du pipeline

1. **Collecte des données**
   - API : [RapidAPI - API-Football](https://rapidapi.com/api-sports/api/api-football)
   - Stockage : PostgreSQL (Railway)

2. **Enrichissement des données**
   - Calcul de la forme des équipes (5 derniers matchs)
   - Statistiques moyennes globales : tirs, possession, passes, xG, etc.
   - Création de variables croisées (e.g. diff_xG, total tirs cadrés)

3. **Modélisation et entraînement**
   - Régression : `CatBoost` (Optuna), `HistGradientBoosting`, `LightGBM` (quantile)
   - Estimation d’intervalles (p25–p75) avec modèles conformaux
   - Sauvegarde des modèles sur GitHub

4. **Scoring et prédiction**
   - Score **GMOS** = 40% prédiction ML + 30% heuristique + 30% incertitude
   - Classement des matchs : `ouverts` / `fermés` / `neutres`
   - Envoi automatisé quotidien par email 📩

---

## 🧠 Modèles ML utilisés

| Modèle                  | Description                          |
|-------------------------|--------------------------------------|
| CatBoost + Optuna       | Modèle principal optimisé (RMSE < 1.5) |
| HistGradientBoosting    | Modèle secondaire d’ensemble         |
| LightGBM quantile       | Estimation de l’intervalle de buts   |

---

## 🧬 Features extraites

- **Historique :** moyenne buts, xG, % BTTS, over 2.5  
- **Défensif :** solidité, clean sheets, std buts encaissés  
- **Forme récente :** marqués/encaissés sur les 5 derniers matchs  
- **Profil match :** fautes, cartons, corners, possession  

---

## 📊 GMOS Score

Le **GMOS (Global Match Open Score)** est un score hybride (0 à 100) qui indique si un match a un fort potentiel offensif.

> **≥ 65 :** Match ouvert 🔓  
> **≤ 50 :** Match fermé 🔒  
> Entre 51–64 : Match neutre ⚪

Il combine :
- la prédiction du total de buts (ML)
- les intervalles de confiance (p25–p75)
- des facteurs heuristiques (forme, xG, tirs, etc.)

---

## ⚙️ Lancement du projet

### 🔧 Prérequis
- Python 3.9+
- PostgreSQL (Railway recommandé)
- Variables d'environnement :  
  - `GITHUB_TOKEN`  
  - Clé API `x-rapidapi-key`

### 📦 Installation

```bash
git clone https://github.com/LilianPamphile/paris-sportifs.git
cd paris-sportifs
pip install -r requirements.txt
