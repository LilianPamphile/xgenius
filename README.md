# XGenius IA Auto

Projet football Data Science/IA auto-apprenant.

Le système :

1. récupère les calendriers et résultats via API-Football ;
2. stocke l’historique dans PostgreSQL Railway ;
3. construit des features prématch sans fuite temporelle ;
4. entraîne automatiquement un modèle IA ;
5. compare le nouveau modèle avec l’ancien ;
6. active le meilleur modèle ;
7. prédit les matchs du lundi → mercredi et du jeudi → dimanche ;
8. envoie les bilans et radars sur Telegram.

---

## Fonctionnement

### Lundi matin

- Bilan du week-end : vendredi → dimanche.
- Import des résultats et statistiques.
- Évaluation des anciennes prédictions.
- Réentraînement automatique.
- Radar IA : lundi → mercredi.

### Jeudi matin

- Bilan de la semaine : lundi → mercredi.
- Import des résultats et statistiques.
- Évaluation des anciennes prédictions.
- Réentraînement automatique.
- Radar IA : jeudi → dimanche.

---

## Ce que prédit l’IA

Pour chaque match :

- buts attendus domicile ;
- buts attendus extérieur ;
- probabilité victoire domicile ;
- probabilité nul ;
- probabilité victoire extérieur ;
- probabilité Over 2.5 ;
- probabilité BTTS ;
- signal principal ;
- profil du match ;
- score de confiance.

L’IA entraîne deux modèles ExtraTrees :

- un modèle pour les buts de l’équipe domicile ;
- un modèle pour les buts de l’équipe extérieure.

Les probabilités 1X2, Over 2.5 et BTTS sont ensuite dérivées par distribution de Poisson.

---

## Structure

```text
main.py
requirements.txt
.github/workflows/xgenius_ai.yml
xgenius/
  api_football.py
  config.py
  db.py
  evaluation.py
  features.py
  jobs.py
  modeling.py
  reporting.py
  telegram.py
  time_windows.py
```

---

## Base de données

Le code crée automatiquement les tables :

```text
ai_fixtures
ai_team_match_stats
ai_predictions
ai_model_runs
ai_reports
```

Le modèle entraîné est stocké directement en base PostgreSQL dans `ai_model_runs.artifact`.
Aucun fichier `.pkl` n’est nécessaire dans GitHub.

---

## Secrets GitHub nécessaires

Dans :

```text
Settings → Secrets and variables → Actions
```

Créer uniquement :

```text
RAPIDAPI_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

La `DATABASE_URL` est volontairement dans `xgenius/config.py`, comme demandé.

---

## Installation dans GitHub

1. Supprimer les anciens fichiers du dépôt.
2. Copier tous les fichiers de ce projet à la racine du dépôt.
3. Ajouter les 3 secrets GitHub.
4. Aller dans `Actions`.
5. Lancer d’abord :

```text
XGenius IA Auto → Run workflow
mode = bootstrap
dry_run = true
bootstrap_days = 90
```

6. Si les logs sont propres, relancer :

```text
mode = bootstrap
dry_run = false
bootstrap_days = 90
```

7. Lancer ensuite :

```text
mode = status
dry_run = false
```

8. Puis tester :

```text
mode = monday
dry_run = true
```

9. Si tout est bon :

```text
mode = monday
dry_run = false
```

---

## Important : première utilisation

Au début, si la base ne contient pas assez de matchs terminés, XGenius utilise une baseline dynamique.
Dès que l’historique atteint assez de matchs, il entraîne automatiquement le modèle IA.

Le seuil par défaut est :

```text
MIN_TRAIN_MATCHES = 120
```

Tu peux le modifier dans `xgenius/config.py`.

---

## Variables de réglage utiles

Dans `xgenius/config.py` :

```python
MIN_TRAIN_MATCHES = 120
MAX_STATS_IMPORT_PER_RUN = 80
MAX_RADAR_TOP_MATCHES = 8
MAX_FULL_LIST_MATCHES = 160
SHOW_FULL_LIST = True
```

---

## Ajouter ou retirer des compétitions

Modifier simplement `MONITORED_LEAGUES` dans `xgenius/config.py`.

Exemple :

```python
MONITORED_LEAGUES = {
    61: "Ligue 1",
    39: "Premier League",
    1: "World Cup",
}
```

---

## Commandes locales

```bash
pip install -r requirements.txt
python main.py --mode bootstrap --dry-run true --bootstrap-days 30
python main.py --mode status --dry-run true
python main.py --mode monday --dry-run true
python main.py --mode thursday --dry-run true
```

---

## Logique anti-doublon

Les bilans et radars envoyés sont enregistrés dans `ai_reports`.
Si GitHub Actions relance le même traitement, le message n’est pas renvoyé.

Pour forcer un renvoi manuel :

```bash
python main.py --mode monday --dry-run false --force true
```
