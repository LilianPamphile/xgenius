# XGenius Match Radar — Telegram minimal

Version minimale de XGenius avec :

- API-Football ;
- PostgreSQL Railway ;
- GitHub Actions ;
- Telegram.

La connexion Railway est écrite directement dans `main.py`, comme demandé.

## Logique des périodes

Le projet fonctionne avec deux exécutions métier par semaine.

### Lundi

Le lundi, le script fait :

- bilan des matchs du week-end : vendredi, samedi, dimanche ;
- radar des matchs à venir : lundi, mardi, mercredi.

Période utilisée :

```text
Bilan : vendredi 00:00 → lundi 00:00
Radar : lundi 00:00 → jeudi 00:00
```

### Jeudi

Le jeudi, le script fait :

- bilan des matchs de début de semaine : lundi, mardi, mercredi ;
- radar des matchs à venir : jeudi, vendredi, samedi, dimanche.

Période utilisée :

```text
Bilan : lundi 00:00 → jeudi 00:00
Radar : jeudi 00:00 → lundi 00:00
```

Cela évite le problème de l’ancienne version qui récupérait trop large, par exemple samedi → vendredi en lancement manuel.

## Fichiers du dépôt

```text
main.py
requirements.txt
README.md
.gitignore
.github/workflows/xgenius.yml
```

## Secrets GitHub nécessaires

Dans `Settings > Secrets and variables > Actions` :

```text
RAPIDAPI_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

Aucun secret `DATABASE_URL` n’est nécessaire.

## GitHub Actions

GitHub Actions planifie les cron en UTC. Le workflow lance donc un contrôle à `06:17` et `07:17` UTC les lundis et jeudis, puis le job ne continue que lorsque l’heure locale de Paris est bien `08:17`.

Cela permet de gérer automatiquement l’heure d’été et l’heure d’hiver.

## Premier test

Dans GitHub :

```text
Actions > XGenius Match Radar > Run workflow
```

Choisir :

```text
mode = monday
 dry_run = true
```

En `dry_run`, le script appelle API-Football et PostgreSQL, mais affiche les messages dans les logs sans les envoyer sur Telegram.

Après vérification, relancer avec :

```text
dry_run = false
```

## Tables PostgreSQL créées

Le script crée automatiquement :

- `radar_matches` : matchs, prédictions et résultats ;
- `radar_reports` : bilans et radars déjà envoyés pour éviter les doublons.

## Sorties Telegram

Chaque radar affiche jusqu’à 5 matchs :

- signal 1X2 le plus net ;
- potentiel offensif ;
- BTTS à surveiller ;
- match le plus indécis ;
- match potentiellement fermé.

## Sortie Telegram

Le radar envoie maintenant deux niveaux de lecture :

1. les tops par catégorie : 1X2 le plus net, potentiel offensif, BTTS, match indécis, match fermé ;
2. la liste compacte de tous les matchs analysés sur la période.

Variables optionnelles :

- `MAX_RADAR_MATCHES` : nombre de tops affichés, défaut `5` ;
- `SHOW_ALL_MATCHES` : `true` ou `false`, défaut `true` ;
- `MAX_FULL_MATCHES` : nombre maximum de matchs listés dans la liste complète, défaut `120`.
