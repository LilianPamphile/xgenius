# XGenius Match Radar

Version minimale de XGenius avec PostgreSQL Railway, GitHub Actions, API-Football et Telegram.

## Fonctionnement

### Lundi à 08 h 17, heure de Paris

- récupération des résultats récents ;
- bilan des anciennes prédictions ;
- récupération des matchs du lundi au dimanche ;
- génération du radar de la semaine ;
- envoi automatique sur Telegram.

### Jeudi à 08 h 17, heure de Paris

- récupération des nouveaux résultats ;
- bilan des matchs joués depuis lundi ;
- récupération des matchs du jeudi au dimanche ;
- actualisation du radar du week-end ;
- envoi automatique sur Telegram.

Les tables PostgreSQL sont créées automatiquement au premier lancement.

## Fichiers du dépôt

```text
main.py
requirements.txt
.github/workflows/xgenius.yml
.gitignore
README.md
```

## Secrets GitHub à conserver

Dans `Settings > Secrets and variables > Actions` :

```text
RAPIDAPI_KEY
TELEGRAM_BOT_TOKEN
TELEGRAM_CHAT_ID
```

La connexion Railway est déjà écrite directement dans `main.py`. Aucun secret `DATABASE_URL` n'est nécessaire.

## Créer le bot Telegram

1. Ouvrir Telegram et écrire à `@BotFather`.
2. Envoyer `/newbot` et récupérer le token.
3. Envoyer un message au nouveau bot.
4. Ouvrir dans un navigateur :

```text
https://api.telegram.org/bot<VOTRE_TOKEN>/getUpdates
```

5. Copier la valeur `message.chat.id` dans le secret `TELEGRAM_CHAT_ID`.

## Premier test

1. Aller dans l'onglet **Actions** du dépôt.
2. Ouvrir **XGenius Match Radar**.
3. Cliquer sur **Run workflow**.
4. Choisir `monday`.
5. Laisser `dry_run` activé.

Le script appelle API-Football et PostgreSQL, mais affiche les messages dans les logs sans les envoyer.

Après validation, relancer avec `dry_run` désactivé.

## Tables créées

- `radar_matches` : matchs, prédictions et résultats ;
- `radar_reports` : bilans et radars déjà envoyés, pour éviter les doublons.

## Prédictions affichées

- probabilités domicile, nul et extérieur ;
- buts estimés ;
- probabilité Over 2,5 ;
- probabilité BTTS ;
- signal principal ;
- confiance ;
- profil du match.

Les probabilités 1X2 viennent de l'endpoint `/predictions` d'API-Football. Les estimations de buts, Over 2,5 et BTTS utilisent une approximation Poisson simple basée sur la forme récente fournie par l'API.

## Compétitions suivies

La liste se trouve dans `COMPETITIONS` en haut de `main.py`.
