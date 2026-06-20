# XGenius Match Radar

Version volontairement minimale de XGenius.

Deux fois par semaine, le projet :

- actualise les résultats récents ;
- publie un bilan des anciennes prédictions ;
- récupère les prochains matchs ;
- appelle les prédictions d'API-Football ;
- calcule un signal Over 2,5 et BTTS avec une approximation Poisson ;
- publie un radar sous forme de thread sur X.

## Fonctionnement

### Lundi à 08 h 17, heure de Paris

- bilan des matchs terminés depuis la précédente exécution ;
- calendrier du lundi au dimanche ;
- radar de la semaine.

### Jeudi à 08 h 17, heure de Paris

- bilan des matchs terminés depuis la précédente exécution ;
- calendrier du jeudi au dimanche ;
- radar du week-end.

Les tables PostgreSQL sont créées automatiquement au premier lancement.

## Fichiers nécessaires

```text
main.py
requirements.txt
.github/workflows/xgenius.yml
.gitignore
README.md
```

Tout l'ancien système ML, Telegram, TikTok et CSV peut être supprimé.

## Secrets GitHub à créer

Dans `Settings > Secrets and variables > Actions > New repository secret` :

```text
DATABASE_URL
RAPIDAPI_KEY
X_API_KEY
X_API_SECRET
X_ACCESS_TOKEN
X_ACCESS_TOKEN_SECRET
```

L'application X doit disposer des droits de lecture et d'écriture. Après avoir modifié les droits, régénérer les Access Token et Access Token Secret.

## Premier test

1. Aller dans l'onglet **Actions** du dépôt.
2. Ouvrir **XGenius Match Radar**.
3. Cliquer sur **Run workflow**.
4. Choisir `monday` ou `thursday`.
5. Laisser `dry_run` activé.

Les publications sont alors affichées dans les logs sans être envoyées sur X.

Quand le résultat est correct, relancer manuellement avec `dry_run` désactivé. Les exécutions programmées du lundi et du jeudi publient automatiquement.

## Compétitions suivies

La liste se trouve en haut de `main.py` dans `COMPETITIONS`. Pour retirer une compétition, supprimer simplement sa ligne.

## Lancement local

```bash
pip install -r requirements.txt
python main.py --mode monday
```

Variables requises :

```text
DATABASE_URL
RAPIDAPI_KEY
```

Pour publier réellement sur X, ajouter également les quatre variables `X_*` et définir :

```text
DRY_RUN=false
```

## Tables créées

- `radar_matches` : matchs, prédictions et résultats ;
- `radar_reports` : publications déjà envoyées, afin d'éviter les doublons.

## Limites assumées

- les probabilités 1X2 viennent directement d'API-Football ;
- toutes les compétitions ne disposent pas forcément de prédictions ;
- le signal BTTS est une estimation Poisson simple, pas une cote bookmaker ;
- le nombre de matchs analysés est limité à 70 par exécution pour préserver le quota API.
