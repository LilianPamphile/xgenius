import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Gestion de Bankroll - Paris Sportifs", layout="centered")
st.title("🎯 Gestion de Bankroll")

# Initialisation de session state pour historique
if "historique" not in st.session_state:
    st.session_state.historique = []

# Bouton de réinitialisation de l'historique
if st.button("🗑️ Réinitialiser l'historique"):
    st.session_state.historique = []
    st.success("Historique réinitialisé !")

# Choix utilisateur simplifié
st.header("📝 Informations sur le Pari")
match = st.text_input("📅 Match (ex: Nadal vs Djokovic, PSG vs OM, etc.)")
sport = st.selectbox("🏟️ Choisis un sport", ["Football", "Basket", "Tennis"])

# Liste des types de paris
liste_types_paris = ["Vainqueur", "Over/Under", "Handicap", "Score exact", "Buts marqués", "Nombre de sets", "Autre"]
type_pari = st.selectbox("🎯 Type de pari", liste_types_paris)

evenement = st.text_input("🧑‍💼 Pari en question (ex: Berrettini, Real Madrid, etc.)")
cote = st.number_input("💸 Cote proposée (sur ton pari)", min_value=1.01, step=0.01, format="%.2f")
cote_adverse = st.number_input("💸 Cote inverse (autre issue principale)", min_value=1.01, step=0.01, format="%.2f")

st.markdown("---")
st.header("📈 Analyse automatique et Bankroll")

# Calcul automatique de la probabilité implicite et de la marge
proba_implicite = 1 / cote
proba_adverse = 1 / cote_adverse
marge_bookmaker = (proba_implicite + proba_adverse - 1) * 100

# Ajustement réel de la probabilité estimée
prob_estimee = proba_implicite / (proba_implicite + proba_adverse)

# Bankroll constante de départ
bankroll = 100.0

# Fonctions de calcul
def calcul_value_bet(prob_estimee, cote):
    return (prob_estimee * cote) - 1

def calcul_mise_kelly(bankroll, prob_estimee, cote):
    edge = (cote * prob_estimee) - 1
    mise_kelly = bankroll * edge / (cote - 1) if cote > 1 else 0
    return max(0, mise_kelly)

# Calculs
value_bet = calcul_value_bet(prob_estimee, cote)
mise_kelly = calcul_mise_kelly(bankroll, prob_estimee, cote)
mise_demi_kelly = mise_kelly / 2

# Validation du pari
if st.button("✅ Valider ce pari"):
    st.session_state.historique.append({
        "Match": match,
        "Sport": sport,
        "Type de pari": type_pari,
        "Pari": evenement,
        "Cote": cote,
        "Cote adverse": cote_adverse,
        "Proba estimée": round(prob_estimee * 100, 2),
        "Marge": round(marge_bookmaker, 2),
        "Value": round(value_bet * 100, 2),
        "Mise Kelly": round(mise_kelly, 2),
        "Mise demi-Kelly": round(mise_demi_kelly, 2)
    })
    st.success("Pari enregistré dans l'historique !")

# Résultats
st.markdown("---")
st.header("📊 Résultats du pari")
st.markdown(f"**Match :** {match}")
st.markdown(f"**Sport :** {sport}")
st.markdown(f"**Type de pari :** {type_pari}")
st.markdown(f"**Pari :** {evenement}")
st.markdown(f"**Cote :** {cote:.2f} | **Cote adverse :** {cote_adverse:.2f}")
st.markdown(f"📉 **Probabilité implicite :** {proba_implicite*100:.2f}%")
st.markdown(f"📉 **Probabilité estimée (corrigée) :** {prob_estimee*100:.2f}%")
st.markdown(f"📊 **Marge bookmaker :** {marge_bookmaker:.2f}%")
st.markdown(f"💼 **Bankroll de départ :** {bankroll:.2f} €")

if value_bet > 0:
    st.success(f"✅ Value Bet détectée : +{value_bet*100:.2f}%")
else:
    st.warning(f"⚠️ Pas de Value Bet : {value_bet*100:.2f}%")

st.markdown(f"💡 **Mise recommandée (Kelly)** : {mise_kelly:.2f} €")
st.markdown(f"💡 **Mise demi-Kelly** : {mise_demi_kelly:.2f} €")

# Affichage de l'historique
if st.session_state.historique:
    st.markdown("---")
    st.header("📋 Historique des paris enregistrés")
    df_hist = pd.DataFrame(st.session_state.historique)
    st.dataframe(df_hist, use_container_width=True)

# Simulateur long terme (100 paris fixés)
st.markdown("---")
st.header("📈 Simulateur Long Terme (100 paris)")
n_paris = 100
bankroll_initiale = bankroll
bankrolls = [bankroll_initiale]
for i in range(n_paris):
    mise = calcul_mise_kelly(bankrolls[-1], prob_estimee, cote)
    gain = mise * (cote - 1)
    pari_gagnant = np.random.rand() < prob_estimee
    nouveau_bankroll = bankrolls[-1] + gain if pari_gagnant else bankrolls[-1] - mise
    bankrolls.append(nouveau_bankroll)

fig2, ax2 = plt.subplots()
ax2.plot(bankrolls)
ax2.set_title("Simulation de l'évolution de la bankroll sur 100 paris")
ax2.set_xlabel("Pari")
ax2.set_ylabel("Bankroll (€)")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("Développé avec ❤️ pour les parieurs intelligents | [GitHub](https://https://github.com/LilianPamphile) | [Contact](lilian.pamphile@gmail.com)")
