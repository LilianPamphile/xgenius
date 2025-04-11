import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Gestion de Bankroll - Paris Sportifs", layout="centered")
st.title("🎯 Application de Paris Sportifs - Gestion de Bankroll")

# Choix utilisateur simplifié
st.header("📝 Informations sur le Pari")
sport = st.selectbox("🏟️ Choisis un sport", ["Football", "Basket", "Tennis"])

# Liste des types de paris
liste_types_paris = ["Vainqueur", "Over/Under", "Handicap", "Score exact", "Buts marqués", "Nombre de sets", "Autre"]
type_pari = st.selectbox("🎯 Type de pari", liste_types_paris)

evenement = st.text_input("🧑‍💼 Pari en question (ex: Berrettini, Real Madrid, etc.)")
cote = st.number_input("💸 Cote proposée", min_value=1.01, step=0.01, format="%.2f")

st.markdown("---")
st.header("📈 Probabilité et Bankroll")
prob_estimee = st.slider("Probabilité estimée du pari (%)", min_value=1, max_value=100, value=50) / 100
bankroll = st.number_input("💰 Bankroll actuelle (€)", min_value=1.0, step=1.0, format="%.2f")

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

# Résultats
st.markdown("---")
st.header("📊 Résultats du pari")
st.markdown(f"**Sport :** {sport}")
st.markdown(f"**Type de pari :** {type_pari}")
st.markdown(f"**Pari :** {evenement}")
st.markdown(f"**Cote :** {cote:.2f}")

if value_bet > 0:
    st.success(f"✅ Value Bet détectée : +{value_bet*100:.2f}%")
else:
    st.warning(f"⚠️ Pas de Value Bet : {value_bet*100:.2f}%")

st.markdown(f"💡 **Mise recommandée (Kelly)** : {mise_kelly:.2f} €")
st.markdown(f"💡 **Mise demi-Kelly** : {mise_demi_kelly:.2f} €")

# Graphique d'évolution fictif
st.markdown("---")
st.header("📉 Évolution fictive de la bankroll")
data = {
    "Match": [f"Pari {i}" for i in range(1, 11)],
    "Bankroll": [bankroll + (i * 10 - 5 * (i % 2)) for i in range(10)]
}
df_bankroll = pd.DataFrame(data)
fig, ax = plt.subplots()
ax.plot(df_bankroll["Match"], df_bankroll["Bankroll"], marker='o')
ax.set_xlabel("Match")
ax.set_ylabel("Bankroll (€)")
ax.set_title("Historique de la Bankroll")
plt.xticks(rotation=45)
st.pyplot(fig)

# Simulateur long terme
st.markdown("---")
st.header("📈 Simulateur Long Terme")
n_paris = st.slider("Nombre de paris à simuler", min_value=10, max_value=500, value=100, step=10)
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
ax2.set_title("Simulation de l'évolution de la bankroll")
ax2.set_xlabel("Pari")
ax2.set_ylabel("Bankroll (€)")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("Développé avec ❤️ pour les parieurs intelligents | [GitHub](https://github.com/) | [Contact](mailto:contact@example.com)")
