import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration de la page
st.set_page_config(page_title="Gestion de Bankroll - Paris Sportifs", layout="wide")
st.title("🎯 Gestion de Bankroll")

# Initialisation de session state pour historique
if "historique" not in st.session_state:
    st.session_state.historique = []

# Bouton de réinitialisation de l'historique
st.sidebar.button("🗑️ Réinitialiser l'historique", on_click=lambda: st.session_state.update({"historique": []}))

# Formulaire dans des colonnes pour ergonomie
st.header("📝 Saisir un nouveau pari")
col1, col2, col3 = st.columns(3)
with col1:
    match = st.text_input("📅 Match")
    sport = st.selectbox("🏟️ Sport", ["Football", "Basket", "Tennis"])
with col2:
    type_pari = st.selectbox("🎯 Type de pari", ["Vainqueur", "Over/Under", "Handicap", "Score exact", "Buts marqués", "Nombre de sets", "Autre"])
    evenement = st.text_input("👤 Pari en question")
with col3:
    cote = st.number_input("💸 Cote proposée", min_value=1.01, step=0.01, format="%.2f")
    cote_adverse = st.number_input("💸 Cote adverse", min_value=1.01, step=0.01, format="%.2f")

# Analyse automatique
st.markdown("---")
st.header("📈 Analyse et recommandations")
proba_implicite = 1 / cote
proba_adverse = 1 / cote_adverse
marge_bookmaker = (proba_implicite + proba_adverse - 1) * 100
prob_estimee = proba_implicite / (proba_implicite + proba_adverse)
bankroll = 100.0

def calcul_value_bet(prob_estimee, cote):
    return (prob_estimee * cote) - 1

def calcul_mise_kelly(bankroll, prob_estimee, cote):
    edge = (cote * prob_estimee) - 1
    mise_kelly = bankroll * edge / (cote - 1) if cote > 1 else 0
    return max(0, mise_kelly)

value_bet = calcul_value_bet(prob_estimee, cote)
mise_kelly = calcul_mise_kelly(bankroll, prob_estimee, cote)
mise_demi_kelly = mise_kelly / 2

resultat = st.selectbox("📌 Résultat du pari", ["Non joué", "Gagné", "Perdu"])
if st.button("✅ Enregistrer ce pari"):
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
        "Mise demi-Kelly": round(mise_demi_kelly, 2),
        "Résultat": resultat
    })
    st.success("Pari enregistré ✅")

# Affichage des résultats
st.subheader("📊 Détails du pari")
st.markdown(f"**Probabilité implicite :** {proba_implicite*100:.2f}%")
st.markdown(f"**Probabilité estimée :** {prob_estimee*100:.2f}%")
st.markdown(f"**Marge du bookmaker :** {marge_bookmaker:.2f}%")
st.markdown(f"**Value bet :** {value_bet*100:.2f}%")
st.markdown(f"💸 **Mise (Kelly)** : {mise_kelly:.2f} €  |  **Demi-Kelly** : {mise_demi_kelly:.2f} €")

# Historique avec filtres visuels
if st.session_state.historique:
    st.markdown("---")
    st.header("📋 Historique des paris")
    df_hist = pd.DataFrame(st.session_state.historique)

    with st.expander("🔍 Filtres avancés"):
        colf1, colf2, colf3 = st.columns(3)
        with colf1:
            sports_dispo = df_hist["Sport"].unique().tolist()
            sport_filtre = st.multiselect("🎽 Filtrer par sport", sports_dispo, default=sports_dispo)
        with colf2:
            statut_filtre = st.multiselect("📌 Filtrer par résultat", ["Non joué", "Gagné", "Perdu"], default=["Non joué", "Gagné", "Perdu"])
        with colf3:
            nb_affiche = st.slider("📄 Nombre de paris affichés", min_value=1, max_value=len(df_hist), value=min(10, len(df_hist)))

    df_filtre = df_hist[(df_hist["Sport"].isin(sport_filtre)) & (df_hist["Résultat"].isin(statut_filtre))]

    # Statistiques
    df_gagnes = df_filtre[df_filtre["Résultat"] == "Gagné"]
    df_perdus = df_filtre[df_filtre["Résultat"] == "Perdu"]

    mise_totale = df_filtre["Mise Kelly"].sum()
    gain_total = (df_gagnes["Mise Kelly"] * (df_gagnes["Cote"] - 1)).sum()
    bankroll_finale = bankroll + gain_total - df_perdus["Mise Kelly"].sum()
    roi = ((bankroll_finale - bankroll) / bankroll) * 100 if bankroll > 0 else 0

    colstat1, colstat2 = st.columns(2)
    colstat1.metric("📈 ROI (%)", f"{roi:.2f}%")
    colstat2.metric("✅ % de paris gagnés", f"{(len(df_gagnes) / len(df_filtre) * 100) if len(df_filtre) > 0 else 0:.1f}%")

    st.dataframe(df_filtre.tail(nb_affiche), use_container_width=True)

# Simulation long terme
st.markdown("---")
st.header("📈 Simulation de bankroll sur 100 paris")
bankrolls = [bankroll]
for i in range(100):
    mise = calcul_mise_kelly(bankrolls[-1], prob_estimee, cote)
    gain = mise * (cote - 1)
    pari_gagnant = np.random.rand() < prob_estimee
    bankrolls.append(bankrolls[-1] + gain if pari_gagnant else bankrolls[-1] - mise)

fig2, ax2 = plt.subplots()
ax2.plot(bankrolls)
ax2.set_title("Simulation de l'évolution de la bankroll")
ax2.set_xlabel("Pari")
ax2.set_ylabel("Bankroll (€)")
st.pyplot(fig2)

# Footer
st.markdown("---")
st.markdown("Développé avec ❤️ pour les parieurs intelligents | [GitHub](https://https://github.com/LilianPamphile) | [Contact](lilian.pamphile@gmail.com)")
