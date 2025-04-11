import streamlit as st
import math
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configuration sobre et raffinée
st.set_page_config(page_title="Bankroll - Paris Sportifs", layout="centered")
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    .stButton>button { border-radius: 8px; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Gestion de Bankroll - Paris Sportifs")

# Initialisation session state
if "historique" not in st.session_state:
    st.session_state.historique = []

# Reset historique
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    if st.button("🔄 Réinitialiser l'historique"):
        st.session_state.historique = []
        st.success("Historique vidé.")

# Formulaire compact
with st.expander("➕ Ajouter un nouveau pari", expanded=True):
    with st.form("form_pari"):
        col1, col2 = st.columns(2)
        with col1:
            match = st.text_input("Match")
            sport = st.selectbox("Sport", ["Football", "Basket", "Tennis"])
            type_pari = st.selectbox("Type de pari", ["Vainqueur", "Over/Under", "Handicap", "Score exact", "Buts marqués", "Nombre de sets", "Autre"])
        with col2:
            evenement = st.text_input("Pari en question")
            cote = st.number_input("Cote proposée", min_value=1.01, step=0.01, format="%.2f")
            cote_adverse = st.number_input("Cote adverse", min_value=1.01, step=0.01, format="%.2f")

        resultat = st.selectbox("Résultat du pari", ["Non joué", "Gagné", "Perdu"])
        submitted = st.form_submit_button("✅ Enregistrer")

        if submitted:
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

            st.session_state.historique.append({
                "Match": match,
                "Sport": sport,
                "Type": type_pari,
                "Pari": evenement,
                "Cote": cote,
                "Cote adverse": cote_adverse,
                "Proba": round(prob_estimee * 100, 2),
                "Marge": round(marge_bookmaker, 2),
                "Value": round(value_bet * 100, 2),
                "Kelly": round(mise_kelly, 2),
                "Demi-Kelly": round(mise_demi_kelly, 2),
                "Résultat": resultat
            })
            st.success("Pari enregistré avec succès ✅")

# Affichage historique & filtres
if st.session_state.historique:
    st.markdown("---")
    st.subheader("📋 Historique des paris")
    df = pd.DataFrame(st.session_state.historique)

    with st.expander("🎛️ Filtres", expanded=True):
        colf1, colf2 = st.columns(2)
        with colf1:
            sports = df["Sport"].unique().tolist()
            filtre_sport = st.multiselect("Sport", sports, default=sports)
        with colf2:
            statut = st.multiselect("Résultat", df["Résultat"].unique().tolist(), default=df["Résultat"].unique().tolist())

    df_filtre = df[(df["Sport"].isin(filtre_sport)) & (df["Résultat"].isin(statut))]
    nb_affiche = st.slider("Nombre de paris affichés", 1, len(df_filtre), min(10, len(df_filtre)))

    df_affiche = df_filtre.tail(nb_affiche)

    df_gagnes = df_affiche[df_affiche["Résultat"] == "Gagné"]
    df_perdus = df_affiche[df_affiche["Résultat"] == "Perdu"]

    bankroll_init = 100.0
    mise_totale = df_affiche["Kelly"].sum()
    gain_total = (df_gagnes["Kelly"] * (df_gagnes["Cote"] - 1)).sum()
    bankroll_finale = bankroll_init + gain_total - df_perdus["Kelly"].sum()
    roi = ((bankroll_finale - bankroll_init) / bankroll_init) * 100 if bankroll_init > 0 else 0

    colr1, colr2 = st.columns(2)
    colr1.metric("💰 ROI", f"{roi:.2f}%")
    colr2.metric("🎯 % gagnés", f"{(len(df_gagnes) / len(df_affiche) * 100):.1f}%")

    st.dataframe(df_affiche, use_container_width=True)

    # Simulation long terme
    st.markdown("---")
    st.subheader("📈 Simulation bankroll sur 100 paris")

    proba = df.iloc[-1]["Proba"] / 100
    cote_sim = df.iloc[-1]["Cote"]

    bankrolls = [100.0]
    for _ in range(100):
        mise = calcul_mise_kelly(bankrolls[-1], proba, cote_sim)
        gain = mise * (cote_sim - 1)
        win = np.random.rand() < proba
        bankrolls.append(bankrolls[-1] + gain if win else bankrolls[-1] - mise)

    fig, ax = plt.subplots()
    ax.plot(bankrolls)
    ax.set_title("Évolution de la bankroll (simulation)")
    ax.set_xlabel("Pari")
    ax.set_ylabel("Bankroll (€)")
    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("Développé avec ❤️ pour les parieurs intelligents | [GitHub](https://https://github.com/LilianPamphile) | [Contact](lilian.pamphile@gmail.com)")
