# ✅ Logique Kelly avec proba boostée en fonction de la cote (meilleure cohérence de mise)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid

st.set_page_config(page_title="Bankroll - Paris Sportifs", layout="centered")
st.title("🎯 Gestion de Bankroll - Paris Sportifs")

# Initialisation
if "historique" not in st.session_state:
    st.session_state.historique = []
if "paris_combine" not in st.session_state:
    st.session_state.paris_combine = []
if "bankroll" not in st.session_state:
    st.session_state.bankroll = 100.0

# Fonction Kelly optimale
def kelly(bankroll, p, c):
    if c <= 1 or not 0 < p < 1:
        return 0.0
    edge = (c * p - 1)
    return bankroll * edge / (c - 1) if edge > 0 else 0.0

# Proba estimée uniquement en fonction de la cote (boostée pour réalisme)
def proba_estimee(c):
    implicite = 1 / c
    return max(0.01, min(0.99, implicite * 1.08))

# Réinitialisation & Graphique dans la sidebar
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    if st.button("🔄 Réinitialiser l'historique"):
        st.session_state.historique = []
        st.session_state.bankroll = 100.0
        st.success("Historique et bankroll réinitialisés.")
    if st.button("🧹 Réinitialiser combiné"):
        st.session_state.paris_combine = []

    # Affichage de la bankroll actuelle
    st.markdown("---")
    st.markdown(f"### 💰 Bankroll actuelle : {st.session_state.bankroll:.2f} €")

    # Mini-graphique Kelly vs Cote
    st.markdown("---")
    st.markdown("### 📈 Courbe Kelly vs Cote")
    cotes_range = np.linspace(1.01, 5.0, 60)
    probas = [proba_estimee(c) for c in cotes_range]
    kelly_vals = [kelly(100, p, c) for p, c in zip(probas, cotes_range)]
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    ax.plot(cotes_range, kelly_vals, color='blue', linewidth=2)
    ax.set_xlabel("Cote")
    ax.set_ylabel("Mise (€)")
    ax.set_title("Kelly vs Cote")
    ax.grid(True)
    st.pyplot(fig, clear_figure=True)
    st.caption("📌 Proba = (1 / cote) × 1.08")

# Type de pari (Simple ou Combiné)
st.markdown("### 🎲 Type de pari")
type_global = st.radio("Choisir le type de pari", ["Simple", "Combiné"], horizontal=True)

# --- Formulaire pari simple ---
if type_global == "Simple":
    with st.expander("➕ Ajouter un pari simple", expanded=True):
        with st.form("form_simple"):
            col1, col2 = st.columns(2)
            with col1:
                match = st.text_input("Match")
                sport = st.selectbox("Sport", ["Football", "Basket", "Tennis"])
                type_pari = st.selectbox("Type", ["Vainqueur", "Over/Under", "Handicap", "Score exact", "Autre"])
            with col2:
                evenement = st.text_input("Pari")
                cote = st.number_input("Cote", 1.01, step=0.01, format="%.2f")

            proba = proba_estimee(cote)
            bankroll = st.session_state.bankroll
            mise_kelly = kelly(bankroll, proba, cote)
            mise_demi = mise_kelly / 2

            col_k1, col_k2 = st.columns(2)
            with col_k1:
                strategie = st.radio("Stratégie de mise", ["Kelly", "Demi-Kelly"], horizontal=True)
            with col_k2:
                st.success(f"💸 Mise recommandée : {mise_kelly:.2f} € (Kelly) | {mise_demi:.2f} € (Demi-Kelly)")

            mise_finale = mise_kelly if strategie == "Kelly" else mise_demi

            submitted = st.form_submit_button("✅ Enregistrer")
            if submitted:
                st.session_state.bankroll -= mise_finale
                st.session_state.historique.append({
                    "ID": str(uuid.uuid4()),
                    "Match": match, "Sport": sport, "Type": type_pari, "Pari": evenement,
                    "Cote": cote, "Cote adv": 0, "Proba": round(proba * 100, 2),
                    "Marge": "~boost 8%", "Mise": round(mise_finale, 2),
                    "Stratégie": strategie, "Résultat": "Non joué",
                    "Gain": 0.0, "Global": type_global
                })
                st.success("Pari enregistré avec succès ✅")

# --- Résultat des paris et mise à jour de la bankroll ---
if st.session_state.historique:
    st.markdown("### 📝 Mettre à jour les résultats")
    for pari in st.session_state.historique:
        if pari["Résultat"] == "Non joué":
            col1, col2, col3 = st.columns([2, 2, 2])
            with col1:
                st.markdown(f"**{pari['Match']}** - {pari['Pari']} @ {pari['Cote']}")
            with col2:
                result = st.radio("Résultat", ["Non joué", "Gagné", "Perdu"], index=0, key=pari["ID"])
            with col3:
                if result != "Non joué":
                    pari["Résultat"] = result
                    if result == "Gagné":
                        gain = pari["Mise"] * pari["Cote"]
                        st.session_state.bankroll += gain
                        pari["Gain"] = round(gain, 2)
                    elif result == "Perdu":
                        pari["Gain"] = 0.0
                    st.success(f"Résultat mis à jour : {result} | Bankroll : {st.session_state.bankroll:.2f} €")
