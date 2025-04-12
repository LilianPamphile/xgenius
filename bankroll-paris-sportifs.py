# ✅ Logique Kelly avec proba estimée basée uniquement sur la cote (proba implicite corrigée)

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid

st.set_page_config(page_title="Bankroll - Paris Sportifs", layout="centered")
st.title("🎯 Gestion de Bankroll - Paris Sportifs")

if "historique" not in st.session_state:
    st.session_state.historique = []
if "paris_combine" not in st.session_state:
    st.session_state.paris_combine = []

# Fonction Kelly optimale
def kelly(bankroll, p, c):
    if c <= 1 or not 0 < p < 1:
        return 0.0
    edge = (c * p - 1)
    return bankroll * edge / (c - 1) if edge > 0 else 0.0

# Proba estimée uniquement en fonction de la cote (corrigée de la marge type 5%)
def proba_estimee(c):
    return max(0.01, min(0.99, (1 / c) - 0.025))

# Réinitialisation
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    if st.button("🔄 Réinitialiser l'historique"):
        st.session_state.historique = []
        st.success("Historique vidé.")
    if st.button("🧹 Réinitialiser combiné"):
        st.session_state.paris_combine = []

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
            bankroll = 100.0
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
                st.session_state.historique.append({
                    "ID": str(uuid.uuid4()),
                    "Match": match, "Sport": sport, "Type": type_pari, "Pari": evenement,
                    "Cote": cote, "Cote adv": 0, "Proba": round(proba * 100, 2),
                    "Marge": "~2.5%", "Mise": round(mise_finale, 2),
                    "Stratégie": strategie, "Résultat": "Non joué",
                    "Global": type_global
                })
                st.success("Pari enregistré avec succès ✅")

# --- Courbe Kelly vs Cote ---
st.markdown("---")
st.subheader("📈 Courbe Kelly vs Cote (proba implicite corrigée)")
cotes_range = np.linspace(1.01, 5.0, 100)
probas = [proba_estimee(c) for c in cotes_range]
kelly_vals = [kelly(100, p, c) for p, c in zip(probas, cotes_range)]

fig, ax = plt.subplots()
ax.plot(cotes_range, kelly_vals, color='blue', linewidth=2)
ax.set_xlabel("Cote")
ax.set_ylabel("Mise Kelly recommandée (€)")
ax.set_title("📊 Impact de la cote sur la mise Kelly (proba implicite corrigée de 2.5%)")
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.caption("📌 Proba estimée = 1 / cote - 2.5% pour simuler une marge bookmaker ✨")
