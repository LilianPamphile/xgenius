# ✅ Logique Kelly améliorée avec proba réaliste basée sur marge bookmaker

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

# Proba estimée corrigée à partir de la marge
def proba_corrigee(cote, cote_adverse):
    try:
        pi = 1 / cote
        pa = 1 / cote_adverse if cote_adverse > 0 else 0
        marge = (pi + pa - 1) if pa > 0 else 0.05  # marge par défaut si pas de cote adverse
        return max(0.01, min(0.99, pi - (marge / 2)))
    except:
        return 0.5

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
                cote_adv = st.number_input("Cote adverse (optionnel)", 1.01, step=0.01, format="%.2f")

            proba_estimee = proba_corrigee(cote, cote_adv)
            bankroll = 100.0
            mise_kelly = kelly(bankroll, proba_estimee, cote)
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
                    "Cote": cote, "Cote adv": cote_adv, "Proba": round(proba_estimee * 100, 2),
                    "Marge": round((1/cote + 1/cote_adv - 1)*100, 2) if cote_adv > 0 else "-",
                    "Mise": round(mise_finale, 2),
                    "Stratégie": strategie, "Résultat": "Non joué",
                    "Global": type_global
                })
                st.success("Pari enregistré avec succès ✅")

# --- Courbe Kelly vs Cote dynamique ---
st.markdown("---")
st.subheader("📈 Courbe Kelly vs Cote (avec correction de la proba)")
cotes_range = np.linspace(1.01, 5.0, 100)
probas = [proba_corrigee(c, 2.0) for c in cotes_range]  # on prend une cote adverse constante pour illustrer
kelly_vals = [kelly(100, p, c) for p, c in zip(probas, cotes_range)]

fig, ax = plt.subplots()
ax.plot(cotes_range, kelly_vals, color='blue', linewidth=2)
ax.set_xlabel("Cote")
ax.set_ylabel("Mise Kelly recommandée (€)")
ax.set_title("📊 Impact de la cote sur la mise Kelly (proba corrigée avec marge)")
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.caption("📌 Proba estimée = 1/cote - (marge/2), avec marge calculée selon cote adverse ✨")
