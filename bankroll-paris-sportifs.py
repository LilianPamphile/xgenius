# ✅ Version corrigée avec :
# - Calcul dynamique de la mise Kelly
# - Mise à jour des résultats fiable
# - Remplacement de experimental_rerun() par rerun()

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid

st.set_page_config(page_title="Bankroll - Paris Sportifs", layout="centered")
st.title("🎯 Gestion de Bankroll - Paris Sportifs")

if "historique" not in st.session_state:
    st.session_state.historique = []

# Réinitialisation
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    if st.button("🔄 Réinitialiser l'historique"):
        st.session_state.historique = []
        st.success("Historique vidé.")

# --- Formulaire d'ajout de pari ---
with st.expander("➕ Ajouter un pari", expanded=True):
    with st.form("form_pari"):
        col1, col2 = st.columns(2)
        with col1:
            match = st.text_input("Match")
            sport = st.selectbox("Sport", ["Football", "Basket", "Tennis"])
            type_pari = st.selectbox("Type", ["Vainqueur", "Over/Under", "Handicap", "Score exact", "Autre"])
        with col2:
            evenement = st.text_input("Pari")
            cote = st.number_input("Cote", 1.01, step=0.01, format="%.2f")
            cote_adv = st.number_input("Cote adverse", 1.01, step=0.01, format="%.2f")

        proba = (1 / cote) / ((1 / cote) + (1 / cote_adv))
        marge = ((1 / cote) + (1 / cote_adv) - 1) * 100

        def kelly(bankroll, p, c):
            return max(0, bankroll * ((c * p - 1) / (c - 1))) if c > 1 else 0

        bankroll = 100.0
        mise_kelly = kelly(bankroll, proba, cote)
        mise_demi = mise_kelly / 2

        col_k1, col_k2 = st.columns(2)
        with col_k1:
            strategie = st.radio("Stratégie de mise", ["Kelly", "Demi-Kelly"], horizontal=True)
        with col_k2:
            st.markdown(f"**💸 Mise recommandée :** {mise_kelly:.2f} € (Kelly) / {mise_demi:.2f} € (Demi-Kelly)")

        mise_finale = mise_kelly if strategie == "Kelly" else mise_demi

        submitted = st.form_submit_button("✅ Enregistrer")
        if submitted:
            st.session_state.historique.append({
                "ID": str(uuid.uuid4()),
                "Match": match, "Sport": sport, "Type": type_pari, "Pari": evenement,
                "Cote": cote, "Cote adv": cote_adv, "Proba": round(proba * 100, 2),
                "Marge": round(marge, 2), "Mise": round(mise_finale, 2),
                "Stratégie": strategie, "Résultat": "Non joué"
            })
            st.success("Pari enregistré avec succès ✅")

# --- Mise à jour des résultats ---
if st.session_state.historique:
    st.markdown("---")
    st.subheader("📌 Mettre à jour les résultats")
    df_hist = pd.DataFrame(st.session_state.historique)
    df_non_joues = df_hist[df_hist["Résultat"] == "Non joué"]

    if not df_non_joues.empty:
        for _, row in df_non_joues.iterrows():
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.markdown(f"**{row['Match']} | {row['Pari']}**")
            with col2:
                if st.button("✅ Gagné", key=f"win_{row['ID']}"):
                    for p in st.session_state.historique:
                        if p["ID"] == row["ID"]:
                            p["Résultat"] = "Gagné"
                    st.rerun()
            with col3:
                if st.button("❌ Perdu", key=f"lose_{row['ID']}"):
                    for p in st.session_state.historique:
                        if p["ID"] == row["ID"]:
                            p["Résultat"] = "Perdu"
                    st.rerun()
    else:
        st.info("Aucun pari en attente de résultat.")

st.markdown("---")
st.caption("✅ Interface corrigée : mise dynamique + gestion fiable des résultats.")
