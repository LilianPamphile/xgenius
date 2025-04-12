# ✅ Courbe Kelly avec probabilité calculée automatiquement et toujours optimisée

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

# Fonction Kelly globale

def kelly(bankroll, p, c):
    if c <= 1 or p <= 0 or p >= 1:
        return 0.0
    edge = (c * p - 1)
    return max(0.01, bankroll * edge / (c - 1)) if edge > 0 else 0.01

# Fonction proba automatique logique (plus stable)
def proba_auto(cote):
    return max(0.05, min(0.95, 1 / cote * 0.98))  # on prend 98% de la proba implicite pour un comportement raisonnable

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

            proba_estimee = proba_auto(cote)
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
                    "Cote": cote, "Cote adv": 0, "Proba": round(proba_estimee * 100, 2),
                    "Marge": 0, "Mise": round(mise_finale, 2),
                    "Stratégie": strategie, "Résultat": "Non joué",
                    "Global": type_global
                })
                st.success("Pari enregistré avec succès ✅")

# --- Formulaire combiné avec 3 sélections max ---
elif type_global == "Combiné":
    st.markdown("### ➕ Ajouter un événement au combiné (max 3)")
    max_combi = 3
    nb_actuels = len(st.session_state.paris_combine)

    if nb_actuels < max_combi:
        with st.form("form_combi"):
            col1, col2 = st.columns(2)
            with col1:
                match = st.text_input("Match combiné")
                sport = st.selectbox("Sport", ["Football", "Basket", "Tennis"], key="sport_combi")
                type_pari = st.selectbox("Type", ["Vainqueur", "Over/Under", "Handicap", "Score exact", "Autre"], key="type_combi")
            with col2:
                evenement = st.text_input("Pari combiné")
                cote = st.number_input("Cote événement", 1.01, step=0.01, format="%.2f")

            add_combi = st.form_submit_button("➕ Ajouter à ce combiné")
            if add_combi:
                st.session_state.paris_combine.append({
                    "Match": match, "Sport": sport, "Type": type_pari, "Pari": evenement, "Cote": cote
                })
                st.success("Événement ajouté au combiné")
    else:
        st.warning("❗ Limite de 3 sélections atteinte")

    # Résumé combiné
    if st.session_state.paris_combine:
        st.markdown("#### 🧩 Détail du combiné en cours")
        df_combi = pd.DataFrame(st.session_state.paris_combine)
        st.dataframe(df_combi)

        cotes = [e["Cote"] for e in st.session_state.paris_combine]
        cote_totale = np.prod(cotes)
        proba_comb = proba_auto(cote_totale)
        mise_k = kelly(100, proba_comb, cote_totale)

        col_a, col_b = st.columns(2)
        col_a.markdown(f"🔢 **Cote combinée : {cote_totale:.2f}**")
        col_b.markdown(f"📊 **Proba estimée automatique : {proba_comb*100:.2f}%**")

        st.success(f"💰 Mise Kelly recommandée : {mise_k:.2f} €")

        if st.button("✅ Valider le combiné"):
            st.session_state.historique.append({
                "ID": str(uuid.uuid4()),
                "Match": " + ".join([e["Match"] for e in st.session_state.paris_combine]),
                "Sport": "Combiné", "Type": "Combiné", "Pari": "+".join([e["Pari"] for e in st.session_state.paris_combine]),
                "Cote": round(cote_totale, 2), "Cote adv": 0, "Proba": round(proba_comb * 100, 2),
                "Marge": 0, "Mise": round(mise_k, 2), "Stratégie": "Kelly", "Résultat": "Non joué",
                "Global": "Combiné"
            })
            st.session_state.paris_combine = []
            st.success("✅ Pari combiné enregistré")

# --- Courbe Kelly automatique (proba logique + stable) ---
st.markdown("---")
st.subheader("📈 Courbe Kelly vs Cote (auto)")
cotes_range = np.linspace(1.01, 5.0, 100)
probas = [proba_auto(c) for c in cotes_range]
kelly_vals = [kelly(100, p, c) for p, c in zip(probas, cotes_range)]

fig, ax = plt.subplots()
ax.plot(cotes_range, kelly_vals, color='green', linewidth=2)
ax.set_xlabel("Cote")
ax.set_ylabel("Mise Kelly recommandée (€)")
ax.set_title("📊 Impact de la cote sur la mise Kelly (proba auto optimisée)")
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.caption("📌 Courbe Kelly générée à partir de proba implicite ajustée automatiquement ✨")
