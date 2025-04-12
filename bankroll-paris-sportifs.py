# ✅ Ajout d'une logique de paris combinés avec interface multi-événements

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
                    "Stratégie": strategie, "Résultat": "Non joué",
                    "Global": type_global
                })
                st.success("Pari enregistré avec succès ✅")

# --- Formulaire combiné ---
elif type_global == "Combiné":
    st.markdown("### ➕ Ajouter un événement au combiné")
    with st.form("form_combi"):
        col1, col2 = st.columns(2)
        with col1:
            match = st.text_input("Match combiné")
            evenement = st.text_input("Événement combiné")
        with col2:
            cote = st.number_input("Cote événement", 1.01, step=0.01, format="%.2f")

        ajouter = st.form_submit_button("➕ Ajouter à ce combiné")
        if ajouter:
            st.session_state.paris_combine.append({
                "Match": match, "Pari": evenement, "Cote": cote
            })
            st.success("Événement ajouté au combiné")

    # Affichage combiné actuel
    if st.session_state.paris_combine:
        st.markdown("#### 🧩 Détail du combiné en cours")
        df_combi = pd.DataFrame(st.session_state.paris_combine)
        st.dataframe(df_combi)

        # Calculs combinés
        cote_totale = np.prod([e["Cote"] for e in st.session_state.paris_combine])
        st.markdown(f"**🎯 Cote combinée totale : {cote_totale:.2f}**")

        proba_comb = 1 / cote_totale
        bankroll = 100.0
        mise_k = kelly(bankroll, proba_comb, cote_totale)

        colk1, colk2 = st.columns(2)
        with colk1:
            st.markdown(f"💡 **Proba combinée : {proba_comb*100:.2f}%**")
        with colk2:
            st.markdown(f"💰 **Mise Kelly recommandée : {mise_k:.2f} €**")

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
            st.success("Pari combiné enregistré ✅")

# --- Affichage courbe Kelly ---
st.markdown("---")
st.subheader("📈 Courbe Kelly vs Cote (proba estimée constante)")
proba_test = st.slider("Probabilité estimée (%) pour la courbe", min_value=30, max_value=90, value=60)
proba_test /= 100
cotes_range = np.linspace(1.01, 5.0, 100)

def kelly(bankroll, p, c):
    return max(0, bankroll * ((c * p - 1) / (c - 1))) if c > 1 else 0

kelly_vals = [kelly(100, proba_test, c) for c in cotes_range]
fig, ax = plt.subplots()
ax.plot(cotes_range, kelly_vals, label=f"Mise Kelly (proba {int(proba_test*100)}%)")
ax.set_xlabel("Cote")
ax.set_ylabel("Mise en € (pour 100€ de BK)")
ax.set_title("Évolution de la mise Kelly en fonction de la cote")
ax.grid(True)
st.pyplot(fig)

st.markdown("---")
st.caption("✅ Gestion avancée des combinés et visualisation Kelly active")
