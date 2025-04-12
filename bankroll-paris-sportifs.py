# ✅ Logique Kelly avec affichage moderne & bouton mini reset bankroll + affichage dynamique des paris

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid
import psycopg2

# --- Connexion BDD ---
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# --- Bankroll helper ---
def get_bankroll():
    cursor.execute("SELECT solde FROM bankroll ORDER BY id DESC LIMIT 1")
    res = cursor.fetchone()
    return res[0] if res else 50.0

def init_bankroll():
    cursor.execute("SELECT COUNT(*) FROM bankroll")
    if cursor.fetchone()[0] == 0:
        cursor.execute("INSERT INTO bankroll (solde) VALUES (50.0)")
        conn.commit()

def update_bankroll(delta):
    solde = get_bankroll() + delta
    cursor.execute("UPDATE bankroll SET solde = %s WHERE id = (SELECT id FROM bankroll ORDER BY id DESC LIMIT 1)", (solde,))
    conn.commit()
    return solde

init_bankroll()

# --- Fonctions de calcul ---
def kelly(bankroll, p, c):
    if c <= 1 or not 0 < p < 1:
        return 0.0
    edge = (c * p - 1)
    return bankroll * edge / (c - 1) if edge > 0 else 0.0

def proba_estimee(c):
    implicite = 1 / c
    return max(0.01, min(0.99, implicite * 1.08))

# --- Interface Streamlit ---
st.set_page_config(page_title="Bankroll - Paris Sportifs", layout="centered")
st.markdown("""
<style>
    .stButton>button {
        border-radius: 8px;
        padding: 0.25rem 0.75rem;
        font-size: 0.85rem;
    }
    .mini-button button {
        background-color: #f5f5f5;
        color: #333;
        padding: 0.25rem 0.5rem;
        font-size: 0.75rem;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Gestion de Bankroll - Paris Sportifs")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")

    col_reset, col_bk = st.columns([1, 2])
    with col_reset:
        if st.button("🔁", help="Réinitialiser la bankroll", key="mini-reset"):
            cursor.execute("UPDATE bankroll SET solde = 50.0")
            conn.commit()
            st.success("Bankroll remise à 50 €")
    with col_bk:
        bankroll = get_bankroll()
        st.markdown(f"### 💰 {bankroll:.2f} €")

    if st.button("🗑️ Réinitialiser l'historique des paris"):
        cursor.execute("DELETE FROM paris")
        conn.commit()
        st.success("Historique vidé")

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

# --- Formulaire de pari ---
st.markdown("### ➕ Ajouter un pari")
with st.form("formulaire_pari"):
    match = st.text_input("Match")
    col1, col2 = st.columns(2)
    with col1:
        sport = st.selectbox("Sport", ["Football", "Basket", "Tennis"])
        type_pari = st.selectbox("Type", ["Vainqueur", "Over/Under", "Handicap", "Score exact", "Autre"])
    with col2:
        pari = st.text_input("Pari")
        cote = st.number_input("Cote", 1.01, step=0.01, format="%.2f")

    proba = proba_estimee(cote)
    bankroll = get_bankroll()
    mise_kelly = kelly(bankroll, proba, cote)
    strategie = st.radio("Stratégie", ["Kelly", "Demi-Kelly"], horizontal=True)
    mise_finale = mise_kelly if strategie == "Kelly" else mise_kelly / 2
    st.success(f"💸 Mise recommandée : {mise_finale:.2f} €")

    submitted = st.form_submit_button("✅ Enregistrer le pari")
    if submitted:
        update_bankroll(-mise_finale)
        cursor.execute("""
            INSERT INTO paris (match, sport, type, pari, cote, mise, strategie)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """, (match, sport, type_pari, pari, cote, round(mise_finale, 2), strategie))
        conn.commit()
        st.success("Pari enregistré et bankroll mise à jour ✅")

# --- Résultat des paris ---
st.markdown("---")
st.markdown("### 📝 Résultats des paris")
cursor.execute("SELECT id, match, pari, cote, mise, resultat FROM paris ORDER BY date DESC")
rows = cursor.fetchall()
for row in rows:
    pid, m, p, c, mise, res = row
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown(f"**{m}** - {p} @ {c} | Mise : {mise:.2f} €")
    with col2:
        if res == "Non joué":
            choix = st.radio("Résultat", ["Non joué", "Gagné", "Perdu"], horizontal=True, key=f"res_{pid}")
            if choix != "Non joué":
                gain = round(mise * c, 2) if choix == "Gagné" else 0.0
                update_bankroll(gain)
                cursor.execute("""
                    UPDATE paris SET resultat = %s, gain = %s WHERE id = %s
                """, (choix, gain, pid))
                conn.commit()
                st.success(f"Pari {choix} | Bankroll à jour")
        else:
            st.markdown(f"✅ Résultat : {res}")
