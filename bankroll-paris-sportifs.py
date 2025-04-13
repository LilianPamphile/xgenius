# ✅ Logique Kelly avec affichage moderne & traitement complet des paris

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2

# --- Connexion BDD ---
DATABASE_URL = "postgresql://postgres:jDDqfaqpspVDBBwsqxuaiSDNXjTxjMmP@shortline.proxy.rlwy.net:36536/railway"
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()

# Forcer le bon schéma
cursor.execute("SET search_path TO public")

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
    solde = float(get_bankroll() + delta)
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
</style>
""", unsafe_allow_html=True)

st.title("🎯 Gestion de Bankroll - Paris Sportifs")

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    if st.button("🔁 Réinitialiser la bankroll"):
        cursor.execute("UPDATE bankroll SET solde = 50.0")
        conn.commit()
        st.success("Bankroll réinitialisée à 50 €")
        st.rerun()

    st.markdown(f"### 💰 Bankroll actuelle : {get_bankroll():.2f} €")

    if st.button("🗑️ Réinitialiser l'historique des paris"):
        cursor.execute("DELETE FROM paris")
        conn.commit()
        st.success("Historique vidé")
        st.rerun()

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
type_pari_general = st.radio("Type de pari :", ["Pari simple", "Pari combiné"], horizontal=True)
if type_pari_general == "Pari simple":
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
                INSERT INTO paris (match, sport, type, pari, cote, mise, strategie, resultat, gain)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'Non joué', 0)
            """, (match, sport, type_pari, pari, cote, round(mise_finale, 2), strategie))
            conn.commit()
            st.success("Pari enregistré et bankroll mise à jour ✅")
            st.rerun()
            
elif type_pari_general == "Pari combiné":
    with st.form("formulaire_combine"):
        selections = []
        for i in range(1, 4):
            with st.expander(f"Sélection {i}"):
                match_c = st.text_input(f"Match {i}", key=f"match_c_{i}")
                pari_c = st.text_input(f"Pari {i}", key=f"pari_c_{i}")
                cote_c = st.number_input(f"Cote {i}", 1.01, step=0.01, format="%.2f", key=f"cote_c_{i}")
                if match_c and pari_c and cote_c > 1:
                    selections.append({"match": match_c, "pari": pari_c, "cote": cote_c})

        strategie = st.radio("Stratégie", ["Kelly", "Demi-Kelly"], horizontal=True, key="strat_c")

        if selections:
            cotes = [s["cote"] for s in selections]
            cote = np.prod(cotes)
            proba = proba_estimee(cote)
            bankroll = get_bankroll()
            mise_kelly = kelly(bankroll, proba, cote)
            mise_finale = mise_kelly if strategie == "Kelly" else mise_kelly / 2

            st.success(f"🎯 Cote combinée : {cote:.2f} | Mise recommandée : {mise_finale:.2f} €")

        submitted_c = st.form_submit_button("✅ Enregistrer le combiné")
        if submitted_c and selections:
            match = "Combiné"
            sport = "Multi"
            type_pari = "Combiné"
            pari = " + ".join([f"{s['match']} - {s['pari']}" for s in selections])

            update_bankroll(-float(mise_finale))
            cursor.execute("""
                INSERT INTO paris (match, sport, type, pari, cote, mise, strategie, resultat, gain)
                VALUES (%s, %s, %s, %s, %s, %s, %s, 'Non joué', 0)
            """, (match, sport, type_pari, pari, round(float(cote), 2), round(float(mise_finale), 2), strategie))
            conn.commit()
            st.success("Combiné enregistré ✅")
            st.rerun()



# --- Traitement des paris non joués ---
st.markdown("---")
st.markdown("### 🔧 Traiter les paris non joués")
cursor.execute("SELECT id, match, pari, cote, mise FROM paris WHERE resultat = 'Non joué' ORDER BY date DESC")
non_joues = cursor.fetchall()

if non_joues:
    for pid, m, p, c, mise in non_joues:
        st.markdown(f"➡️ **{m}** - {p} @ {c} | Mise : {mise:.2f} €")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("✅ Gagné", key=f"g{pid}"):
                gain = round(mise * c, 2)
                update_bankroll(gain)
                cursor.execute("UPDATE paris SET resultat = 'Gagné', gain = %s WHERE id = %s", (gain, pid))
                conn.commit()
                st.success("Pari mis à jour comme Gagné")
                st.rerun()

        with col2:
            if st.button("❌ Perdu", key=f"p{pid}"):
                cursor.execute("UPDATE paris SET resultat = 'Perdu', gain = 0 WHERE id = %s", (pid,))
                conn.commit()
                st.error("Pari mis à jour comme Perdu")
                st.rerun()

        with col3:
            if st.button("🗑️ Annuler", key=f"a{pid}"):
                update_bankroll(mise)  # On rembourse la mise
                cursor.execute("DELETE FROM paris WHERE id = %s", (pid,))
                conn.commit()
                st.warning("Pari annulé et mise remboursée")
                st.rerun()
else:
    st.info("Aucun pari à traiter.")

# --- Top Gagnés ---
st.markdown("---")
st.markdown("### 🏆 Top 3 gains")
cursor.execute("SELECT match, pari, gain FROM paris WHERE resultat = 'Gagné' ORDER BY gain DESC LIMIT 3")
gagnes = cursor.fetchall()
for m, p, g in gagnes:
    st.markdown(f"✅ **{m}** - {p} : **+{g:.2f} €**")

# --- Top Pertes ---
st.markdown("### ❌ Top 3 pertes")
cursor.execute("SELECT match, pari, mise FROM paris WHERE resultat = 'Perdu' ORDER BY mise DESC LIMIT 3")
perdus = cursor.fetchall()
for m, p, m_ in perdus:
    st.markdown(f"❌ **{m}** - {p} : **-{m_:.2f} €**")


# --- Dashboard avancé ---
st.markdown("---")
st.markdown("## 📊 Dashboard avancé – Aide à la décision")
st.caption("Analyse stratégique pour améliorer tes performances de paris")

kpi_choisi = st.selectbox("📌 Sélectionne un KPI à analyser :", [
    "1. ROI par sport",
    "2. ROI par type de pari",
    "3. % réussite par tranche de cote",
    "4. Simples vs combinés",
    "5. Taux de réussite par type de paris",
    "6. Cote moyenne des paris gagnés/perdus",
    "7. Gain net par sport",
    "8. Répartition des mises par type",
    "9. Taux de réussite par niveau de mise",
    "10. Taux de réussite par sport"
])

# --- KPI 1 : ROI par sport ---
if kpi_choisi == "1. ROI par sport":
    cursor.execute("""
        SELECT sport, COUNT(*) as nb, SUM(mise) as mises, SUM(gain) as gains
        FROM paris
        GROUP BY sport
    """)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=["Sport", "Nb Paris", "Mises", "Gains"])
    df["ROI (%)"] = ((df["Gains"] - df["Mises"]) / df["Mises"]) * 100
    st.dataframe(df.sort_values("ROI (%)", ascending=False).round(2))

# --- KPI 2 : ROI par type de pari ---
elif kpi_choisi == "2. ROI par type de pari":
    cursor.execute("""
        SELECT type, COUNT(*) as nb, SUM(mise) as mises, SUM(gain) as gains
        FROM paris
        GROUP BY type
    """)
    rows = cursor.fetchall()
    df = pd.DataFrame(rows, columns=["Type", "Nb Paris", "Mises", "Gains"])
    df["ROI (%)"] = ((df["Gains"] - df["Mises"]) / df["Mises"]) * 100
    st.dataframe(df.sort_values("ROI (%)", ascending=False).round(2))

# --- KPI 3 : % réussite par tranche de cote ---
elif kpi_choisi == "3. % réussite par tranche de cote":
    cursor.execute("""
        SELECT cote, resultat FROM paris WHERE resultat IN ('Gagné', 'Perdu')
    """)
    rows = cursor.fetchall()
    tranches = {
        "1.01–1.49": {"total": 0, "gagnés": 0},
        "1.50–1.99": {"total": 0, "gagnés": 0},
        "2.00–2.49": {"total": 0, "gagnés": 0},
        "2.50–2.99": {"total": 0, "gagnés": 0},
        "3.00+": {"total": 0, "gagnés": 0}
    }
    for cote, res in rows:
        if cote < 1.50:
            key = "1.01–1.49"
        elif cote < 2.00:
            key = "1.50–1.99"
        elif cote < 2.50:
            key = "2.00–2.49"
        elif cote < 3.00:
            key = "2.50–2.99"
        else:
            key = "3.00+"
        tranches[key]["total"] += 1
        if res == "Gagné":
            tranches[key]["gagnés"] += 1
    df = pd.DataFrame([
        [k, v["total"], v["gagnés"], (v["gagnés"] / v["total"] * 100) if v["total"] else 0]
        for k, v in tranches.items()
    ], columns=["Tranche de cote", "Nb Paris", "Gagnés", "Taux de réussite (%)"])
    st.bar_chart(df.set_index("Tranche de cote")["Taux de réussite (%)"])

# Tu veux que je code les 3 suivants (#4, #5, #6) dans la foulée ?
