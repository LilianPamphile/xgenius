# ✅ Logique Kelly avec affichage moderne & traitement complet des paris

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import psycopg2
import datetime

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

# --- Navigation via onglets ---
tab1, tab2 = st.tabs(["🎯 Gestion des paris", "📊 Dashboard avancé"])

# Onglet 1 : Ton app actuelle
with tab1:
    # Tout ton code actuel ici (formulaires, historique, etc.)
    st.markdown("## 🎯 Interface principale")

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
        
    # --- Choix du type de pari général ---
    st.markdown("### ➕ Ajouter un nouveau pari")
    type_global = st.radio("Quel type de pari veux-tu ajouter ?", ["Pari simple", "Pari combiné"], horizontal=True)
    
    # --- Pari Simple ---
    if type_global == "Pari simple":
        if "paris_simple_ready" not in st.session_state:
            st.session_state.paris_simple_ready = False
    
        with st.form("formulaire_pari_simple"):
            match = st.text_input("Match / Événement", placeholder="Ex : PSG - Marseille")
            
            col1, col2 = st.columns(2)
            with col1:
                sport = st.selectbox("Sport", ["Football", "Basket", "Tennis"])
            with col2:
                type_pari = st.selectbox(
                    "Type de pari",
                    ["Vainqueur", "Over/Under", "Score exact", "Gagne 1er set", "Gagne au moins un set", "Les deux équipes marquent", "Autre"]
                )
            
            pari = st.text_input("Ton pari", placeholder="Ex : PSG gagne")
            cote = st.number_input("Cote", min_value=1.01, max_value=50.0, step=0.01, format="%.2f")
            strategie = st.radio("Stratégie de mise :", ["Kelly", "Demi-Kelly"], horizontal=True)
    
            calculer = st.form_submit_button("💸 Calculer la mise recommandée")
        
        if calculer:
            if match and pari and cote >= 1.01:
                proba = proba_estimee(cote)
                bankroll = get_bankroll()
                mise_kelly = kelly(bankroll, proba, cote)
                mise_finale = mise_kelly if strategie == "Kelly" else mise_kelly / 2
    
                # Stocker pour récap et validation
                st.session_state.match_simple = match
                st.session_state.sport_simple = sport
                st.session_state.type_pari_simple = type_pari
                st.session_state.pari_simple = pari
                st.session_state.cote_simple = cote
                st.session_state.strategie_simple = strategie
                st.session_state.mise_finale_simple = mise_finale
                st.session_state.paris_simple_ready = True
    
                st.success(f"💸 Mise recommandée : {mise_finale:.2f} €")
            else:
                st.error("Merci de remplir tous les champs correctement pour calculer la mise.")
    
        # --- Partie validation après calcul
        if st.session_state.paris_simple_ready:
            st.markdown("---")
            st.markdown("### 🔍 Récapitulatif de ton pari")
            st.info(f"**{st.session_state.match_simple}** ➔ **{st.session_state.pari_simple}** @ **{st.session_state.cote_simple:.2f}** ({st.session_state.sport_simple} - {st.session_state.type_pari_simple})")
    
            with st.form("formulaire_validation_simple"):
                enregistrer = st.form_submit_button("✅ Enregistrer le pari maintenant")
                if enregistrer:
                    update_bankroll(-st.session_state.mise_finale_simple)
                    cursor.execute("""
                        INSERT INTO paris (match, sport, type, pari, cote, mise, strategie, resultat, gain, date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'Non joué', 0, %s)
                    """, (
                        st.session_state.match_simple,
                        st.session_state.sport_simple,
                        st.session_state.type_pari_simple,
                        st.session_state.pari_simple,
                        st.session_state.cote_simple,
                        round(st.session_state.mise_finale_simple, 2),
                        st.session_state.strategie_simple,
                        datetime.datetime.now()
                    ))
                    conn.commit()
                    st.success("✅ Pari enregistré et bankroll mise à jour !")
                    st.session_state.paris_simple_ready = False
                    st.rerun()
                else:
                    st.error("Merci de remplir tous les champs correctement pour calculer la mise.")

    
    # --- Pari Combiné ---
    if type_global == "Pari combiné":
        st.markdown("### ➕ Ajouter un pari combiné")
    
        with st.form("formulaire_combine_final"):
            selections = []
            for i in range(1, 4):  # Max 3 sélections
                with st.expander(f"🎯 Sélection {i}"):
                    match_c = st.text_input(f"Match {i}", key=f"match_c_{i}")
    
                    col1, col2 = st.columns(2)
                    with col1:
                        sport_c = st.selectbox(f"Sport {i}", ["Football", "Basket", "Tennis"], key=f"sport_c_{i}")
                    with col2:
                        type_pari_c = st.selectbox(
                            f"Type de pari {i}",
                            ["Vainqueur", "Over/Under", "Score exact", "Gagne 1er set", "Gagne au moins un set", "Les deux équipes marquent", "Autre"],
                            key=f"type_pari_c_{i}"
                        )
    
                    pari_c = st.text_input(f"Pari {i}", key=f"pari_c_{i}")
                    cote_c = st.number_input(f"Cote {i}", min_value=1.01, max_value=50.0, step=0.01, format="%.2f", key=f"cote_c_{i}")
    
                    if match_c and pari_c and cote_c >= 1.01:
                        selections.append({
                            "match": match_c,
                            "sport": sport_c,
                            "type_pari": type_pari_c,
                            "pari": pari_c,
                            "cote": cote_c
                        })
    
            strategie = st.radio("Stratégie de mise :", ["Kelly", "Demi-Kelly"], horizontal=True, key="strat_c")
    
            submit_combine = st.form_submit_button("💸 Calculer la mise et enregistrer")
    
        # --- Traitement après clic bouton
        if submit_combine:
            if len(selections) < 2:
                st.error("❌ Merci de remplir au moins 2 sélections valides pour un combiné.")
            else:
                # Calcul
                cotes = [s["cote"] for s in selections]
                cote_combinee = np.prod(cotes)
                proba = proba_estimee(cote_combinee)
                bankroll = get_bankroll()
                mise_kelly = kelly(bankroll, proba, cote_combinee)
                mise_finale = mise_kelly if strategie == "Kelly" else mise_kelly / 2
    
                # Insertion en base
                try:
                    pari_text = " + ".join([f"{s['match']} - {s['pari']}" for s in selections])
    
                    update_bankroll(-mise_finale)
    
                    cursor.execute("SET search_path TO public")
                    cursor.execute("""
                        INSERT INTO paris (match, sport, type, pari, cote, mise, strategie, resultat, gain, date)
                        VALUES (%s, %s, %s, %s, %s, %s, %s, 'Non joué', 0, %s)
                    """, (
                        "Combiné", "Multi", "Combiné",
                        pari_text,
                        float(round(cote_combinee, 2)),
                        float(round(mise_finale, 2)),
                        strategie,
                        datetime.datetime.now()
                    ))
                    conn.commit()
    
                    st.success(f"✅ Combiné enregistré avec succès ! Cote finale {cote_combinee:.2f} | Mise : {mise_finale:.2f} €")
                    st.rerun()
    
                except Exception as e:
                    st.error(f"Erreur lors de l'enregistrement : {e}")

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
    
    # --- Résumé du jour ---
    st.markdown("---")
    st.markdown("### 📅 Résumé de ta journée de paris")
    
    # Date actuelle
    today = datetime.date.today()
    
    # Récupération des stats du jour
    cursor.execute("""
        SELECT 
            COUNT(*) AS nb_paris,
            SUM(mise) AS total_mises,
            SUM(gain) AS total_gains,
            SUM(CASE WHEN resultat = 'Gagné' THEN 1 ELSE 0 END) AS nb_gagnes
        FROM paris
        WHERE DATE(date) = %s
    """, (today,))
    row = cursor.fetchone()
    
    nb_paris = row[0] if row[0] else 0
    total_mises = row[1] if row[1] else 0
    total_gains = row[2] if row[2] else 0
    nb_gagnes = row[3] if row[3] else 0
    
    taux_reussite = (nb_gagnes / nb_paris * 100) if nb_paris > 0 else 0
    gain_net = total_gains - total_mises
    roi_global = (gain_net / total_mises * 100) if total_mises > 0 else 0
    
    # Seuil d'alerte
    seuil_paris = 5
    
    # Définir couleurs dynamiques
    color = "green"
    roi_emoji = "📈" if roi_global >= 0 else "📉"
    taux_emoji = "🔥" if taux_reussite >= 50 else "❄️"
    
    if nb_paris > seuil_paris:
        color = "red"  # Si plus de 5 paris ➔ tout en rouge !
    
    # Affichage KPI
    col1, col2, col3, col4 = st.columns(4)
    
    col1.markdown(
        f"<div style='text-align:center; font-size:1.5rem; color:{color};'>{nb_paris}</div>"
        "<div style='text-align:center;'>Paris joués</div>",
        unsafe_allow_html=True
    )
    
    col2.markdown(
        f"<div style='text-align:center; font-size:1.5rem; color:{color};'>{gain_net:.2f} €</div>"
        "<div style='text-align:center;'>Gain net</div>",
        unsafe_allow_html=True
    )
    
    col3.markdown(
        f"<div style='text-align:center; font-size:1.5rem; color:{color};'>{roi_emoji} {roi_global:.1f}%</div>"
        "<div style='text-align:center;'>ROI global</div>",
        unsafe_allow_html=True
    )
    
    col4.markdown(
        f"<div style='text-align:center; font-size:1.5rem; color:{color};'>{taux_emoji} {taux_reussite:.1f}%</div>"
        "<div style='text-align:center;'>Taux réussite</div>",
        unsafe_allow_html=True
    )
    
    # Alerte visuelle spécifique
    if nb_paris > seuil_paris:
        st.error(f"🚨 Attention : {nb_paris} paris effectués aujourd'hui. Risque de surbetting, reste concentré !")


with tab2:
    st.markdown("## 📊 Dashboard Avancé – Suivi intelligent")
    st.caption("Analyse rapide pour comprendre ta performance et prendre de meilleures décisions 🔍")

    today = datetime.date.today()

    # --- Récupération des stats globales ---
    cursor.execute("""
        SELECT 
            COUNT(*) AS nb_paris,
            SUM(mise) AS total_mises,
            SUM(gain) AS total_gains,
            SUM(CASE WHEN resultat = 'Gagné' THEN 1 ELSE 0 END) AS nb_gagnes
        FROM paris
    """)
    row = cursor.fetchone()

    nb_paris = row[0] if row[0] else 0
    total_mises = row[1] if row[1] else 0
    total_gains = row[2] if row[2] else 0
    nb_gagnes = row[3] if row[3] else 0

    taux_reussite = (nb_gagnes / nb_paris * 100) if nb_paris > 0 else 0
    gain_net = total_gains - total_mises
    roi_global = (gain_net / total_mises * 100) if total_mises > 0 else 0

    # --- Affichage des 4 KPI principaux ---
    st.markdown("### 🎯 Résumé global")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Paris joués", nb_paris)
    col2.metric("Gain net (€)", f"{gain_net:.2f}")
    col3.metric("ROI global (%)", f"{roi_global:.1f}%")
    col4.metric("Taux de réussite", f"{taux_reussite:.1f}%")

    # --- Conseils automatiques ---
    st.markdown("---")
    st.markdown("### 🧠 Conseil personnalisé")

    if roi_global >= 5 and taux_reussite >= 55:
        st.success("🚀 Excellente performance : ROI et Taux de réussite très bons. Continue ta stratégie actuelle !")
    elif roi_global >= 0 and taux_reussite < 55:
        st.info("💡 Tes gains sont là mais ton taux de réussite est faible. Peut-être viser des cotes plus sûres.")
    elif roi_global < 0 and taux_reussite >= 55:
        st.warning("⚠️ Ton taux de réussite est bon mais tu perds de l'argent. Revois la sélection de tes paris.")
    else:
        st.error("🛑 Attention : ROI négatif et taux de réussite faible. Revoie ta méthode.")

    # --- Analyse Forces / Faiblesses : Sports ---
    st.markdown("---")
    st.markdown("### 🏆 Ton meilleur et ton pire sport")

    cursor.execute("""
        SELECT sport, SUM(gain - mise) AS gain_net
        FROM paris
        GROUP BY sport
        ORDER BY gain_net DESC
        LIMIT 1
    """)
    best_sport = cursor.fetchone()

    if best_sport:
        st.success(f"🥇 Meilleur sport : **{best_sport[0]}** (+{best_sport[1]:.2f} €)")
    else:
        st.info("Aucun sport enregistré pour l’instant.")

    cursor.execute("""
        SELECT sport, SUM(gain - mise) AS gain_net
        FROM paris
        GROUP BY sport
        ORDER BY gain_net ASC
        LIMIT 1
    """)
    worst_sport = cursor.fetchone()

    if worst_sport:
        st.error(f"🥶 Sport le moins rentable : **{worst_sport[0]}** ({worst_sport[1]:.2f} €)")

    # --- Analyse Forces / Faiblesses : Types ---
    st.markdown("---")
    st.markdown("### 🎯 Type de pari le plus et le moins rentable")

    cursor.execute("""
        SELECT type, SUM(gain - mise) AS gain_net
        FROM paris
        GROUP BY type
        ORDER BY gain_net DESC
        LIMIT 1
    """)
    best_type = cursor.fetchone()

    if best_type:
        st.success(f"✅ Meilleur type de pari : **{best_type[0]}** (+{best_type[1]:.2f} €)")
    else:
        st.info("Aucun type enregistré pour l’instant.")

    cursor.execute("""
        SELECT type, SUM(gain - mise) AS gain_net
        FROM paris
        GROUP BY type
        ORDER BY gain_net ASC
        LIMIT 1
    """)
    worst_type = cursor.fetchone()

    if worst_type:
        st.error(f"❌ Type le moins rentable : **{worst_type[0]}** ({worst_type[1]:.2f} €)")

    # --- Analyse par tranches de cotes ---
    st.markdown("---")
    st.markdown("### 📈 Meilleure tranche de cote")

    cursor.execute("SELECT cote, resultat FROM paris WHERE resultat IN ('Gagné', 'Perdu')")
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

    best_tranche = None
    best_taux = 0
    for tranche, data in tranches.items():
        if data["total"] > 0:
            taux = (data["gagnés"] / data["total"]) * 100
            if taux > best_taux:
                best_taux = taux
                best_tranche = tranche

    if best_tranche:
        st.success(f"🏅 Ta meilleure tranche de cote est **{best_tranche}** avec {best_taux:.1f}% de réussite.")

    # --- Analyse risque de Bankroll ---
    st.markdown("---")
    st.markdown("### 🛡️ Gestion du risque sur ta bankroll")

    bankroll_actuelle = get_bankroll()

    cursor.execute("SELECT mise FROM paris")
    mises = cursor.fetchall()

    grosses_mises = [mise[0] for mise in mises if mise[0] > 0.1 * bankroll_actuelle]
    pourcentage_grosses_mises = (len(grosses_mises) / len(mises) * 100) if mises else 0

    if pourcentage_grosses_mises > 20:
        st.error(f"🚨 {pourcentage_grosses_mises:.1f}% de tes mises dépassent 10% de ta bankroll actuelle ({bankroll_actuelle:.2f} €).")
    else:
        st.success(f"🛡️ Seulement {pourcentage_grosses_mises:.1f}% de grosses mises. Bonne gestion du risque.")
