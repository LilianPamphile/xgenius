import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bankroll - Paris Sportifs", layout="centered")
st.markdown("""
<style>
    .main { background-color: #f8f9fa; }
    .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
    .stButton>button { border-radius: 8px; padding: 0.4rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

st.title("🎯 Gestion de Bankroll - Paris Sportifs")

# Init historique
if "historique" not in st.session_state:
    st.session_state.historique = []

with st.sidebar:
    st.markdown("## ⚙️ Paramètres")
    if st.button("🔄 Réinitialiser l'historique"):
        st.session_state.historique = []
        st.success("Historique vidé.")

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

        resultat = st.selectbox("Résultat", ["Non joué", "Gagné", "Perdu"])
        submit = st.form_submit_button("✅ Enregistrer")

        if submit:
            prob = (1 / cote) / ((1 / cote) + (1 / cote_adv))
            marge = ((1 / cote) + (1 / cote_adv) - 1) * 100
            def kelly(bk, p, c): return max(0, bk * ((c * p - 1) / (c - 1)))
            mise = kelly(100, prob, cote)
            st.session_state.historique.append({
                "Match": match, "Sport": sport, "Type": type_pari, "Pari": evenement,
                "Cote": cote, "Cote adv": cote_adv, "Proba": round(prob * 100, 2),
                "Marge": round(marge, 2), "Kelly": round(mise, 2), "Résultat": resultat
            })
            st.success("Pari enregistré ✅")

if st.session_state.historique:
    st.markdown("---")
    st.subheader("📊 Historique des paris")
    df = pd.DataFrame(st.session_state.historique)

    with st.expander("🎛️ Filtres"):
        col1, col2 = st.columns(2)
        with col1:
            f_sport = st.multiselect("Sport", df["Sport"].unique(), default=df["Sport"].unique())
        with col2:
            f_res = st.multiselect("Résultat", df["Résultat"].unique(), default=df["Résultat"].unique())

    df_f = df[df["Sport"].isin(f_sport) & df["Résultat"].isin(f_res)]

    df_g = df_f[df_f["Résultat"] == "Gagné"]
    df_p = df_f[df_f["Résultat"] == "Perdu"]
    bk0 = 100
    gain = (df_g["Kelly"] * (df_g["Cote"] - 1)).sum()
    loss = df_p["Kelly"].sum()
    bk_final = bk0 + gain - loss
    roi = (bk_final - bk0) / bk0 * 100 if bk0 else 0

    colr1, colr2, colr3 = st.columns([1, 1, 2])
    colr1.metric("💰 ROI", f"{roi:.2f}%")
    colr2.metric("✅ % Gagnés", f"{(len(df_g)/len(df_f)*100 if len(df_f)>0 else 0):.1f}%")
    colr3.progress(min(bk_final/200, 1.0), text=f"Bankroll: {bk_final:.2f} €")

    def badge(row):
        if row == "Gagné": return "🟢 Gagné"
        elif row == "Perdu": return "🔴 Perdu"
        else: return "⏳ Non joué"
    df_f["Résultat"] = df_f["Résultat"].apply(badge)

    st.dataframe(df_f.style.format({"Kelly": "{:.2f}"}), use_container_width=True)

    # 📈 Graphe réel d'évolution
    st.markdown("---")
    st.subheader("📉 Évolution réelle de la bankroll")
    bankrolls = [bk0]
    for _, row in df_f.iterrows():
        if "Gagné" in row["Résultat"]:
            gain = row["Kelly"] * (row["Cote"] - 1)
            bankrolls.append(bankrolls[-1] + gain)
        elif "Perdu" in row["Résultat"]:
            bankrolls.append(bankrolls[-1] - row["Kelly"])
        else:
            bankrolls.append(bankrolls[-1])

    fig1, ax1 = plt.subplots()
    ax1.plot(bankrolls)
    ax1.set_title("Bankroll réelle")
    ax1.set_xlabel("Pari")
    ax1.set_ylabel("€")
    st.pyplot(fig1)

    # 📊 Comparaison Kelly vs Flat
    st.markdown("---")
    st.subheader("📊 Kelly vs Flat (simulation)")
    p = df.iloc[-1]["Proba"] / 100
    c = df.iloc[-1]["Cote"]
    kelly_bk = [100.0]
    flat_bk = [100.0]
    flat_bet = 2.0
    ruin_count = 0
    for _ in range(100):
        mise_k = kelly(kelly_bk[-1], p, c)
        g = np.random.rand() < p
        kelly_bk.append(kelly_bk[-1] + mise_k*(c-1) if g else kelly_bk[-1]-mise_k)
        flat_bk.append(flat_bk[-1] + flat_bet*(c-1) if g else flat_bk[-1]-flat_bet)
        if kelly_bk[-1] <= 0: ruin_count += 1

    fig2, ax2 = plt.subplots()
    ax2.plot(kelly_bk, label="Kelly")
    ax2.plot(flat_bk, label="Flat", linestyle="--")
    ax2.set_title("Simulation 100 paris")
    ax2.legend()
    st.pyplot(fig2)

    # Risque de ruine estimé
    st.markdown("---")
    st.metric("☠️ Risque de ruine (Kelly)", f"{ruin_count}% sur 100 runs")
# Footer
st.markdown("---")
st.markdown("Développé avec ❤️ pour les parieurs intelligents | [GitHub](https://https://github.com/LilianPamphile) | [Contact](lilian.pamphile@gmail.com)")
