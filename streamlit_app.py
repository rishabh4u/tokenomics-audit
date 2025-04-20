# ----------------------- PART 1 -----------------------
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from fpdf import FPDF
import tempfile
import os
import random
from openai import OpenAI
import base64
from io import BytesIO

# --- Branding & Config ---
st.set_page_config(page_title="Tokenomics Audit AI", layout="wide")
st.image("logo.png", width=200)
st.markdown("""
    <style>
    .stApp { background-color: #F7F8FA; }
    .title { font-size: 32px; font-weight: bold; color: #001f3f; margin-bottom: 10px; }
    </style>
""", unsafe_allow_html=True)
st.markdown("<div class='title'>Tokenomics Audit AI Dashboard By TDeFi</div>", unsafe_allow_html=True)

# --- Step Tracker ---
if "step" not in st.session_state:
    st.session_state.step = 1

# --- Step-by-Step Inputs ---
if st.session_state.step == 1:
    name = st.text_input("What's the Token Name")
    if st.button("Next") and name:
        st.session_state.project_name = name
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 2:
    supply = st.number_input("What is Total Token Supply", min_value=1, value=1_000_000_000)
    if st.button("Next"):
        st.session_state.total_supply = supply
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 3:
    price = st.number_input("What is the Token Price at TGE (in USD)", min_value=0.00001, value=0.05, step=0.01, format="%.5f")
    if st.button("Next"):
        st.session_state.price = price
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 4:
    liq = st.number_input("How much Liquidity Fund (USD)is being allocated", min_value=0.0, value=500000.0, step=10000.0)
    if st.button("Next"):
        st.session_state.liq = liq
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 5:
    category = st.selectbox("Select the company category", ["Gaming", "DeFi", "NFT", "Infrastructure"])
    if st.button("Next"):
        st.session_state.ptype = category
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 6:
    ubase = st.number_input("What is the Current User Base", min_value=0, value=10000)
    if st.button("Next"):
        st.session_state.ubase = ubase
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 7:
    growth = st.number_input("What is the Monthly User Growth Rate (%)", min_value=0.0, value=10.0)
    if st.button("Next"):
        st.session_state.ugrow = growth / 100
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 8:
    burn = st.number_input("What is the Monthly Burn Rate (USD)", min_value=0.0, value=50000.0)
    if st.button("Next"):
        st.session_state.mburn = burn
        st.session_state.step += 1
        st.rerun()

elif st.session_state.step == 9:
    rev = st.number_input("What is the current ARR (USD)", min_value=0.0, value=1_000_000.0, step=10000.0)
    if st.button("Next"):
        st.session_state.rev = rev
        st.session_state.step += 1
        st.rerun()

# --- Token Pool Setup ---
elif st.session_state.step == 10:
    st.markdown("### 🎯 Token Allocation & Pool Structure")
    num_pools = st.number_input("How many token pools would you like to define?", 1, 30, 9)

    default_pools = [
        ("Seed Sale", 5, "VC"), ("Private Sale", 15, "VC"), ("Public Sale", 10, "Community"),
        ("Team", 20, "Team"), ("Advisors", 5, "Team"), ("Ecosystem", 20, "Community"),
        ("Liquidity", 10, "Liquidity"), ("Staking", 5, "Community"), ("Treasury", 10, "Team")
    ]

    pool_names, pool_percents, pool_tags = [], [], []
    cliffs, vestings, tge_unlocks, sellables = [], [], [], []

    for i in range(num_pools):
        cols = st.columns([2, 1, 1, 1, 1, 1, 1])
        name = cols[0].text_input(f"Pool Name {i+1}", value=default_pools[i][0] if i < len(default_pools) else f"Pool {i+1}", key=f"name_{i}")
        perc = cols[1].number_input("%", min_value=0.0, max_value=100.0, value=float(default_pools[i][1]) if i < len(default_pools) else 0.0, key=f"perc_{i}")
        tag = cols[2].selectbox("Tag", ["VC", "Community", "Team", "Liquidity"], index=["VC", "Community", "Team", "Liquidity"].index(default_pools[i][2]) if i < len(default_pools) else 0, key=f"tag_{i}")
        cliff = cols[3].number_input("Cliff (months)", 0, 48, 0, key=f"cliff_{i}")
        vest = cols[4].number_input("Vest (months)", 0, 72, 12, key=f"vest_{i}")
        tge = cols[5].number_input("TGE %", 0.0, 100.0, 0.0, key=f"tge_{i}")
        sell = cols[6].radio("Sell@TGE", ["Yes", "No"], horizontal=True, key=f"sell_{i}")

        pool_names.append(name)
        pool_percents.append(perc)
        pool_tags.append(tag)
        cliffs.append(cliff)
        vestings.append(vest)
        tge_unlocks.append(tge)
        sellables.append(True if sell == "Yes" else False)

    if sum(pool_percents) != 100:
        st.warning(f"🚨 Total Allocation is {sum(pool_percents)}%. Please ensure it adds up to 100%.")
    else:
        if st.button("✅ Submit Pool Setup"):
            st.session_state.pool_names = pool_names
            st.session_state.pool_percents = pool_percents
            st.session_state.pool_tags = pool_tags
            st.session_state.cliffs = cliffs
            st.session_state.vestings = vestings
            st.session_state.tge_unlocks = tge_unlocks
            st.session_state.sellables = sellables
            st.session_state.step = "done"
            st.rerun()

if st.session_state.step == "done":
    st.success("✅ All project data captured. Proceeding to audit analysis...")
# --- Part 2: Audit Calculations & Charts ---

import numpy as np

if st.session_state.step == "done":
    st.header("📊 Tokenomics Audit Dashboard")
    # --- Show Input Summary as Table ---
    st.markdown("### 📋 Project Input Summary")

    input_data = {
    "Field": [
        "Project Name",
        "Total Token Supply",
        "Token Price at TGE (USD)",
        "Liquidity Fund (USD)",
        "Project Category",
        "Current User Base",
        "Monthly Growth Rate (%)",
        "Monthly Incentive Spend (USD)",
        "Annual Revenue (USD)"
    ],
    "Value": [
        str(st.session_state.project_name),
        f"{st.session_state.total_supply:,}",
        f"${st.session_state.price:.5f}",
        f"${st.session_state.liq:,.0f}",
        str(st.session_state.ptype),
        f"{st.session_state.ubase:,}",
        f"{st.session_state.ugrow * 100:.2f}%",
        f"${st.session_state.mburn:,.0f}",
        f"${st.session_state.rev:,.0f}"
    ]
}

    df_inputs = pd.DataFrame(input_data)
    st.table(df_inputs)
    st.markdown("### 🧱 Token Pool Setup")

    df_pools = pd.DataFrame({
        "Pool Name": st.session_state.pool_names,
        "Allocation %": st.session_state.pool_percents,
        "Tag": st.session_state.pool_tags,
        "Cliff (mo)": st.session_state.cliffs,
        "Vest (mo)": st.session_state.vestings,
        "TGE Unlock %": st.session_state.tge_unlocks,
        "Sell @ TGE": ["Yes" if s else "No" for s in st.session_state.sellables]
})
    st.dataframe(df_pools, use_container_width=True)
    project_name = st.session_state.project_name
    total_supply_tokens = st.session_state.total_supply
    tge_price = st.session_state.price
    liquidity_fund = st.session_state.liq
    category = st.session_state.ptype
    user_base = st.session_state.ubase
    growth_rate = st.session_state.ugrow
    monthly_burn = st.session_state.mburn
    revenue = st.session_state.rev

    pool_names = st.session_state.pool_names
    pool_percents = st.session_state.pool_percents
    pool_tags = st.session_state.pool_tags
    cliffs = st.session_state.cliffs
    vestings = st.session_state.vestings
    tge_unlocks = st.session_state.tge_unlocks
    sellables = st.session_state.sellables

    # --- Token Release Modeling ---
    months = list(range(72))
    df = pd.DataFrame({"Month": months})
    df["Monthly Release %"] = 0
    df["Cumulative %"] = 0

    for i in range(len(pool_names)):
        tge_amount = pool_percents[i] * tge_unlocks[i] / 100
        monthly_amount = (pool_percents[i] - tge_amount) / vestings[i] if vestings[i] > 0 else 0
        release = [0] * 72
        release[0] += tge_amount
        for m in range(cliffs[i] + 1, cliffs[i] + vestings[i] + 1):
            if m < 72:
                release[m] += monthly_amount
        df[pool_names[i]] = release
        df["Monthly Release %"] += df[pool_names[i]]
    df["Cumulative %"] = df["Monthly Release %"].cumsum()

    # --- Audit Metric Calculations ---
    def inflation_guard(df):
        return [round((df.loc[min((y)*12,71), "Cumulative %"] - df.loc[min((y-1)*12,71), "Cumulative %"]) / max(df.loc[min((y-1)*12,71), "Cumulative %"], 1) * 100, 2) for y in range(1, 6)]

    def shock_stopper(df):
        return {
            "0–5%": df[df["Monthly Release %"] <= 5].shape[0],
            "5–10%": df[(df["Monthly Release %"] > 5) & (df["Monthly Release %"] <= 10)].shape[0],
            "10–15%": df[(df["Monthly Release %"] > 10) & (df["Monthly Release %"] <= 15)].shape[0],
            "15%+": df[df["Monthly Release %"] > 15].shape[0]
        }

    def governance_hhi(percs):
        shares = [p / 100 for p in percs]
        return round(sum([(s * 100)**2 for s in shares]) / 10000, 3)

    def liquidity_shield():
        tge_percent = sum([pool_percents[i] * (tge_unlocks[i] / 100) for i in range(len(pool_names))])
        tokens_at_tge = total_supply_tokens * (tge_percent / 100)
        market_cap_at_tge = tokens_at_tge * tge_price
        return round(liquidity_fund / market_cap_at_tge, 2) if market_cap_at_tge > 0 else 0

    def lockup_ratio():
        return round(sum([1 for c in cliffs if c >= 12]) / len(cliffs), 2)

    def vc_dominance():
        return round(sum([pool_percents[i] for i in range(len(pool_names)) if pool_tags[i] == "VC"]) / 100, 2)

    def community_index():
        return round(sum([pool_percents[i] for i in range(len(pool_names)) if pool_tags[i] == "Community"]) / 100, 2)

    def vesting_coverage():
        total_tokens_released = total_supply_tokens * df["Cumulative %"].iloc[-1] / 100
        usd_available = total_tokens_released * tge_price
        return round(usd_available / (monthly_burn * 12), 2)

    def sustainability_ratio():
        total_tokens = total_supply_tokens * df["Cumulative %"].iloc[-1] / 100
        emissions = total_tokens * tge_price
        return round(revenue / emissions, 2) if emissions else 0

    def nvir():
        emission_value = total_supply_tokens * df["Cumulative %"].iloc[-1] / 100 * tge_price
        return round(revenue / emission_value, 2) if emission_value else 0

    def emission_taper():
        first_12 = df.loc[0:11, "Monthly Release %"].sum()
        last_12 = df.loc[60:71, "Monthly Release %"].sum()
        return round(first_12 / last_12, 2) if last_12 else 0

    # --- Monte Carlo Simulation ---
    def monte_sim():
        if category == "Gaming":
            arpu = 76 * 12 * 0.03
        elif category == "DeFi":
            arpu = 3300 * 0.03
        elif category == "NFT":
            arpu = 59 * 0.03
        else:
            arpu = 50

        monte = []
        for _ in range(100):
            growth = [user_base]
            for _ in range(12):
                growth.append(growth[-1] * (1 + random.uniform(growth_rate - 0.02, growth_rate + 0.02)))
            total_buy_pressure = sum([g * arpu for g in growth])
            total_release_value = total_supply_tokens * df.loc[0:11, "Monthly Release %"].sum() / 100 * tge_price
            resilience_score = round(total_buy_pressure / total_release_value, 2) if total_release_value else 0
            monte.append(resilience_score)

        monte_median = round(np.median(monte), 2)
        return monte, monte_median

    # Run Metrics
    inflation = inflation_guard(df)
    shock = shock_stopper(df)
    hhi_score = governance_hhi(pool_percents)
    shield = liquidity_shield()
    lock_ratio = lockup_ratio()
    vc_dom = vc_dominance()
    comm_idx = community_index()
    vcp = vesting_coverage()
    sustain_ratio = sustainability_ratio()
    nvir_score = nvir()
    taper = emission_taper()
    monte, monte_median = monte_sim()
    avg_resilience = round(np.mean(monte), 2)


    with st.expander("🎲 Monte Carlo Survivability (12-Month Simulation)"):
        st.markdown("""
    **What is this?**  
    This simulation estimates whether the projected demand (user growth × ARPU) can absorb the token supply released over the next 12 months.

    **How does it work?**  
    We simulate 100 market conditions where user growth fluctuates monthly. For each scenario, we calculate:

    - **Buy Pressure**: Value generated from projected users × average revenue per user (ARPU).
    - **Token Release Value**: Value of tokens unlocked in the first 12 months at TGE price.

    Then we compute the **Resilience Score** = Buy Pressure ÷ Token Release Value.

    **How to interpret it?**
    - A **score > 1** means demand can potentially absorb token emissions — positive signal.
    - A **score < 1** indicates a mismatch — emissions could outweigh value creation, causing price pressure.

    The **histogram** below shows distribution of all 100 scores:
    """)

    fig_mc, ax3 = plt.subplots(figsize=(5, 3))
    ax3.hist(monte, bins=min(20, len(set(monte))), color='skyblue', edgecolor='black')
    ax3.axvline(monte_median, color='green', linestyle='--', label=f'Median: {monte_median}x')
    ax3.set_title("Monte Carlo – Survivability Distribution")
    ax3.set_xlabel("Resilience Score")
    ax3.set_ylabel("Number of Simulations")
    ax3.legend()
    st.pyplot(fig_mc)
    st.markdown(f"**Median Resilience Score:** `{monte_median}x` — interpreted as median scenario out of 100 runs.")
# 🧠 Tokenomics Audit AI – GPT Summary & PDF Report (Part 3)
if st.session_state.step == "done":
    def calculate_risk_rating(inflation, shock, hhi_score, shield):
        red_flags = 0
        if inflation[0] > 300: red_flags += 1
        if inflation[1] > 80: red_flags += 1
        if shock["10–15%"] + shock["15%+"] > 7: red_flags += 1
        if hhi_score > 0.25: red_flags += 1
        if shield < 1.0: red_flags += 1
        return "Low" if red_flags <= 1 else "Medium" if red_flags == 2 else "High"

    if st.button("Show me the Audit Report"):
        client = OpenAI(api_key=st.secrets["openai"]["api_key"])
        risk = calculate_risk_rating(inflation, shock, hhi_score, shield)

        col1, col2 = st.columns(2)
        with col1:
            fig_inf, ax = plt.subplots(figsize=(5, 3))
            ax.plot(range(1, 6), inflation, marker="o")
            ax.set_title("Inflation Guard")
            st.pyplot(fig_inf)
        with col2:
            fig_shock, ax2 = plt.subplots(figsize=(5, 3))
            ax2.bar(shock.keys(), shock.values(), color=["green", "orange", "red", "darkred"])
            ax2.set_title("Shock Stopper")
            st.pyplot(fig_shock)

        prompt = f"""
Tokenization is rapidly transforming how value is created and distributed across real-world assets, gaming, and infrastructure networks. In this context, tokenomics isn't just a technical detail—it's an investment logic layer that needs to mirror the economic dynamics of the underlying business.

Act like a Senior Tokenomics Analyst who has reviewed more than 150 token models and published institutional-grade audit reports.

You are writing an institutional-grade tokenomics audit report for the following crypto project.

🔹 Project Name: {project_name}  
🔹 Total Supply: {total_supply_tokens:,}  
🔹 Token Price at TGE: ${tge_price}  
🔹 Liquidity Fund: ${liquidity_fund}  
🔹 Category: {category}  
🔹 User Base: {user_base}  
🔹 Monthly Growth Rate: {growth_rate*100:.2f}%  
🔹 Monthly Incentive Spend: ${monthly_burn}  
🔹 Annual Revenue: ${revenue}  

Tokenomics Metrics:  
- Year-on-Year Inflation: {inflation}  
- Supply Shock Months: {shock}  
- Governance HHI Index: {hhi_score}  
- Liquidity Shield Ratio: {shield}  
- Lockup Ratio: {lock_ratio}  
- VC Dominance: {vc_dom}  
- Community Control Index: {comm_idx}  
- Vesting Coverage Period (years): {vcp}  
- Sustainability Ratio: {sustain_ratio}  
- NVIR: {nvir_score}  
- Emission Taper Score: {taper}  
- Median Survivability (Monte Carlo): {monte_median}x  
- Risk Rating: {risk}  

🎯 Report Instructions:  
1. Start with a paragraph on the growth of tokenised markets with statistics in terms of growth rate and number of tokens launched every year since 2015
2. Talk about the importance of tokenomics and aligning emissions with business logic.
3. For each metric:
   - Explain what it means and why it matters.
   - Analyze provided values and implications.
   - Use clean bold headings, no markdown characters.
   - Add input values as tables under each heading.
   - Include inflation and shock charts where applicable.
4. Finish with 'Token Design by TDeFi' paragraph where the following has to be copied as it is 
Token engineering is not about distribution schedules or supply caps alone - it is the structured discipline of aligning incentives, behavior, and long-term value creation. At its core, it is the science of designing economic systems where every stakeholder, from early investors to late-stage contributors, is guided by aligned motivations. A well-engineered token system creates harmony between product usage, network growth, and token demand. Poor token design, on the other hand, often leads to value leakage, unsustainable emissions, and ultimately, failure of both the product and its economy. At TDeFi, we don’t treat tokenomics as an afterthought. We build token models that function as economic engines - driven by utility, governed by logic, and sustained by real-world adoption.
"""

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a tokenomics analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        st.session_state["gpt_summary"] = response.choices[0].message.content
        st.markdown(st.session_state["gpt_summary"])


