import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from fpdf import FPDF
import tempfile
import os
import re
from openai import OpenAI
import random

# App setup
st.set_page_config(page_title="Tokenomics Audit AI", layout="centered")
st.title("🧠 Tokenomics Audit AI Tool by TDeFi")

# Project Inputs
project_name = st.text_input("Enter Project Name")
total_supply_tokens = st.number_input("Total Token Supply", min_value=1, value=1_000_000_000)
tge_price = st.number_input("Token Price at TGE (USD)", min_value=0.00001, value=0.05, step=0.01, format="%.5f")
liquidity_fund = st.number_input("Liquidity Fund (USD)", min_value=0.0, value=500000.0, step=10000.0)
project_type = st.selectbox("Select Project Category", ["Gaming", "DeFi", "NFT", "Infrastructure"])
user_base = st.number_input("Current User Base", min_value=0, value=10000)
user_growth_rate = st.number_input("Expected Monthly User Growth Rate (%)", min_value=0.0, value=10.0) / 100
monthly_burn = st.number_input("Monthly Incentive Spend (USD)", min_value=0.0, value=50000.0, step=1000.0)
revenue = st.number_input("Annual Revenue (USD)", min_value=0.0, value=1000000.0, step=10000.0)

# Token Allocation
st.header("Step 1: Token Allocation (%)")
pool_names = st.text_area("Enter pool names (comma-separated)", value="Private Sale, Public Sale, Team, Ecosystem, Advisors, Listing, Airdrop, Staking")
pools = [p.strip() for p in pool_names.split(",") if p.strip()]
allocations, tags = {}, {}
total_alloc = 0
category_options = ["VC", "Community", "Team", "Liquidity", "Advisor", "Other"]

for pool in pools:
    allocations[pool] = st.number_input(f"{pool} Allocation (%)", 0.0, 100.0, 0.0, step=1.0)
    tags[pool] = st.selectbox(f"Tag for {pool}", category_options, key=f"tag_{pool}")
    total_alloc += allocations[pool]

if total_alloc != 100:
    st.warning(f"Allocation must total 100%. Current: {total_alloc:.2f}%")
    st.stop()

# Vesting Input or Excel Upload
st.header("Step 2: Vesting Schedule")
input_mode = st.radio("How would you like to input the token release schedule?", ["Manual Entry", "Upload Excel"])
vesting_schedule = {}
df = pd.DataFrame()

if input_mode == "Manual Entry":
    months = list(range(72))
    df = pd.DataFrame({"Month": months})
    df["Monthly Release %"] = 0
    df["Cumulative %"] = 0

    for pool in pools:
        with st.expander(f"{pool} Vesting"):
            cliff = st.number_input(f"{pool} Cliff (months)", 0, 48, 0)
            vesting = st.number_input(f"{pool} Vesting Duration (months)", 0, 72, 12)
            tge = st.number_input(f"{pool} TGE Unlock (%)", 0.0, 100.0, 0.0)
            sellable = st.radio(f"{pool} Sellable at TGE?", ["Yes", "No"], key=pool) == "Yes"
            vesting_schedule[pool] = {"cliff": cliff, "vesting": vesting, "tge": tge, "sellable": sellable}

        # Build release schedule
        tge_amount = allocations[pool] * vesting_schedule[pool]["tge"] / 100
        cliff = vesting_schedule[pool]["cliff"]
        vest = vesting_schedule[pool]["vesting"]
        monthly_amount = (allocations[pool] - tge_amount) / vest if vest > 0 else 0
        release = [0] * 72
        release[0] += tge_amount
        for m in range(cliff + 1, cliff + vest + 1):
            if m < 72:
                release[m] += monthly_amount
        df[pool] = release
        df["Monthly Release %"] += df[pool]

    df["Cumulative %"] = df["Monthly Release %"].cumsum()

elif input_mode == "Upload Excel":
    st.info("Upload an Excel file with rows = months and columns = pool names. Values = % released in that month.")
    uploaded_file = st.file_uploader("Upload your Excel (.xlsx) file", type=["xlsx"])
    if uploaded_file:
        df = pd.read_excel(uploaded_file)
        if "Month" not in df.columns:
            st.error("Excel must contain a 'Month' column.")
            st.stop()
        df["Monthly Release %"] = df.drop(columns=["Month"]).sum(axis=1)
        df["Cumulative %"] = df["Monthly Release %"].cumsum()

st.success("✅ Vesting schedule processed successfully.")
# --- Audit Metrics ---

def inflation_guard(df):
    inflation = []
    for year in range(1, 6):
        prev = df.loc[(year - 1) * 12, "Cumulative %"] if (year - 1) * 12 < len(df) else 0
        curr = df.loc[year * 12, "Cumulative %"] if year * 12 < len(df) else df["Cumulative %"].iloc[-1]
        rate = ((curr - prev) / max(prev, 1)) * 100
        inflation.append(round(rate, 2))
    return inflation

def shock_stopper(df):
    return {
        "0–5%": df[(df["Monthly Release %"] <= 5)].shape[0],
        "5–10%": df[(df["Monthly Release %"] > 5) & (df["Monthly Release %"] <= 10)].shape[0],
        "10–15%": df[(df["Monthly Release %"] > 10) & (df["Monthly Release %"] <= 15)].shape[0],
        "15%+": df[df["Monthly Release %"] > 15].shape[0]
    }

def governance_hhi(allocations):
    shares = [v / 100 for v in allocations.values()]
    return round(sum([(s * 100) ** 2 for s in shares]) / 10000, 3)

def liquidity_shield(total_supply, tge_price, liquidity_usd, allocations, vesting):
    tge_percent = sum([allocations[p] * (vesting[p]["tge"] / 100) for p in allocations if p in vesting])
    tokens_at_tge = total_supply * (tge_percent / 100)
    market_cap_at_tge = tokens_at_tge * tge_price
    return round(liquidity_usd / market_cap_at_tge, 2) if market_cap_at_tge > 0 else 0

def lockup_ratio(vesting):
    locked = sum([1 for p in vesting if vesting[p]["cliff"] >= 12])
    return round(locked / len(vesting), 2)

def vc_dominance(allocations, tags):
    return round(sum([allocations[p] for p in allocations if tags.get(p) == "VC"]) / 100, 2)

def community_index(allocations, tags):
    return round(sum([allocations[p] for p in allocations if tags.get(p) == "Community"]) / 100, 2)

def emission_taper(df):
    first_12m = df.loc[0:11, "Monthly Release %"].sum()
    last_12m = df.loc[-12:, "Monthly Release %"].sum()
    return round(first_12m / last_12m if last_12m else 0, 2)

# --- Monte Carlo Simulation for Survivability ---
def monte_carlo_simulation(df, user_base, growth, months, category):
    if category == "Gaming":
        arpu = 76 * 12 * 0.03
    elif category == "DeFi":
        arpu = 3300 * 0.03
    elif category == "NFT":
        arpu = 59 * 0.03
    else:
        arpu = 50

    simulations = []
    for _ in range(100):
        price = 1.0
        for m in range(min(months, len(df))):
            users = user_base * ((1 + growth) ** (m / 12))
            buy_pressure = users * arpu
            sell_pressure = df.loc[m, "Monthly Release %"] * total_supply_tokens * tge_price
            price_change = (buy_pressure - sell_pressure) / max(sell_pressure, 1)
            price *= (1 + price_change * 0.01)
        simulations.append(price)
    return simulations

# --- Game Theory Audit ---
def game_theory_audit(hhi, shield, inflation):
    score = 0
    if hhi < 0.15: score += 1
    if shield >= 1.0: score += 1
    if inflation[0] <= 300: score += 1
    if inflation[1] <= 100: score += 1
    if inflation[2] <= 50: score += 1
    label = "Excellent" if score >= 4 else "Moderate" if score == 3 else "Needs Improvement"
    return label, score
if st.button("Show Audit Report"):
    inflation = inflation_guard(df)
    shock = shock_stopper(df)
    hhi = governance_hhi(allocations)
    shield = liquidity_shield(total_supply_tokens, tge_price, liquidity_fund, allocations, vesting_schedule if input_mode == "Manual Entry" else {})
    lock = lockup_ratio(vesting_schedule if input_mode == "Manual Entry" else {})
    vc = vc_dominance(allocations, tags)
    community = community_index(allocations, tags)
    taper = emission_taper(df)
    monte = monte_carlo_simulation(df, user_base, user_growth_rate, 24, project_type)
    game_label, game_score = game_theory_audit(hhi, shield, inflation)

    st.metric("Game Theory Audit", game_label)

    # Safe plotting
    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 6), inflation, marker='o')
    ax1.set_title("Inflation Guard")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots()
    ax2.bar(shock.keys(), shock.values(), color=['green', 'orange', 'red', 'darkred'])
    ax2.set_title("Shock Stopper")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots()
    if len(set(monte)) > 1:
        ax3.hist(monte, bins=min(20, len(set(monte))))
        ax3.set_title("Monte Carlo Simulation")
    else:
        ax3.bar(["Simulated"], [monte[0]])
        ax3.set_title("Flat Simulation Output")
    st.pyplot(fig3)

    # GPT prompt
    client = OpenAI(api_key=st.secrets["openai"]["api_key"])
    prompt = f"""
You are a senior tokenomics analyst auditing the project '{project_name}'. You've reviewed over 150 token models.

For each metric below, give an analytical breakdown in this format:

1. State the actual metric result (range or value)
2. Interpret it: If high → what risk it carries; if low → what strength it shows
3. Explain the possible impact on price behavior or investor perception

Metrics to report:
- 🟠 YoY Inflation: List values for Year 1–5
- 🔴 Supply Shock: Count of months in ranges: 0–5%, 5–10%, >10%
- 🟡 Governance HHI: Show score and interpret decentralization risk
- 🔵 Liquidity Shield Ratio: Value and implication for price support
- 🔒 Lockup Ratio: Fraction of pools with 12+ month cliff
- 💼 VC Dominance: Combined allocation % of VC-tagged pools
- 👥 Community Control Index: % allocated to community-tagged pools
- 📉 Emission Taper Score: Ratio of first 12M emission to last 12M
- 🎲 Monte Carlo Survivability: Summary of price outcomes after 100 runs
- 🧠 Game Theory Score: Score (0–5) and what it reveals about design resilience

Guidelines:
- Do NOT repeat the definitions of each metric
- Be analytical, insightful, and precise
- This is for institutional readers — avoid fluff
"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a tokenomics audit analyst."},
            {"role": "user", "content": prompt}
        ]
    )
    summary = response.choices[0].message.content
    st.markdown(summary)

    def clean_gpt_text(summary):
        lines = summary.splitlines()
        formatted = []
        for line in lines:
            line = re.sub(r'^#+', '', line).strip()
            line = re.sub(r'^[-*•]\\s+', '', line).strip()
            if line:
                formatted.append(line)
        return formatted

    class PDF(FPDF):
        def header(self):
            self.set_font("Arial", "B", 14)
            self.cell(0, 10, f"Tokenomics Audit Report - {project_name}", ln=True, align="C")

        def footer(self):
            self.set_y(-15)
            self.set_font("Arial", "I", 8)
            self.cell(0, 10, "Report by TDeFi", 0, 0, "C")

    def save_fig_temp(fig, name):
        path = os.path.join(tempfile.gettempdir(), name)
        fig.savefig(path)
        return path

    # Images
    img1 = save_fig_temp(fig1, "inflation.png")
    img2 = save_fig_temp(fig2, "shock.png")
    img3 = save_fig_temp(fig3, "sim.png")

    # PDF
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    for para in clean_gpt_text(summary):
        pdf.multi_cell(0, 10, para.encode("latin-1", "ignore").decode("latin-1"))
        pdf.ln(1)

    for fig_path in [img1, img2, img3]:
        pdf.image(fig_path, x=10, w=180)
        pdf.ln(5)

    # TDeFi Note
    pdf.set_font("Arial", "B", 12)
    pdf.multi_cell(0, 10, "Token Design by TDeFi")
    pdf.set_font("Arial", "", 11)
    tdefi_note = """
Token engineering is not about distribution schedules or supply caps alone - it is the structured discipline of aligning incentives, behavior, and long-term value creation. At its core, it is the science of designing economic systems where every stakeholder, from early investors to late-stage contributors, is guided by aligned motivations. A well-engineered token system creates harmony between product usage, network growth, and token demand. Poor token design, on the other hand, often leads to value leakage, unsustainable emissions, and ultimately, failure of both the product and its economy. At TDeFi, we don't treat tokenomics as an afterthought. We build token models that function as economic engines - driven by utility, governed by logic, and sustained by real-world adoption.

Designing for Demand, Not Just Distribution
A common pitfall in tokenized ecosystems is the decoupling of token emissions from actual product usage or demand. When supply outpaces utility, it triggers sell pressure, user attrition, and a negative flywheel. Our approach at TDeFi ensures that token release schedules are dynamically tied to verifiable demand metrics - whether that's active users, protocol revenue, or ecosystem participation. We engineer feedback loops that reward value creation, not just speculation. Tokenization is a powerful instrument, but only when wielded with precision - designed not as a shortcut to liquidity, but as a long-term mechanism for decentralized ownership, utility alignment, and sustainable network growth.
"""
    for para in tdefi_note.strip().split("\n\n"):
        pdf.multi_cell(0, 10, para.strip().encode("latin-1", "ignore").decode("latin-1"))
        pdf.ln(1)

    pdf_bytes = pdf.output(dest='S').encode('latin-1', 'ignore')
    st.download_button("📄 Download PDF", data=pdf_bytes, file_name=f"{project_name}_Audit_Report.pdf", mime="application/pdf")
