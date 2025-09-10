import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Streamlit Page Config
st.set_page_config(page_title="F1 2025 Race Predictor", layout="wide")

st.title("🏁 F1 2025 Race Time Prediction Dashboard")
st.markdown("Compare predicted race times based on qualifying and sector data.")

# Sidebar race selection
race_option = st.sidebar.selectbox("Select Grand Prix", ["Chinese GP", "Japanese GP", "Australian GP"])

# File mapping
prediction_files = {
    "Chinese GP": "predictions/prediction_china_2025.csv",
    "Japanese GP": "predictions/prediction_japan_2025.csv",
    "Australian GP": "predictions/prediction_australia_2025.csv"
}

track_images = {
    "Chinese GP": "images/china_track.png",
    "Japanese GP": "images/japan_track.png",
    "Australian GP": "images/australia_track.png"
}

mae_files = {
    "Chinese GP": "predictions/mae_china_2025.txt",
    "Japanese GP": "predictions/mae_japan_2025.txt",
    "Australian GP": "predictions/mae_australia_2025.txt"
}

selected_file = prediction_files[race_option]

# Main visualization block
try:
    df = pd.read_csv(selected_file)
    st.subheader(f"📊 Predicted Results - {race_option}")

    df["PredictedRaceTime (s)"] = df["PredictedRaceTime (s)"].round(3)
    df = df.sort_values("PredictedRaceTime (s)").reset_index(drop=True)
    df.index = df.index + 1
    df["Position"] = df.index

    st.dataframe(df[["Position", "Driver", "PredictedRaceTime (s)"]], use_container_width=True)

    # Podium Display
    st.markdown("### 🏆 Predicted Podium Finishers")
    st.success(f"🥇 {df.iloc[0]['Driver']}")
    st.info(f"🥈 {df.iloc[1]['Driver']}")
    st.warning(f"🥉 {df.iloc[2]['Driver']}")

    # Model Error
    mae_file = mae_files.get(race_option)
    if mae_file and os.path.exists(mae_file):
        with open(mae_file, "r") as f:
            mae_value = f.read().strip()
            st.metric("Mean Absolute Error (MAE)", f"{float(mae_value):.2f} seconds")

    # Bar Chart
    fig = px.bar(
        df,
        x="Driver",
        y="PredictedRaceTime (s)",
        color="Driver",
        text="PredictedRaceTime (s)",
        title=f"{race_option} - Predicted Race Times",
        height=500
    )
    fig.update_traces(texttemplate='%{text:.2f}s', textposition='outside')
    fig.update_layout(xaxis_tickangle=-45, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    # Show track image if available
    if race_option in track_images and os.path.exists(track_images[race_option]):
        st.image(track_images[race_option], caption=f"{race_option} Track Layout", use_container_width=True)

    # Check for sector columns
    sector_cols = ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]
    if all(col in df.columns for col in sector_cols):
        st.markdown("### Average Sector Time Comparison")

        df_sector = df[["Driver"] + sector_cols].copy()
        df_melted = df_sector.melt(id_vars=["Driver"], 
                                var_name="Sector", 
                                value_name="Time (s)")

        sector_bar = px.bar(
            df_melted,
            x="Driver",
            y="Time (s)",
            color="Sector",
            barmode="group",
            text_auto=".2s",
            title=f"{race_option} - Sector-wise Time Comparison"
        )
        sector_bar.update_layout(xaxis_tickangle=-45, height=500)
        st.plotly_chart(sector_bar, use_container_width=True)

        #Qualifying Time vs Predicted Race Time as Bubble Chart
        if "QualifyingTime (s)" in df.columns and "PredictedRaceTime (s)" in df.columns:
            st.markdown("### Qualifying vs Predicted Race Time (Bubble Chart)")

            # Create size feature
            df["TimeDelta"] = (df["PredictedRaceTime (s)"] - df["QualifyingTime (s)"]).abs()

            # Drop rows where TimeDelta is missing (optional: use .fillna(0) instead)
            df_clean = df.dropna(subset=["TimeDelta", "QualifyingTime (s)", "PredictedRaceTime (s)"])

            # Create bubble chart
            bubble_chart = px.scatter(
                df_clean,
                x="QualifyingTime (s)",
                y="PredictedRaceTime (s)",
                color="Driver",
                size="TimeDelta",
                hover_name="Driver",
                title="Qualifying Time vs Predicted Race Time (Bubble Size = Time Delta)",
                height=500
            )
            bubble_chart.update_traces(marker=dict(opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(bubble_chart, use_container_width=True)

        # ✅ Rank Change Table (Qualifying vs Predicted)
        if "QualifyingTime (s)" in df.columns and "PredictedRaceTime (s)" in df.columns:
            st.markdown("### 🔁 Position Delta (Qualifying vs Predicted Rank)")

            df_ranked = df.dropna(subset=["QualifyingTime (s)", "PredictedRaceTime (s)"]).copy()
            df_ranked["QualifyingRank"] = df_ranked["QualifyingTime (s)"].rank().astype(int)
            df_ranked["PredictedRank"] = df_ranked["PredictedRaceTime (s)"].rank().astype(int)
            df_ranked["Δ Position"] = df_ranked["QualifyingRank"] - df_ranked["PredictedRank"]

            st.dataframe(df_ranked[["Driver", "QualifyingRank", "PredictedRank", "Δ Position"]].sort_values("PredictedRank"), use_container_width=True)

            heatmap = px.bar(
                df_ranked.sort_values("Δ Position"),
                x="Driver",
                y="Δ Position",
                color="Δ Position",
                color_continuous_scale="RdBu",
                title="Position Change from Qualifying to Race Prediction"
            )
            heatmap.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(heatmap, use_container_width=True)

except FileNotFoundError:
    st.error(f"No predictions found for {race_option}. Please run the model pipeline first.")
