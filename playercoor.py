import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from nba_api.stats.endpoints import playergamelogs
import subprocess

# Function to update requirements.txt
def update_requirements_file():
    try:
        # Use pip freeze to get the list of installed packages
        result = subprocess.run(["pip", "freeze"], capture_output=True, text=True, check=True)
        installed_packages = result.stdout

        # Write the output to requirements.txt
        with open("requirements.txt", "w") as f:
            f.write(installed_packages)
        print("requirements.txt updated successfully!")
    except Exception as e:
        print(f"An error occurred: {e}")

# Update requirements.txt dynamically
update_requirements_file()

# Load player stats for the season using the updated caching method
@st.cache_data
def load_data():
    return playergamelogs.PlayerGameLogs(season_nullable='2024-25').get_data_frames()[0]

player_stats = load_data()

# Streamlit App
st.title("NBA Player Correlation and Regression Tool")

# Select Team
teams = player_stats['TEAM_ABBREVIATION'].unique()
selected_team = st.selectbox("Select a Team:", sorted(teams))

# Dynamically Load Players
team_players = player_stats[player_stats['TEAM_ABBREVIATION'] == selected_team]['PLAYER_NAME'].unique()
selected_player = st.selectbox("Select a Player:", sorted(team_players))

# Filter Player Data
player_data = player_stats[player_stats['PLAYER_NAME'] == selected_player]

# Drop Rows with Missing Data
player_data = player_data.dropna()

# Select Base Stat and Comparison Stats
stats = player_data.select_dtypes(include=['number']).columns  # Numeric columns only
base_stat = st.selectbox("Select Base Stat (Independent Variable):", sorted(stats))
comparison_stats = st.multiselect("Select Up to 3 Stats for Comparison (Dependent Variables):", sorted(stats), max_selections=3)

# Analysis and Results
if st.button("Analyze Correlation and Regression") and base_stat and comparison_stats:
    results = []
    for stat in comparison_stats:
        if stat in player_data.columns:
            X = player_data[[base_stat]].values
            y = player_data[stat].values

            # Calculate Correlation
            correlation = player_data[base_stat].corr(player_data[stat])

            # Perform Linear Regression
            model = LinearRegression()
            model.fit(X, y)
            r2 = model.score(X, y)
            coefficient = model.coef_[0]

            results.append({
                "Base": base_stat,
                "Comparison Stat": stat,
                "Correlation": correlation,
                "R-squared": r2,
                "Coefficient": coefficient
            })

    # Display Results
    st.subheader("Results")
    for result in results:
        st.markdown(f"**{result['Comparison Stat']} vs {result['Base']}:**")
        st.write(f"Correlation: {result['Correlation']:.2f}")
        st.write(f"R-squared: {result['R-squared']:.2f}")
        st.write(f"Regression Coefficient: {result['Coefficient']:.2f}")
