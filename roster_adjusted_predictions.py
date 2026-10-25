import pandas as pd

teams = pd.read_csv(
    "nba_2026_27_predictions.csv"
)

roster_changes = pd.read_csv(
    "player data/2026-27_roster_changes.csv"
)

# Convert full NBA team names to abbreviations
team_abbreviations = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHO",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS"
}

teams["Team"] = teams["Team"].replace(
    team_abbreviations
)

print("Team columns:", teams.columns.tolist())


# PREPARE ROSTER ADJUSTMENTS

adjustments = roster_changes[
    ["Team", "Roster_Impact_Change"]
].copy()

# Limit extreme roster impact scores
adjustments["Roster_Impact_Change"] = (
    adjustments["Roster_Impact_Change"].clip(-1.0, 1.0)
)

# Apply a conservative adjustment
adjustments["Roster_Adjustment"] = (
    adjustments["Roster_Impact_Change"] * 0.5
)

# MERGE WITH TEAM DATA
predictions = teams.merge(
    adjustments,
    on="Team",
    how="left",
    validate="one_to_one"
)

if predictions["Roster_Adjustment"].isna().any():
    missing = predictions.loc[
        predictions["Roster_Adjustment"].isna(),
        "Team"
    ].tolist()

    raise ValueError(
        f"Missing roster adjustments for: {missing}"
    )

# =====================================
# ADJUST TEAM STRENGTH
# =====================================

# Keep the original statistics
predictions["Original_NRtg"] = predictions["NRtg"]
predictions["Original_SRS"] = predictions["SRS"]
predictions["Original_MOV"] = predictions["MOV"]

# Experimental offseason adjustment
predictions["NRtg"] += predictions["Roster_Adjustment"]
predictions["SRS"] += predictions["Roster_Adjustment"]
predictions["MOV"] += predictions["Roster_Adjustment"]

# =====================================
# SAVE NEW PREDICTIONS
# =====================================

predictions.to_csv(
    "nba_2026_27_roster_adjusted.csv",
    index=False
)

print()
print("==============================")
print("2026-27 ROSTER ADJUSTMENTS")
print("==============================")

print(
    predictions[
        [
            "Team",
            "Roster_Impact_Change",
            "Roster_Adjustment",
            "Original_NRtg",
            "NRtg"
        ]
    ].round(3).to_string(index=False)
)

