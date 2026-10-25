import pandas as pd

# Load completed roster statistics
roster_df = pd.read_csv(
    "player data/2026-27_complete_roster_stats.csv"
)

# Load original player statistics
original_players = pd.read_csv(
    "player data/2025-26_player_advanced.csv",
    encoding="cp1252"
)


# Prioritize combined-season rows for traded players
original_players["Combined_Row"] = (
    original_players["Team"]
    .astype(str)
    .str.fullmatch(r"\d+TM")
    .fillna(False)
)

# Put combined-season rows first
original_players = original_players.sort_values(
    "Combined_Row",
    ascending=False
)

# Rebuild the Win Shares data
win_shares = original_players[
    ["Player", "WS"]
].copy()

win_shares["Clean_Name"] = (
    win_shares["Player"]
    .str.lower()
    .str.strip()
)

# Keep one row per player
win_shares = win_shares.drop_duplicates(
    subset="Clean_Name"
)
# Remove the extra Player column
win_shares = win_shares.drop(columns=["Player"])

# Merge using cleaned player names
roster_df["Clean_Name"] = (
    roster_df["Clean_Name"]
    .str.lower()
    .str.strip()
)

roster_df = roster_df.merge(
    win_shares,
    on="Clean_Name",
    how="left",
    validate="many_to_one"
)

print(
    "Players with Win Shares:",
    roster_df["WS"].notna().sum()
)

print(
    "Total roster players:",
    len(roster_df)
)

# CALCULATE TEAM ROSTER IMPACT

# Keep players with usable BPM and minutes
valid_players = roster_df.dropna(
    subset=["BPM", "MP"]
).copy()

# Calculate minutes-weighted BPM
valid_players["Weighted_BPM"] = (
    valid_players["BPM"] * valid_players["MP"]
)

# Group players by their current team
team_stats = valid_players.groupby("Team").agg(
    Total_Weighted_BPM=("Weighted_BPM", "sum"),
    Total_Minutes=("MP", "sum"),
    Players_With_Stats=("Player", "count"),
    Total_VORP=("VORP", "sum"),
    Total_WS=("WS", "sum")
).reset_index()

# Calculate average BPM
team_stats["Average_BPM"] = (
    team_stats["Total_Weighted_BPM"]
    / team_stats["Total_Minutes"]
)

# Rank teams separately for each statistic
team_stats["BPM_Rank"] = team_stats["Average_BPM"].rank(
    ascending=False
)

team_stats["VORP_Rank"] = team_stats["Total_VORP"].rank(
    ascending=False
)

team_stats["WS_Rank"] = team_stats["Total_WS"].rank(
    ascending=False
)

# Average the three rankings
team_stats["Overall_Rank_Score"] = (
    team_stats["BPM_Rank"]
    + team_stats["VORP_Rank"]
    + team_stats["WS_Rank"]
) / 3

# Sort from strongest to weakest
team_stats = team_stats.sort_values(
    "Overall_Rank_Score",
    ascending=True
)
# Save results
team_stats.to_csv(
    "player data/2026-27_team_roster_impact.csv",
    index=False
)

print()
print("==============================")
print("TEAM ROSTER IMPACT RANKINGS")
print("==============================")

print(
    team_stats[
        [
            "Team",
            "Players_With_Stats",
            "Average_BPM",
            "Total_VORP",
            "Total_WS",
            "Overall_Rank_Score"
        ]
    ].round(2).to_string(index=False)
)


# CHECK TEAM DATA COVERAGE
team_coverage = roster_df.groupby("Team").agg(
    Total_Players=("Player", "count"),
    Players_With_BPM=("BPM", "count")
).reset_index()

team_coverage["Missing_Players"] = (
    team_coverage["Total_Players"]
    - team_coverage["Players_With_BPM"]
)

team_coverage["Coverage_Percent"] = (
    team_coverage["Players_With_BPM"]
    / team_coverage["Total_Players"]
    * 100
).round(1)

team_coverage = team_coverage.sort_values(
    "Coverage_Percent"
)

print()
print("==============================")
print("TEAM DATA COVERAGE")
print("==============================")

print(team_coverage.to_string(index=False))

# =====================================
# TOP PLAYERS BY HISTORICAL MINUTES
# =====================================

top_players = valid_players.sort_values(
    ["Team", "MP"],
    ascending=[True, False]
)

top_players = top_players.groupby("Team").head(8)

print()
print("==============================")
print("TOP 8 PLAYERS BY TEAM")
print("==============================")

for team in ["OKC", "SAS", "TOR", "CLE", "DEN"]:

    print()
    print("TEAM:", team)

    team_players = top_players[
        top_players["Team"] == team
    ]

    print(
        team_players[
            ["Player", "MP", "BPM"]
        ].to_string(index=False)
    )