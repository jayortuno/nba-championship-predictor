import pandas as pd

# Load last season's player data
old_players = pd.read_csv(
    "player data/2025-26_player_final_teams.csv"
)

# Load current roster data
new_players = pd.read_csv(
    "player data/2026-27_complete_roster_stats.csv"
)

# Load original advanced player statistics
advanced = pd.read_csv(
    "player data/2025-26_player_advanced.csv",
    encoding="cp1252"
)

# Prefer combined-season rows for traded players
advanced["Combined_Row"] = (
    advanced["Team"].astype(str)
    .str.fullmatch(r"\d+TM")
    .fillna(False)
)

advanced = advanced.sort_values(
    "Combined_Row",
    ascending=False
)

# Prepare Win Shares for matching
win_shares = advanced[["Player", "WS"]].copy()

win_shares["Match_Name"] = (
    win_shares["Player"].str.lower().str.strip()
)

win_shares = win_shares.drop_duplicates(
    subset="Match_Name"
)[["Match_Name", "WS"]]

# Add Win Shares to both rosters
for df in [old_players, new_players]:
    df["Match_Name"] = (
        df["Player"].str.lower().str.strip()
    )

# Old roster already contains Win Shares
# Only merge Win Shares into the new roster

new_players = new_players.merge(
    win_shares,
    on="Match_Name",
    how="left",
    validate="many_to_one"
)
# Standardize old team abbreviations
old_players["Final_Team"] = old_players["Final_Team"].replace({
    "BKN": "BRK",
    "CHA": "CHO",
    "PHX": "PHO"
})

# Keep players with valid statistics
old_valid = old_players.dropna(
    subset=["BPM", "MP", "Final_Team"]
).copy()

new_valid = new_players.dropna(
    subset=["BPM", "MP", "Team"]
).copy()

# Calculate minutes-weighted BPM
old_valid["Weighted_BPM"] = (
    old_valid["BPM"] * old_valid["MP"]
)

new_valid["Weighted_BPM"] = (
    new_valid["BPM"] * new_valid["MP"]
)

# Calculate last season's roster strength
old_team = old_valid.groupby("Final_Team").agg(
    Old_Weighted_BPM=("Weighted_BPM", "sum"),
    Old_Minutes=("MP", "sum"),
    Old_VORP=("VORP", "sum"),
    Old_WS=("WS", "sum")
).reset_index()

old_team["Old_Average_BPM"] = (
    old_team["Old_Weighted_BPM"]
    / old_team["Old_Minutes"]
)

old_team = old_team.rename(
    columns={"Final_Team": "Team"}
)

# Calculate current roster strength
new_team = new_valid.groupby("Team").agg(
    New_Weighted_BPM=("Weighted_BPM", "sum"),
    New_Minutes=("MP", "sum"),
    New_VORP=("VORP", "sum"),
    New_WS=("WS", "sum")
).reset_index()

new_team["New_Average_BPM"] = (
    new_team["New_Weighted_BPM"]
    / new_team["New_Minutes"]
)

# Compare team strengths
comparison = old_team.merge(
    new_team,
    on="Team",
    how="inner",
    validate="one_to_one"
)

comparison["BPM_Change"] = (
    comparison["New_Average_BPM"]
    - comparison["Old_Average_BPM"]
)


# Changes in VORP and Win Shares
comparison["VORP_Change"] = (
    comparison["New_VORP"]
    - comparison["Old_VORP"]
)

comparison["WS_Change"] = (
    comparison["New_WS"]
    - comparison["Old_WS"]
)

# Standardize each change across teams
change_columns = [
    "BPM_Change",
    "VORP_Change",
    "WS_Change"
]

for col in change_columns:
    std = comparison[col].std()

    if std > 0:
        comparison[col + "_Scaled"] = (
            (comparison[col] - comparison[col].mean()) / std
        )
    else:
        comparison[col + "_Scaled"] = 0.0

# Overall roster change score
comparison["Roster_Impact_Change"] = (
    comparison["BPM_Change_Scaled"]
    + comparison["VORP_Change_Scaled"]
    + comparison["WS_Change_Scaled"]
) / 3

# Sort teams by overall roster improvement
comparison = comparison.sort_values(
    "Roster_Impact_Change",
    ascending=False
)

# Save results
comparison.to_csv(
    "player data/2026-27_roster_changes.csv",
    index=False
)

print()
print("==============================")
print("ROSTER STRENGTH CHANGES")
print("==============================")

print(
    comparison[
        [
            "Team",
            "BPM_Change",
            "VORP_Change",
            "WS_Change",
            "Roster_Impact_Change"
        ]
    ].round(3).to_string(index=False)
)

# INSPECT PLAYERS WHO CHANGED TEAMS

roster_comparison = pd.read_csv(
    "player data/2026-27_complete_roster_stats.csv"
)

changed_players = roster_comparison[
    roster_comparison["Status"] == "Changed Teams"
].copy()

print()
print("==============================")
print("PLAYERS JOINING SELECTED TEAMS")
print("==============================")

for team in ["IND", "TOR", "DAL", "OKC", "LAC"]:

    print()
    print("TEAM:", team)

    arrivals = changed_players[
        changed_players["Team"] == team
    ].sort_values("MP", ascending=False)

    print(
        arrivals[
            ["Player", "Final_Team", "BPM", "MP"]
        ].to_string(index=False)
    )

# =====================================
# IDENTIFY PLAYERS WHO LEFT EACH TEAM
# =====================================

# Get players from the 2025-26 season
old_roster = pd.read_csv(
    "player data/2025-26_player_final_teams.csv"
)

# Get players from the 2026-27 roster
new_roster = pd.read_csv(
    "player data/2026-27_complete_roster_stats.csv"
)

# Standardize team abbreviations
old_roster["Final_Team"] = old_roster["Final_Team"].replace({
    "BKN": "BRK",
    "CHA": "CHO",
    "PHX": "PHO"
})

# Compare using cleaned player names
def normalize_name(name):
    import unicodedata

    name = str(name).lower().strip()
    name = unicodedata.normalize("NFKD", name)

    return "".join(
        char for char in name
        if not unicodedata.combining(char)
    )


old_roster["Clean_Name"] = old_roster["Player"].apply(
    normalize_name
)

new_roster["Clean_Name"] = new_roster["Player"].apply(
    normalize_name
)

# Apply the confirmed name corrections from roster_comparison.py
name_fixes = {
    "Karlo Matković": "Karlo Matkovi?",
    "DaRon Holmes II": "DaRon Holmes",
    "Nikola Jokić": "Nikola Joki?",
    "Kristaps Porziņģis": "Kristaps Porzi??is",
    "Jimmy Butler III": "Jimmy Butler",
    "Alperen Sengun": "Alperen ?engün",
    "Bogdan Bogdanović": "Bogdan Bogdanovi?",
    "Luka Dončić": "Luka Don?i?",
    "Nikola Jović": "Nikola Jovi?",
    "Bobby Portis Jr.": "Bobby Portis",
    "AJ Green": "A.J. Green",
    "Kasparas Jakučionis": "Kasparas Jaku?ionis",
    "Terrence Shannon Jr": "Terrence Shannon Jr.",
    "Nikola Vučević": "Nikola Vu?evi?",
    "Vít Krejčí": "Vít Krej?í",
    "Robert Williams III": "Robert Williams",
    "Nikola Topić": "Nikola Topi?",
    "Trey Jemison III": "Trey Jemison",
    "Jusuf Nurkić": "Jusuf Nurki?",
    "Walter Clayton Jr.": "Walter Clayton",
    "GG Jackson": "GG Jackson II",
    "Ronald Holland II": "Ron Holland"
}

new_roster["Clean_Name"] = new_roster["Player"].replace(
    name_fixes
).apply(normalize_name)

# Players in old data but not current rosters
departed = old_roster[
    ~old_roster["Clean_Name"].isin(
        new_roster["Clean_Name"]
    )
].copy()

# Players who changed teams are also departures
changed_teams = old_roster.merge(
    new_roster[["Clean_Name", "Team"]].rename(
        columns={"Team": "New_Team"}
    ),
    on="Clean_Name",
    how="inner"
)

changed_teams = changed_teams[
    changed_teams["Final_Team"] != changed_teams["New_Team"]
]

departed = pd.concat(
    [departed, changed_teams],
    ignore_index=True
)

departed = departed.drop_duplicates(
    subset=["Player-additional"]
)

print()
print("==============================")
print("PLAYERS WHO LEFT SELECTED TEAMS")
print("==============================")

for team in ["IND", "TOR", "DAL", "OKC", "LAC"]:

    print()
    print("TEAM:", team)

    team_departures = departed[
        departed["Final_Team"] == team
    ].sort_values("MP", ascending=False)

    print(
        team_departures[
            ["Player", "BPM", "MP"]
        ].head(15).to_string(index=False)
    )


# DEPARTURE IMPACT BY TEAM

departure_stats = departed.groupby("Final_Team").agg(
    Departed_Players=("Player", "count"),
    Departed_Minutes=("MP", "sum"),
    Departed_VORP=("VORP", "sum")
).reset_index()

departure_stats = departure_stats.rename(
    columns={"Final_Team": "Team"}
)

departure_stats = departure_stats.sort_values(
    "Departed_Minutes",
    ascending=False
)

print()
print("==============================")
print("TEAM DEPARTURE IMPACT")
print("==============================")

print(
    departure_stats[
        ["Team", "Departed_Players",
         "Departed_Minutes", "Departed_VORP"]
    ].round(2).to_string(index=False)
)