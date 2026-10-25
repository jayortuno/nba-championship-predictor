import pandas as pd

# Load roster comparison
roster_df = pd.read_csv(
    "player data/2026-27_roster_comparison.csv"
)

# Load recovered advanced statistics
recovered_df = pd.read_csv(
    "player data/2026-27_recovered_advanced.csv"
)

# Keep only the columns we need
recovered_df = recovered_df[
    ["NBA_Player_ID", "BPM", "VORP", "MP",]
]

# Merge recovered statistics using NBA player IDs
roster_df = roster_df.merge(
    recovered_df,
    on="NBA_Player_ID",
    how="left",
    suffixes=("", "_Recovered"),
    validate="one_to_one"
)

# Fill missing statistics with recovered values
for column in ["BPM", "VORP", "MP"]:
    roster_df[column] = roster_df[column].fillna(
        roster_df[f"{column}_Recovered"]
    )

# Remove temporary columns
roster_df = roster_df.drop(
    columns=["BPM_Recovered", "VORP_Recovered", "MP_Recovered"]
)

# Save completed roster statistics
roster_df.to_csv(
    "player data/2026-27_complete_roster_stats.csv",
    index=False
)

print()
print("==============================")
print("COMPLETE ROSTER STATISTICS")
print("==============================")

print("Total Players:", len(roster_df))
print("Players With BPM:", roster_df["BPM"].notna().sum())
print("Players Missing BPM:", roster_df["BPM"].isna().sum())