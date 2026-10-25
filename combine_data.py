import pandas as pd
import glob
import os

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

files = glob.glob("nba data/*.csv")

all_seasons = []

for file in files:
    df = pd.read_csv(file, header=1)

    # Remove completely blank rows
    df = df.dropna(how="all")

    # Clean team names
    df["Team"] = df["Team"].str.replace("*", "", regex=False)

    df = df[df["Team"] != "League Average"]

    # Get season from filename
    season = file.split("Stats ")[1].split("(")[0]

    # Add season column
    df["Season"] = season

    # Add this season's data to our list
    all_seasons.append(df)

combined_df = pd.concat(all_seasons, ignore_index=True)

print(combined_df["Season"].value_counts().sort_index())
print(combined_df.shape)

combined_df.to_csv("nba_historical_team_stats.csv", index=False)

print("Dataset saved!")

combined_df = pd.concat(all_seasons, ignore_index=True)

champions = {
    "2015-16": "Cleveland Cavaliers",
    "2016-17": "Golden State Warriors",
    "2017-18": "Golden State Warriors",
    "2018-19": "Toronto Raptors",
    "2019-20": "Los Angeles Lakers",
    "2020-21": "Milwaukee Bucks",
    "2021-22": "Golden State Warriors",
    "2022-23": "Denver Nuggets",
    "2023-24": "Boston Celtics",
    "2024-25": "Oklahoma City Thunder",
    "2025-26": "New York Knicks"
}

combined_df["Champion"] = (
    combined_df.apply(
        lambda row: 1 if champions[row["Season"]] == row["Team"] else 0,
        axis=1
    )
)

print(combined_df[combined_df["Champion"] == 1][["Season", "Team"]])
combined_df.to_csv("nba_historical_team_stats.csv", index=False)
print("Final Dataset saved! ")

print(combined_df.columns.tolist())

features = [
    "W",
    "MOV",
    "SRS",
    "NRtg",
    "ORtg",
    "DRtg",
    "eFG%",
    "TS%",
    "TOV%",
    "ORB%",
    "DRB%"
]

X = combined_df[features]
y = combined_df["Champion"]

print(X.head())
print(y.head())


# Training seasons
seasons = sorted(combined_df["Season"].unique())

# Store results
logistic_results = []
rf_results = []
rf_top5_results = []

# Start with 2016-17 because we need at least one
# previous season to train on
for i in range(1, len(seasons)):

    # Seasons before the season we are predicting
    train_seasons = seasons[:i]

    # Season we are testing
    test_season = seasons[i]

    # Separate training and testing data
    train_data = combined_df[
        combined_df["Season"].isin(train_seasons)
    ]

    test_data = combined_df[
        combined_df["Season"] == test_season
    ].copy()

    # Create training and testing variables
    X_train = train_data[features]
    y_train = train_data["Champion"]

    X_test = test_data[features]

    # Logistic Regression
    logistic_model = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced",
            max_iter=1000
        ))
    ])

    logistic_model.fit(X_train, y_train)

    test_data["Logistic_Probability"] = (
        logistic_model.predict_proba(X_test)[:, 1] * 100
    )

    logistic_pick = test_data.sort_values(
        "Logistic_Probability",
        ascending=False
    ).iloc[0]

    actual_champion = test_data[
        test_data["Champion"] == 1
    ].iloc[0]

    logistic_correct = (
        logistic_pick["Team"] == actual_champion["Team"]
    )

    logistic_results.append(logistic_correct)

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )

    rf_model.fit(X_train, y_train)

    test_data["RF_Probability"] = (
        rf_model.predict_proba(X_test)[:, 1] * 100
    )

    rf_pick = test_data.sort_values(
        "RF_Probability",
        ascending=False
    ).iloc[0]

    rf_correct = (
        rf_pick["Team"] == actual_champion["Team"]
    )

    #check if champion is in model's top 5
    rf_top5 = test_data.sort_values(
        "RF_Probability",
        ascending=False
    ).head(5)

    rf_top5_correct = (
        actual_champion["Team"] in rf_top5["Team"].values
    )

    rf_ranked = test_data.sort_values(
        "RF_Probability",
        ascending=False
    ).reset_index(drop=True)

    champion_rank = (
        rf_ranked[
            rf_ranked["Team"] == actual_champion["Team"]
        ].index[0] + 1
    )

    rf_results.append(rf_correct)
    rf_top5_results.append(rf_top5_correct)

    print()
    print("Season:", test_season)
    print("Actual Champion:", actual_champion["Team"])

    print(
        "Logistic Pick:",
        logistic_pick["Team"],
        "|",
        round(logistic_pick["Logistic_Probability"], 2),
        "%"
    )

    print(
        "Random Forest Pick:",
        rf_pick["Team"],
        "|",
        round(rf_pick["RF_Probability"], 2),
        "%"
    )

    print("Logistic Correct:", logistic_correct)
    print("Random Forest Correct:", rf_correct)
    print("Actual Champion Rank:", champion_rank)


# Final accuracy
logistic_accuracy = (
    sum(logistic_results) / len(logistic_results) * 100
)

rf_accuracy = (
    sum(rf_results) / len(rf_results) * 100
)

rf_top5_accuracy = (
    sum(rf_top5_results) / len(rf_top5_results) * 100
)



print()
print("==============================")
print("FINAL MODEL COMPARISON")
print("==============================")

print(
    "Logistic Regression:",
    round(logistic_accuracy, 2),
    "%"
)

print(
    "Random Forest:",
    round(rf_accuracy, 2),
    "%"
)
print(
    "Random Forest Top-5:",
    round(rf_top5_accuracy, 2),
    "%"
)

rf_importance = pd.DataFrame({
    "Feature": features,
    "Importance": rf_model.feature_importances_
})

rf_importance = rf_importance.sort_values(
    "Importance",
    ascending=False
)

print()
print("==============================")
print("RANDOM FOREST FEATURE IMPORTANCE")
print("==============================")
print(rf_importance)

# =====================================
# TRAIN FINAL RANDOM FOREST MODEL
# =====================================

X_final = combined_df[features]
y_final = combined_df["Champion"]

final_rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=5,
    min_samples_leaf=3,
    class_weight="balanced",
    random_state=42
)

final_rf_model.fit(X_final, y_final)

print()
print("==============================")
print("FINAL MODEL TRAINED")
print("==============================")
print("Training Rows:", len(X_final))
print("Features Used:", len(features))


# Get the most recent season
prediction_data = combined_df[
    combined_df["Season"] == "2025-26"
].copy()

#Change the season
prediction_data["Season"] = "2026-27"

prediction_data = prediction_data.drop(
    columns=["Champion"]
)

prediction_data.to_csv(
    "nba_2026_27_predictions.csv",
    index=False
)

print()
print("==============================")
print("2026-27 DATASET CREATED")
print("==============================")
print("Teams:", len(prediction_data))
print("Features:", len(features))
print(prediction_data[["Team", "Season"]].head())

future_teams = pd.read_csv(
    "nba_2026_27_roster_adjusted.csv"
)

features = [
    "W", "MOV", "SRS", "NRtg",
    "ORtg", "DRtg",
    "eFG%", "TS%", "TOV%",
    "ORB%", "DRB%"
]

X_future = future_teams[features].copy()

# Predict using the trained Random Forest
future_teams["Championship_Score"] = (
    final_rf_model.predict_proba(X_future)[:, 1]
)

# Normalize scores across the 30 teams
score_total = future_teams["Championship_Score"].sum()

if score_total == 0:
    raise ValueError("All championship scores are zero.")

future_teams["Normalized_Score"] = (
    future_teams["Championship_Score"]
    / score_total
    * 100
)

future_teams = future_teams.sort_values(
    "Normalized_Score",
    ascending=False
)

future_teams.to_csv(
    "nba_2026_27_championship_predictions.csv",
    index=False
)

print()
print("==============================")
print("2026-27 NBA CHAMPIONSHIP RANKINGS")
print("==============================")

print(
    future_teams[
        ["Team", "Normalized_Score"]
    ].head(15).round(2).to_string(index=False)
)


# COMPARE ORIGINAL VS ADJUSTED PREDICTIONS
original_teams = pd.read_csv(
    "nba_2026_27_predictions.csv"
)

# Convert original full team names to abbreviations
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

original_teams["Team"] = original_teams["Team"].replace(
    team_abbreviations
)

# Original model scores
original_teams["Original_Score"] = (
    final_rf_model.predict_proba(
        original_teams[features]
    )[:, 1]
)

original_total = original_teams["Original_Score"].sum()

if original_total == 0:
    raise ValueError("All original model scores are zero.")

original_teams["Original_Score"] = (
    original_teams["Original_Score"]
    / original_total
    * 100
)

# Compare scores
comparison = original_teams[
    ["Team", "Original_Score"]
].merge(
    future_teams[["Team", "Normalized_Score"]],
    on="Team",
    validate="one_to_one"
)

comparison["Score_Change"] = (
    comparison["Normalized_Score"]
    - comparison["Original_Score"]
)

comparison = comparison.sort_values(
    "Normalized_Score",
    ascending=False
)

print()
print("==============================")
print("ORIGINAL VS ADJUSTED PREDICTIONS")
print("==============================")

print(
    comparison[
        ["Team", "Original_Score",
         "Normalized_Score", "Score_Change"]
    ].round(2).to_string(index=False)
)


#MODEL SENSITIVITY TEST
sensitivity_teams = pd.read_csv(
    "nba_2026_27_roster_adjusted.csv"
)

multipliers = [0.0, 0.25, 0.5, 0.75, 1.0]

all_results = []

for multiplier in multipliers:

    test_teams = sensitivity_teams.copy()

    # Use the same capped BPM values as Step 38
    bpm_change = test_teams["BPM_Change"]

    adjustment = (
        bpm_change * multiplier
    ).clip(-0.5, 0.5)

    # Adjust team statistics
    for stat in ["NRtg", "SRS", "MOV"]:

        test_teams[stat] = (
            test_teams[f"Original_{stat}"]
            + adjustment
        )

    # Generate Random Forest scores
    test_teams["Score"] = (
        final_rf_model.predict_proba(
            test_teams[features]
        )[:, 1]
    )

    total_score = test_teams["Score"].sum()

    if total_score == 0:
        raise ValueError(
            f"All scores are zero for multiplier {multiplier}"
        )

    test_teams["Normalized_Score"] = (
        test_teams["Score"]
        / total_score
        * 100
    )

    test_teams = test_teams.sort_values(
        "Normalized_Score",
        ascending=False
    )

    print()
    print("==============================")
    print(f"MULTIPLIER: {multiplier}")
    print("==============================")

    print(
        test_teams[
            ["Team", "Normalized_Score"]
        ].head(5).round(2).to_string(index=False)
    )

    for _, row in test_teams.iterrows():

        all_results.append({
            "Multiplier": multiplier,
            "Team": row["Team"],
            "Score": row["Normalized_Score"]
        })

# Save all results
sensitivity_results = pd.DataFrame(all_results)

sensitivity_results.to_csv(
    "nba_2026_27_sensitivity_results.csv",
    index=False
)


