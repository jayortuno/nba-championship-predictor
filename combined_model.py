import pandas as pd
from sklearn.ensemble import RandomForestClassifier


#COMBINED MODEL TEST
historical = pd.read_csv(
    "nba_historical_team_stats.csv"
)

preseason = pd.read_csv(
    "nba_preseason_training_data.csv"
)

features = [
    "W", "MOV", "SRS", "NRtg",
    "ORtg", "DRtg",
    "eFG%", "TS%", "TOV%",
    "ORB%", "DRB%"
]


def create_model():
    return RandomForestClassifier(
        n_estimators=300,
        max_depth=5,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42
    )


def normalize_scores(scores):
    total = scores.sum()

    if total == 0:
        raise ValueError("All model scores are zero.")

    return scores / total * 100


seasons = sorted(preseason["Next_Season"].unique())

results = []


# TEST ON HISTORICAL SEASONS
for i in range(1, len(seasons)):

    test_season = seasons[i]

    # Original model: train on past seasons
    original_train = historical[
        historical["Season"] < test_season
    ]

    original_model = create_model()

    original_model.fit(
        original_train[features],
        original_train["Champion"]
    )

    # Preseason model: train on previous
    # preseason examples
    preseason_train = preseason[
        preseason["Next_Season"] < test_season
    ]

    preseason_model = create_model()

    preseason_model.fit(
        preseason_train[features],
        preseason_train["Next_Champion"]
    )

    # Same preseason inputs for both models
    test = preseason[
        preseason["Next_Season"] == test_season
    ].copy()

    # Generate model scores
    original_scores = original_model.predict_proba(
        test[features]
    )[:, 1]

    preseason_scores = preseason_model.predict_proba(
        test[features]
    )[:, 1]

    # Normalize separately before combining
    test["Original_Score"] = normalize_scores(
        original_scores
    )

    test["Preseason_Score"] = normalize_scores(
        preseason_scores
    )

    # 50/50 combination
    test["Combined_Score"] = (
        test["Original_Score"] * 0.5
        + test["Preseason_Score"] * 0.5
    )

    # Compare the three models
    season_result = {
        "Season": test_season,
        "Actual": test.loc[
            test["Next_Champion"] == 1, "Team"
        ].iloc[0]
    }

    for name, column in [
        ("Original", "Original_Score"),
        ("Preseason", "Preseason_Score"),
        ("Combined", "Combined_Score")
    ]:

        ranked = test.sort_values(
            column, ascending=False
        ).reset_index(drop=True)

        champion_rank = (
            ranked.index[
                ranked["Next_Champion"] == 1
            ][0] + 1
        )

        season_result[f"{name}_Pick"] = (
            ranked.iloc[0]["Team"]
        )

        season_result[f"{name}_Rank"] = champion_rank

    results.append(season_result)

    print()
    print("==============================")
    print(f"SEASON: {test_season}")
    print("==============================")

    print("Actual:", season_result["Actual"])

    for name in ["Original", "Preseason", "Combined"]:
        print(
            f"{name}: "
            f"{season_result[f'{name}_Pick']} "
            f"(Champion Rank: "
            f"{season_result[f'{name}_Rank']})"
        )



# FINAL PERFORMANCE COMPARISON
results_df = pd.DataFrame(results)

print()
print("==============================")
print("COMBINED MODEL PERFORMANCE")
print("==============================")

for name in ["Original", "Preseason", "Combined"]:

    ranks = results_df[f"{name}_Rank"]

    print()
    print(name)
    print("------------------------------")

    for n in [1, 2, 3, 5]:
        accuracy = (ranks <= n).mean() * 100

        print(
            f"Top-{n} Accuracy: {accuracy:.1f}%"
        )

    print(
        "Average Champion Rank:",
        round(ranks.mean(), 2)
    )


results_df.to_csv(
    "nba_combined_model_results.csv",
    index=False
)

# Train both models using all available data
final_original_model = create_model()
final_original_model.fit(
    historical[features],
    historical["Champion"]
)

final_preseason_model = create_model()
final_preseason_model.fit(
    preseason[features],
    preseason["Next_Champion"]
)

# Load 2026-27 roster-adjusted statistics
future = pd.read_csv(
    "nba_2026_27_roster_adjusted.csv"
)

# Get scores from both models
original_scores = final_original_model.predict_proba(
    future[features]
)[:, 1]

preseason_scores = final_preseason_model.predict_proba(
    future[features]
)[:, 1]

# Normalize each model's scores
future["Original_Score"] = normalize_scores(
    original_scores
)

future["Preseason_Score"] = normalize_scores(
    preseason_scores
)

# Combine with equal weights
future["Combined_Score"] = (
    future["Original_Score"] * 0.5
    + future["Preseason_Score"] * 0.5
)

# Rank teams
future = future.sort_values(
    "Combined_Score",
    ascending=False
).reset_index(drop=True)

future["Combined_Rank"] = future.index + 1

print()
print("==============================")
print("2026-27 COMBINED PREDICTIONS")
print("==============================")

print(
    future[
        [
            "Combined_Rank",
            "Team",
            "Original_Score",
            "Preseason_Score",
            "Combined_Score"
        ]
    ].round(2).to_string(index=False)
)

future.to_csv(
    "nba_2026_27_combined_predictions.csv",
    index=False
)


# STEP 49: CHECK ZERO SCORES

print()
print("==============================")
print("ZERO SCORE ANALYSIS")
print("==============================")

print(
    future[
        [
            "Team",
            "Original_Score",
            "Preseason_Score",
            "Combined_Score"
        ]
    ].tail(10).to_string(
        index=False,
        float_format=lambda x: f"{x:.8f}"
    )
)

print()
print(
    "Exact zero combined scores:",
    (future["Combined_Score"] == 0).sum()
)

print(
    "Positive combined scores:",
    (future["Combined_Score"] > 0).sum()
)