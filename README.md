# NBA Championship Predictor (2026–27)

## Project Overview

This project uses Python and machine learning to predict which NBA team is most likely to win the 2026–27 NBA Championship.

I built this project to explore how historical team performance, player statistics, and offseason roster changes can be used to predict NBA championship contenders.

## Tools and Technologies

- Python
- pandas and NumPy
- scikit-learn
- NBA API
- Basketball Reference data
- Git and GitHub

## Data Collection

The project uses NBA team statistics from the 2015–16 through 2025–26 seasons, along with player statistics and 2026–27 roster data.

Some of the statistics used include:

- Wins (W)
- Net Rating (NRtg)
- Simple Rating System (SRS)
- Margin of Victory (MOV)
- Offensive and Defensive Ratings
- Effective Field Goal Percentage (eFG%)
- True Shooting Percentage (TS%)

## Machine Learning Models

I developed two Random Forest models:

1. **Historical Performance Model:** Uses team statistics from a season to identify championship contenders.
2. **Preseason Model:** Uses the previous season's statistics to predict the following season's champion.

The final model combines the normalized scores from both models using equal weights.

## Offseason Roster Adjustments

To account for changes between seasons, I collected player statistics and updated team rosters.

I analyzed roster changes using statistics such as Box Plus/Minus (BPM), Value Over Replacement Player (VORP), and Win Shares (WS).

I then applied a small adjustment to each team's previous-season statistics to reflect estimated roster improvements or losses.

These adjustments are estimates rather than predictions learned directly from historical roster changes.

## Model Evaluation

I evaluated the combined model using historical seasons, training on earlier seasons before predicting later ones.

| Metric | Result |
|---|---|
| Top-1 Accuracy | 22.2% |
| Top-2 Accuracy | 44.4% |
| Top-3 Accuracy | 44.4% |
| Top-5 Accuracy | 55.6% |
| Average Actual Champion Rank | 8.22 |

These results measure the historical combined model without the offseason roster adjustments.

## 2026–27 Championship Predictions

The top five teams in the final rankings are:

| Rank | Team | Normalized Model Score |
|---|---|---|
| 1 | New York Knicks | 14.18% |
| 2 | Oklahoma City Thunder | 13.92% |
| 3 | San Antonio Spurs | 10.00% |
| 4 | Boston Celtics | 9.96% |
| 5 | Denver Nuggets | 8.68% |

The scores are normalized model outputs, **not calibrated championship-winning probabilities**.

The full rankings are available in `nba_2026_27_combined_predictions.csv`.

## Limitations

- The historical dataset contains a relatively small number of championship-winning teams.
- The models rely heavily on previous-season statistics.
- Offseason roster adjustments are estimates.
- Injuries, coaching changes, and unexpected player development are not fully accounted for.
- The normalized scores should not be interpreted as exact probabilities.

## Future Improvements

- Incorporate injury information and projected player minutes.
- Test additional machine learning models.
- Improve roster adjustment estimates.
- Add more historical seasons to the training dataset.
- Build visualizations to compare championship contenders.