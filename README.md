![](https://github.com/senselogic/NEXT/blob/master/LOGO/next.png)

# Next

Constrained activity schedule forecasting with deep learning.

## Description

Next is a time series forecasting tool designed for predicting structured activity schedules with temporal and categorical constraints. It leverages deep learning to generate realistic activity timelines while enforcing strict business rules such as non-overlapping intervals, valid time ranges, and categorical consistency.

## Features

- **Transformer-based forecasting** : Uses Temporal Fusion Transformer (TFT) with probabilistic sampling via quantile regression for uncertainty-aware forecasts.
- **Multi-step prediction** : Supports forecasting for N future days while maintaining realistic daily patterns and temporal relationships.
- **Constraint enforcement** : Automatically enforces interval consecutivity and non-overlap through a post-processing scheduler that validates and corrects predictions.
- **Categorical handling** : Handles structured categorical inputs (category-subject pairs) using temporal and cyclical covariates.
- **Temporal awareness** : Incorporates time-of-day, day-of-week, and month features as covariates to capture seasonal and weekly patterns.
- **Data validation** : Comprehensive validation ensures all generated activities comply with temporal constraints, subject definitions, and boundary conditions.

## Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install "u8darts[all]" pandas numpy
```

## Usage

### Train a model:

```bash
python next.py train {model_name} {subject data file path} {past activity data file path}
```

Example:
```bash
python next.py train daily_activity user_input_files/daily_activity_subject_data.csv user_input_files/daily_activity_data.csv
```

### Predict future activities for N days:

```bash
python next.py predict {model_name} {subject data file path} {future activity data file path} {day count}
```

Example:
```bash
python next.py predict daily_activity user_input_files/daily_activity_subject_data.csv predicted_daily_activity_data.csv 7
```

## Examples

### Daily Activities
```bash
python next.py train daily_activity user_input_files/daily_activity_subject_data.csv user_input_files/daily_activity_data.csv
python next.py predict daily_activity user_input_files/daily_activity_subject_data.csv predicted_daily_activity_data.csv 7
```

### YouTube Video Activity
```bash
python next.py train youtube_video user_input_files/youtube_video_subject_data.csv user_input_files/youtube_video_activity_data.csv
python next.py predict youtube_video user_input_files/youtube_video_subject_data.csv predicted_youtube_video_activity_data.csv 5
```

## Subject data

Subject data files start with a header and have the following columns :
- category (double-quoted string)
- subject (double-quoted string)

Sample daily activities subject data :
```csv
"category","subject"
"leisure","playing video game"
"leisure","watching television"
"leisure","watching tv series"
"leisure","reading"
"leisure","browsing internet"
"leisure","listening to music"
"sport","playing tennis"
"sport","running"
"sport","going to gym"
"sport","cycling"
"transport","driving"
"transport","commuting"
"work","developing application"
"work","code review"
"work","debugging"
"work","writing documentation"
"work","managing team"
"work","attending a meeting"
"work","calling a customer"
"work","planning sprint"
"rest","sleeping"
"meal","eating breakfast"
"meal","eating lunch"
"meal","eating dinner"
"meal","eating snack"
"meal","drinking coffee"
"hygiene","brushing teeth"
"hygiene","taking a shower"
"hygiene","washing hands"
```

Sample watched YouTube video subject data :
```csv
"category","subject"
"sport","tennis"
"language","spanish"
"history","medieval times"
"science","biology"
"technology","programming"
"music","jazz"
"travel","japan"
"gaming","strategy games"
```

## Activity data

Activity data files start with a header and have the following columns :
- year (integer)
- year_day (0-364 integer for non-leap years, 0-365 integer for leap years)
- week_day (0-6 integer)
- hour (0-23 integer)
- minute (0-55 integer, multiple of 5)
- duration (5-1440 integer, multiple of 5)
- category (double-quoted string)
- subject (double-quoted string)

Enforced constraints :
- Year day values must not exceed 365 for leap years, and 364 for non-leap years.
- Week day values must not exceed 6.
- Hour values must be between 0 and 23.
- Minute values must be between 0 and 55, and must be a multiple of 5.
- Times and durations are multiples of 5 minutes.
- Duration values must not exceed 1440.
- Activities must not have interval overlap.
- All (category, subject) pairs must belong to the subject table.

Sample daily activity data :
```csv
"year","year_day","week_day","hour","minute","duration","category","subject"
2025,100,2,7,0,15,"hygiene","taking a shower"
2025,100,2,7,15,5,"hygiene","brushing teeth"
2025,100,2,7,30,15,"meal","eating breakfast"
2025,100,2,9,0,180,"work","developing application"
2025,100,2,12,30,30,"meal","eating lunch"
2025,100,2,13,30,240,"work","attending a meeting"
```

Sample watched YouTube video activity data :
```csv
"year","year_day","week_day","hour","minute","duration","category","subject"
2025,100,2,18,45,30,"technology","programming"
2025,100,2,19,15,30,"science","biology"
2025,100,2,19,45,30,"language","spanish"
2025,101,3,18,45,30,"technology","programming"
2025,101,3,19,15,30,"history","medieval times"
2025,101,3,19,45,30,"gaming","strategy games"
2025,102,4,18,45,30,"technology","programming"
2025,102,4,19,15,30,"science","biology"
2025,102,4,19,45,30,"language","spanish"
2025,103,5,10,0,30,"sport","tennis"
2025,103,5,20,0,30,"music","jazz"
2025,103,5,21,0,30,"travel","japan"
```

Input and output activity data files share the same format, structure and constraints.

## Version

0.1

## Author

Eric Pelzer (ecstatic.coder@gmail.com).

## License

This project is licensed under the GNU General Public License version 3.

See the [LICENSE.md](LICENSE.md) file for details.
