# Bot Detection Gateway: Behavioural Biometrics for Login Pages

A behavioural-biometrics classifier that tells **automated bot traffic** apart from **genuine users** on a login page, using mouse-movement and keystroke-timing logs. Results are presented through a login-page **bot-mitigation gateway** that returns an *allow*, *challenge* or *block* decision, with a dashboard that explains every calculation behind the decision.

## Features

- **Data pipeline:** simulated human and bot sessions (plus real human sessions collected from the login page), data cleaning, feature extraction, model training and evaluation.
- **27 behavioural features:** cursor path (straightness, curvature), speed and acceleration, pauses, typing rhythm, key hold and flight time, clicks and scrolling.
- **Three models compared:** Logistic Regression, Random Forest, Gradient Boosting. Metrics: accuracy, precision, recall, F1, ROC-AUC, false-positive rate, confusion matrix, ROC curves, feature importance.
- **Data cleaning stage:** duplicate removal, missing-value filling, empty-session removal, outlier capping (1st to 99th percentile), with a printed and saved report.
- **Authentication:** sign in and new-user sign-up on the first page, users stored in a SQLite database (`users.db`) with hashed passwords. Wrong password and unknown account show clear messages.
- **Gateway decision:** risk below 0.35 = allow, 0.35 to 0.70 = challenge (maths question), above 0.70 = block.
- **Automation checks:** hidden honeypot field, `navigator.webdriver` flag and headless-browser signature, which can raise the risk on their own.
- **Explainable dashboard:** cursor path, speed graph, typing and key-hold charts, the strongest evidence, and a table of all 27 calculations compared with the human and bot averages.
- **Real data collection and retraining:** tick "I am a real person" when signing in to save a session, then retrain from the Training data page.

## How it works

```
Login page  ->  1. Authentication (username + password, SQLite)
            ->  2. Behaviour check (mouse + keystroke features -> ML model, plus honeypot/webdriver rules)
            ->  3. Decision: ALLOW / CHALLENGE / BLOCK  ->  Dashboard with the calculations
```

## Quick start

Requires Python 3.10 or newer.

```bash
pip install -r requirements.txt
python pipeline.py      # generates data, cleans it, trains models, saves model.joblib and charts
python app.py           # starts the gateway
```

Open http://127.0.0.1:5000

On Windows you may need `py -3.12` instead of `python`.

**Demo account:** username `demo`, password `Demo@123` (or create your own on the **New user** tab).

## Try the demo

1. Move your mouse, type, and sign in: you should see **HUMAN**, or the maths challenge if the behaviour is uncertain.
2. Open **Demo tools** and click **Simulate bot attack**: **BOT**, access denied.
3. Click **Simulate Selenium bot**: the model sees human-like moves, but the automation checks block it.
4. Open **User database** to see the saved accounts and the login log.

## Project structure

| File | Purpose |
|---|---|
| `pipeline.py` | Data simulation, cleaning, feature extraction, training, evaluation, charts |
| `app.py` | Flask gateway: authentication, scoring, challenge, dashboard, retraining |
| `requirements.txt` | Python libraries |
| `Procfile` | Start command for hosting (for example Render) |

Generated when you run the pipeline (not stored in the repository): `model.joblib`, `dataset.csv`, `results.csv`, `cleaning_report.csv`, `roc_curves.png`, `confusion_matrix.png`, `feature_importance.png`, `cleaning_summary.png`.

## Results

Run `python pipeline.py` to print the table and create `results.csv`. On the simulated test set the best model (Gradient Boosting or Random Forest) reaches roughly 98% accuracy with ROC-AUC around 0.999, detecting all simple bots and about 98% of smart bots that imitate human noise.

## Limitations

- The training data is **simulated**, so scores are optimistic. Real traffic will behave differently.
- Real user data is collected only for humans; real bot data would need a separate collection.
- The wrong-password message names the exact problem (account missing or password wrong). Production systems use one generic message.
- The gateway needs a mouse, so it does not score properly on phones.

## Future work

- Validate and retrain on large real user logs, including real bot traffic.
- Add sequence models (LSTM) on raw mouse paths, and touch and swipe data for mobile.
- Monitor drift and retrain automatically as bots change behaviour.
- Add rate limiting, account lockout and a stronger CAPTCHA.

## Screenshots

Add screenshots here: login page, human dashboard, bot dashboard, challenge page, user database page.
