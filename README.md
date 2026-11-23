
🛡️ Bot Detection Gateway: Behavioural Biometrics for Login Pages

Live Demo
Python
Flask
Scikit-Learn
SQLite

An intelligent, explainable bot-mitigation gateway that analyzes client-side behavioural biometrics (mouse cursor paths, speed variations, typing cadence, and key hold/flight intervals) combined with browser integrity heuristics to distinguish automated bot traffic from genuine users in real-time.

🌐 Live Application & Demo

Try the live production deployment hosted on Railway:

👉 https://bot-detection-gateway-production.up.railway.app/

🔑 Test Credentials & Quick Demo:
Demo Account: Username: demo | Password: Demo@123
Admin Account: Username: admin | Password: Admin@123
(You can also create your own custom account via the New user tab on the login page)
🧪 What to Try on the Live Demo:
Genuine Human Login: Move your mouse naturally, type into the fields, and sign in. You will see ACCESS GRANTED (ALLOW) with a low bot risk score and smooth trajectory visualizer.
Simulate Bot Attack (Linear): Click Demo tools (for presentation) → Simulate bot attack. Watch the gateway immediately return ACCESS DENIED (BLOCK) due to robotic straight-line cursor movement, lack of pauses, and instantaneous key taps.
Simulate Selenium Headless Bot: Click Simulate Selenium bot. Even with synthetic Bézier curvature imitation, browser automation flags (navigator.webdriver & honeypot traps) trigger an immediate block.
Audit Trail & Forensics: Inspect /users for registered accounts and /history for real-time audit logs of all access decisions.
📸 System Screenshots & Visual Walkthrough
1. Secure Access Gateway (Login Interface)

The login portal seamlessly captures client-side cursor movements, typing intervals, clicks, and scrolls in the background without requiring disruptive CAPTCHAs.
Secure Access Gateway

2. Detection Dashboards: Human vs. Bot Verdict
Genuine User Decision (ALLOW — Access Granted)	Automated Bot Attack (BLOCK — Access Denied)

Human Allowed Dashboard
	
Bot Blocked Dashboard

Verdict: HUMAN (Risk: 16.4%)
Features organic mouse curvature, dynamic acceleration/deceleration, human typing cadence, and zero automation signals.	Verdict: BOT (Risk: 95.0%)
Flags robotic straight-line pathing, lack of natural pause ratios, abnormal typing flight time, and automation signatures.
3. User Database & Comprehensive Audit Log (/users)

User credentials are protected using one-way cryptographic password hashing (scrypt). The gateway maintains a persistent log of every login attempt, recording credentials status, risk score, decision, and outcome.
User Database and Login Log

4. Interactive Training Data & Retraining Pipeline (/data)

Enables collecting verified real human login sessions and triggering online retraining directly from the web interface.
Training Data Pipeline

5. Access Attempt History (/history)

An audit trail recording chronological login requests with direct links to view detailed biometric calculation dashboards for each session.
Access History Logs

⚙️ How the Gateway Works
                     ┌─────────────────────────────────────────┐
                     │          Client Login Request           │
                     │  (Credentials + Behavioural Telemetry)  │
                     └────────────────────┬────────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │   Stage 1: Authentication Check     │
                       │   (SQLite users.db, scrypt hash)    │
                       └──────────────────┬──────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │   Stage 2: Heuristic Security Rules │
                       │  • Honeypot field trap              │
                       │  • navigator.webdriver check        │
                       │  • HeadlessChrome User-Agent        │
                       └──────────────────┬──────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │   Stage 3: Feature Extraction       │
                       │  27 mouse, keystroke & timing feats │
                       └──────────────────┬──────────────────┘
                                          │
                                          ▼
                       ┌─────────────────────────────────────┐
                       │   Stage 4: ML Classifier Scoring    │
                       │     (Trained Gradient Boosting)     │
                       └──────────────────┬──────────────────┘
                                          │
                   ┌──────────────────────┴──────────────────────┐
                   │                                             │
             Risk < 0.35                                   Risk > 0.70
                   ▼                                             ▼
        ┌──────────────────────┐                     ┌──────────────────────┐
        │  🟢 Verdict: ALLOW   │                     │   🔴 Verdict: BLOCK  │
        │    ACCESS GRANTED    │                     │    ACCESS DENIED     │
        └──────────────────────┘                     └──────────────────────┘
                   │                                             │
                   └──────────────────┬──────────────────────────┘
                                      │
                               0.35 ≤ Risk ≤ 0.70
                                      ▼
                           ┌──────────────────────┐
                           │ 🟡 Verdict: CHALLENGE│
                           │   Maths Verification │
                           └──────────────────────┘
📊 27 Behavioural Biometric Features

The pipeline extracts 27 high-signal behavioural features across four core dimensions:

Category	Features Extracted	Description & Significance
Cursor Path Geometry	straightness, mean_curvature, std_curvature, mean_angle_change, path_length	Measures deviations from a straight line. Bots move along mathematical vectors; humans exhibit natural path tremors and variable curvature.
Velocity & Dynamics	mean_speed, std_speed, max_speed, speed_cv, mean_acc, std_acc, max_acc	Analyzes velocity profiles and acceleration curves. Humans follow bell-shaped velocity profiles (Fitts's Law); bots maintain fixed or step speeds.
Pauses & Idle Gaps	n_pauses, pause_ratio, duration, n_points	Quantifies hesitation and pauses before actions. Human thinking time creates pauses; bots execute sequentially without hesitation.
Keystroke Dynamics	key_count, key_hold_mean, key_hold_std, key_flight_mean, key_flight_std, key_mean_interval, key_std_interval	Measures key press duration (hold time) and inter-key latency (flight time). Human typing is rhythmically heterogeneous; scripted typing is rigid.
Interaction Events	click_count, click_mean_interval, scroll_count, scroll_distance	Captures micro-interactions including mouse clicks, intervals between double clicks, and scroll gestures.
🔬 Machine Learning Evaluation & Results

Three supervised learning models were trained, tuned, and evaluated on a benchmark dataset containing simulated bots (simple linear bots, randomized bots, and advanced human-imitating bots) and genuine user sessions:

Model	Accuracy	Precision	Recall	F1-Score	ROC-AUC	False Positive Rate (FPR)
Gradient Boosting ⭐	98.67%	97.90%	99.47%	98.68%	0.9997	2.13%
Random Forest	98.00%	96.39%	99.73%	98.03%	0.9994	3.73%
Logistic Regression	97.47%	95.64%	99.47%	97.52%	0.9971	4.53%
📈 Model Evaluation Visualizations
Feature Importance (Top Predictors)	ROC Curves (Model Comparison)

Feature Importance
	
ROC Curves

Key Predictors: Pause count, pause ratio, curvature variance, and key hold duration provide the strongest discriminative power.	ROC Performance: All three models demonstrate near-perfect area under the curve, with Gradient Boosting reaching 0.9997 ROC-AUC.
Confusion Matrix (Gradient Boosting)	Data Cleaning & Outlier Management

Confusion Matrix
	
Data Cleaning Summary

High Sensitivity: Accurately catches 99.5% of bot intrusions while maintaining a minimal false alarm rate on human sessions.	Preprocessing: Automated deduplication, median imputation, empty session pruning, and 1st–99th percentile outlier capping.
💡 Explainable AI & Forensic Transparency

Unlike black-box commercial WAFs, this gateway offers complete mathematical transparency on every access decision:

Interactive Trajectory Map: Visualizes the user's SVG mouse track against the theoretical direct vector (dashed line).
Speed Over Time Graph: Plots acceleration peaks and hesitation valleys.
Biometric Metric Comparison: Displays every individual feature compared against the human benchmark and bot baseline, scaled by standard deviations.
Multi-signal Rule Audit: Confirms state of honeypot traps, navigator.webdriver execution tags, and headless browser user agents.
🚀 Getting Started & Local Setup
Prerequisites
Python 3.10 or newer
Git
Installation

Clone the repository:

bash
git clone https://github.com/feroj07/bot-detection-gateway-.git
cd bot-detection-gateway-

Create and activate a virtual environment:

bash
# Windows (PowerShell)
python -m venv .venv
.venv\Scripts\Activate.ps1
# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate

Install dependencies:

bash
pip install -r requirements.txt

Run the data pipeline & train models:

bash
python pipeline.py

This generates synthetic and real sessions, cleans the dataset, trains the ML classifiers, evaluates metrics, and saves model.joblib and evaluation charts.

Start the Bot Detection Gateway:

bash
python app.py

Open in your browser: Navigate to http://127.0.0.1:5000

🐳 Docker & Cloud Deployment
Run with Docker:
bash
docker build -t bot-detection-gateway .
docker run -p 5000:5000 bot-detection-gateway
Deploy to Railway / Render:
Procfile: web: gunicorn app:app
Port Configuration: Automatically binds to the $PORT environment variable provided by Railway/Render.
📁 Repository Structure
bot-detection-gateway/
├── app.py                      # Flask web application & bot detection decision gateway
├── pipeline.py                 # Data simulation, preprocessing, feature extraction & ML training
├── requirements.txt            # Python dependencies (Flask, scikit-learn, pandas, numpy, etc.)
├── Procfile                    # Production server startup configuration for Railway / Render
├── Dockerfile                  # Containerization specification
├── render.yaml                 # Infrastructure-as-code deployment configuration
├── users.db                    # SQLite database storing accounts (scrypt hashes) & login audit logs
├── model.joblib                # Serialized trained ML model bundle and feature scaler
├── dataset.csv                 # Preprocessed dataset of extracted behavioural features
├── results.csv                 # Model evaluation metrics benchmark table
├── cleaning_report.csv         # Data cleaning and outlier capping summary log
├── screenshots/                # Application UI screenshots and ML evaluation plots
│   ├── 01_login_page.png
│   ├── 02_bot_blocked_dashboard.png
│   ├── 03_human_allowed_dashboard.png
│   ├── 04_user_database.png
│   ├── 05_training_data_pipeline.png
│   ├── 06_history_logs.png
│   ├── cleaning_summary.png
│   ├── confusion_matrix.png
│   ├── feature_importance.png
│   └── roc_curves.png
└── Bot_Detection_Mini_Project_Report.docx  # Full project research report & documentation
🛡️ Limitations & Future Improvements
Mobile Touch Dynamics: The current classifier evaluates cursor movements and mouse events. Future iterations can integrate swipe curvature, touch pressure, and touch contact area for smartphones and tablets.
Deep Sequence Modeling: Complementing tabular features with deep recurrent networks (LSTM / Transformers) trained directly on raw coordinate time-series vectors.
Adaptive Drift Mitigation: Implementing online learning loops that continually adapt decision boundaries as automated adversary scripts evolve.
Rate-Limiting & IP Intelligence: Combining biometric signals with IP reputation heuristics and progressive tarpitting.
👤 Author
Firoj Ansari — @feroj07
Repository: feroj07/bot-detection-gateway-
Live Demo: bot-detection-gateway-production.up.railway.app
