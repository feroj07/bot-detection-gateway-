
🛡️ Bot Detection Gateway
A smart, CAPTCHA-free bot detection system that uses mouse movements and typing behavior to tell real humans apart from automated bots on a login page.

🔗 Live Demo: https://bot-detection-gateway-production.up.railway.app/

🚀 Live Demo & How to Test
Open the Live Demo
 and try these 3 scenarios:

Genuine Human Login:
Enter username demo and password Demo@123.
Move your mouse and type normally → You get ACCESS GRANTED (ALLOW).
Simulate Bot Attack:
Click Demo tools → Simulate bot attack.
The system detects straight-line robotic mouse paths and blocks access → ACCESS DENIED (BLOCK).
Simulate Selenium Bot:
Click Demo tools → Simulate Selenium bot.
Automated browser flags (navigator.webdriver) and honeypot traps trigger an instant block.
📸 Screenshots
1. Login Page
Clean login portal with background behavioral biometric tracking. Preview unavailable

2. Human vs. Bot Decision Dashboard
🟢 Human Allowed (Risk: 16.4%)	🔴 Bot Blocked (Risk: 95.0%)
Preview unavailable	Preview unavailable
Natural curves, varied speed, real human typing rhythm.	Rigid straight lines, no pauses, robotic timing.
3. User Database & History Logs
User Database (users.db)	Access Attempt History
Preview unavailable	Preview unavailable
✨ Key Features
Zero CAPTCHAs: Completely invisible and seamless for genuine users.
27 Behavioral Features: Analyzes mouse path curvature, speed, acceleration, typing cadence, and key hold/flight times.
3-Tier Decision Engine:
Risk < 0.35: ALLOW (Access Granted)
0.35 to 0.70: CHALLENGE (Math question verification)
Risk > 0.70: BLOCK (Access Denied)
High Accuracy: Trained Gradient Boosting model achieves 98.67% accuracy and 0.9997 ROC-AUC.
Explainable AI: Shows mouse path visualizer, speed charts, and exact biometric calculations behind every decision.
Security Checks: Hidden honeypot fields and navigator.webdriver detection.
Online Retraining: Collects real human sessions and retrains the model directly from /data.
💻 Quick Start (Run Locally)
Requires Python 3.10+.

bash


# 1. Clone repository
git clone https://github.com/feroj07/bot-detection-gateway-.git
cd bot-detection-gateway-
# 2. Install dependencies
pip install -r requirements.txt
# 3. Train model & generate charts
python pipeline.py
# 4. Start web gateway
python app.py
Open http://127.0.0.1:5000
 in your browser.

Demo Credentials:

Username: demo | Password: Demo@123
Username: admin | Password: Admin@123
📁 Project Structure
app.py — Flask gateway, authentication, and decision dashboard.
pipeline.py — Data simulation, feature extraction, and ML model training.
model.joblib — Trained Gradient Boosting model.
users.db — SQLite database for user accounts and login logs.
screenshots/ — Images from the live deployment.
👤 Author
Firoj Ansari — @feroj07
Live App: bot-detection-gateway-production.up.railway.app
To apply it on GitHub:
Open 👉 Edit README on GitHub
Replace the text with the block above and click Commit changes.
Upload the screenshot files from your local folder 
screenshots/
 using Add file → Upload files on GitHub.

