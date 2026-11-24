# 🛡️ Bot Detection Gateway

A Flask web app that uses mouse movements and typing patterns to distinguish human login attempts from automated bots. It assigns each attempt a risk score and decides whether to allow it, show a verification challenge, or block it.

## 🔗 Live Demo

[Open Bot Detection Gateway](https://bot-detection-gateway-production.up.railway.app/)

## 🚀 How to Use

1. Open the live demo.
2. Sign in with username `demo` and password `Demo@123`.
3. Move your mouse and type normally to try a regular login.
4. Open **Demo tools** to try the built-in bot and Selenium simulations.
5. Check the decision dashboard to see the risk score and the signals behind the result.

## ✨ Features

- Analyzes 27 behavioral features, including mouse speed, path curvature, and typing intervals.
- Uses a Gradient Boosting model to score login attempts.
- Returns one of three decisions: **Allow**, **Challenge**, or **Block**.
- Shows mouse paths and speed charts on the decision dashboard.
- Uses a honeypot field and browser automation checks to help detect bots.
- Stores user accounts and login attempt history in SQLite.

## 📦 Requirements

- Python 3.10 or newer
- Python dependencies listed in `requirements.txt`

## 👨‍💻 Developed By

**Shaik Feroj**  
B.Tech — Artificial Intelligence & Data Science

[GitHub](https://github.com/feroj07)

⭐ If you find this project useful, consider starring the repository!
