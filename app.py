"""Login-page gateway + dashboard + real-data collection + retraining.
Run pipeline.py first (creates model.joblib and dataset.csv), then:  python app.py
"""
import os
import csv
import random
import re
import sqlite3
import json
import time
import joblib
import numpy as np
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
from flask import (Flask, request, jsonify, render_template_string, redirect,
                   send_file, abort)
import pipeline
from pipeline import FEATURES, prepare_session, RAW_FILE, real_count

CSV_FILE = "real_sessions.csv"
DB_FILE = "users.db"
OLD_USERS_FILE = "users.json"      # imported once if it exists
DEMO_USERS = {"demo": "Demo@123", "admin": "Admin@123"}      # created on first run
app = Flask(__name__)
attempts, store = [], {}
ALLOW_BELOW, BLOCK_ABOVE = 0.35, 0.70
model = MODEL_NAME = CLIP = None
HUMAN_AVG, BOT_AVG, SCALE = {}, {}, {}


def load_all():
    """(Re)load the trained model and the human/bot averages used on the dashboard."""
    global model, MODEL_NAME, CLIP, HUMAN_AVG, BOT_AVG, SCALE
    bundle = joblib.load("model.joblib")
    if list(bundle["features"]) != list(FEATURES):
        raise SystemExit("model.joblib is from an older version. "
                         "Run  python pipeline.py  once, then start the app again.")
    model, MODEL_NAME = bundle["model"], bundle["name"]
    CLIP = bundle.get("clip")          # outlier caps learned during cleaning
    if os.path.exists("dataset.csv"):
        d = pd.read_csv("dataset.csv")
        HUMAN_AVG = d[d.label == 0][FEATURES].mean().to_dict()
        BOT_AVG = d[d.label == 1][FEATURES].mean().to_dict()
        SCALE = d[FEATURES].std().replace(0, 1).to_dict()


load_all()


def db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Creates users.db with a users table and a login_log table (first run only)."""
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created TEXT NOT NULL,
            last_login TEXT,
            failed_logins INTEGER DEFAULT 0)""")
        c.execute("""CREATE TABLE IF NOT EXISTS login_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            time TEXT, username TEXT, credentials_ok INTEGER,
            bot_risk REAL, decision TEXT, outcome TEXT)""")
        if c.execute("SELECT COUNT(*) FROM users").fetchone()[0] == 0:
            seed = {}
            if os.path.exists(OLD_USERS_FILE):                 # move old JSON users over
                with open(OLD_USERS_FILE) as fh:
                    seed = json.load(fh)
                for u, h in seed.items():
                    c.execute("INSERT INTO users (username, password_hash, created) VALUES (?,?,?)",
                              (u, h, time.strftime("%Y-%m-%d %H:%M:%S")))
            else:
                for u, p in DEMO_USERS.items():
                    c.execute("INSERT INTO users (username, password_hash, created) VALUES (?,?,?)",
                              (u, generate_password_hash(p), time.strftime("%Y-%m-%d %H:%M:%S")))


def add_user(username, password):
    try:
        with db() as c:
            c.execute("INSERT INTO users (username, password_hash, created) VALUES (?,?,?)",
                      (username, generate_password_hash(password), time.strftime("%Y-%m-%d %H:%M:%S")))
        return True
    except sqlite3.IntegrityError:
        return False


def check_login(username, password):
    """Returns (ok, message). Passwords are stored hashed, never in plain text."""
    username = (username or "").strip().lower()
    if not username or not password:
        return False, "Enter both a username and a password."
    with db() as c:
        row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
        if row is None:
            return False, "Account does not exist. Please create a new account."
        if not check_password_hash(row["password_hash"], password):
            c.execute("UPDATE users SET failed_logins = failed_logins + 1 WHERE username=?", (username,))
            return False, "Incorrect password. Please try again."
        c.execute("UPDATE users SET last_login=?, failed_logins=0 WHERE username=?",
                  (time.strftime("%Y-%m-%d %H:%M:%S"), username))
    return True, f"Welcome, {username}. Login successful."


def log_attempt(username, cred_ok, risk, decision, outcome):
    with db() as c:
        c.execute("INSERT INTO login_log (time, username, credentials_ok, bot_risk, decision, outcome) "
                  "VALUES (?,?,?,?,?,?)", (time.strftime("%Y-%m-%d %H:%M:%S"), username,
                                           int(cred_ok), round(risk, 4), decision, outcome))


init_db()


def evidence(n, val):
    """Positive = closer to the human average, negative = closer to the bot average
    (difference of distances, measured in standard deviations)."""
    h, b, sc = HUMAN_AVG.get(n), BOT_AVG.get(n), SCALE.get(n, 1) or 1
    if h is None:
        return 0.0
    return (abs(val - b) - abs(val - h)) / sc


# label, unit, how it is calculated
INFO = {
    "duration": ("Mouse activity time", "s", "last timestamp - first timestamp"),
    "n_points": ("Mouse points recorded", "", "count of mouse move events"),
    "path_length": ("Cursor path length", "px", "sum of distances between consecutive points"),
    "straightness": ("Path straightness", "0-1", "straight distance start to end / path length (1 = perfect line)"),
    "mean_speed": ("Average speed", "px/s", "distance between points / time between points, averaged"),
    "std_speed": ("Speed variation (std)", "px/s", "standard deviation of the speeds"),
    "max_speed": ("Top speed", "px/s", "largest speed value"),
    "speed_cv": ("Speed consistency (CV)", "", "std of speed / mean speed (bots are very low)"),
    "mean_acc": ("Average acceleration", "px/s2", "change in speed / time, absolute, averaged"),
    "std_acc": ("Acceleration variation", "px/s2", "standard deviation of acceleration"),
    "max_acc": ("Peak acceleration", "px/s2", "largest absolute acceleration"),
    "mean_curvature": ("Average curvature", "rad/px", "change of direction / distance, averaged"),
    "std_curvature": ("Curvature variation", "rad/px", "standard deviation of curvature"),
    "mean_angle_change": ("Average turn angle", "rad", "mean absolute change of direction between steps"),
    "n_pauses": ("Pauses in mouse", "", "gaps longer than 0.15 s"),
    "pause_ratio": ("Pause share of time", "0-1", "total pause time / total time"),
    "key_count": ("Keystrokes", "", "number of key presses"),
    "key_mean_interval": ("Average typing gap", "s", "mean time between key presses"),
    "key_std_interval": ("Typing rhythm variation", "s", "standard deviation of the gaps (bots are very low)"),
    "key_hold_mean": ("Key hold time (average)", "s", "key release time - key press time, averaged"),
    "key_hold_std": ("Key hold variation", "s", "standard deviation of hold times (bots are very low)"),
    "key_flight_mean": ("Key flight time (average)", "s", "next key press - previous key release, averaged"),
    "key_flight_std": ("Key flight variation", "s", "standard deviation of flight times"),
    "click_count": ("Mouse clicks", "", "number of mouse button presses"),
    "click_mean_interval": ("Average time between clicks", "s", "mean gap between clicks"),
    "scroll_count": ("Scroll events", "", "number of mouse-wheel events"),
    "scroll_distance": ("Scroll distance", "px", "sum of absolute wheel movement"),
}
GROUPS = [("Cursor path", ["duration", "n_points", "path_length", "straightness",
                           "mean_curvature", "std_curvature", "mean_angle_change"]),
          ("Speed and acceleration", ["mean_speed", "std_speed", "max_speed", "speed_cv",
                                      "mean_acc", "std_acc", "max_acc"]),
          ("Pauses", ["n_pauses", "pause_ratio"]),
          ("Typing", ["key_count", "key_mean_interval", "key_std_interval", "key_hold_mean",
                      "key_hold_std", "key_flight_mean", "key_flight_std"]),
          ("Clicks and scrolling", ["click_count", "click_mean_interval",
                                    "scroll_count", "scroll_distance"])]


RULES = [("honeypot", "Honeypot field", 0.99,
          "A hidden form field that humans never see was filled in."),
         ("webdriver", "Browser automation flag", 0.90,
          "navigator.webdriver is true, which tools such as Selenium and Puppeteer set."),
         ("headless", "Headless browser signature", 0.85,
          "The browser identifies itself as a headless (no-window) browser.")]


def check_rules(sig):
    """Rule-based checks that run next to the model. Returns (checks, rule_risk)."""
    checks, rule_risk = [], 0.0
    for key, name, risk, detail in RULES:
        hit = bool(sig.get(key))
        checks.append({"name": name, "hit": hit, "detail": detail if hit else "Clean, no sign of automation."})
        if hit:
            rule_risk = max(rule_risk, risk)
    return checks, rule_risk


def analyse(raw, sig):
    m, downs, ups, feats = prepare_session(raw)
    checks, rule_risk = check_rules(sig)
    if len(m) < 3:
        model_risk = 0.95                              # no mouse movement at all
    else:
        x = pd.DataFrame([feats], columns=FEATURES)
        if CLIP:                                       # same outlier caps as in training
            x = x.clip(lower=pd.Series(CLIP["low"]), upper=pd.Series(CLIP["high"]), axis=1)
        model_risk = float(model.predict_proba(x)[0, 1])
    return m, downs, ups, feats, model_risk, rule_risk, checks


def decide(risk):
    if risk < ALLOW_BELOW:
        return "allow", "HUMAN", "Looks like a real person. Access granted."
    if risk <= BLOCK_ABOVE:
        return "challenge", "SUSPICIOUS", "Uncertain. Ask for a CAPTCHA or extra verification."
    return "block", "BOT", "Automated behaviour detected. Access denied."


def series(m, downs, ups):
    s = {"path": [], "speed": [], "key_gaps": [], "holds": []}
    if len(m) >= 3:
        step = max(1, len(m) // 400)
        s["path"] = [[round(a, 1), round(b, 1)] for a, b in m[::step, 1:3]]
        dt = np.maximum(np.diff(m[:, 0]), 1e-4)
        v = np.hypot(np.diff(m[:, 1]), np.diff(m[:, 2])) / dt
        s["speed"] = [[round(float(t), 3), round(float(x), 1)]
                      for t, x in zip(m[1:, 0] - m[0, 0], v)][::step]
    if len(downs) >= 2:
        s["key_gaps"] = [round(float(x), 3) for x in np.diff(downs)]
    if ups is not None and len(ups) and len(downs) == len(ups):
        s["holds"] = [round(float(x), 3) for x in np.clip(ups - downs, 0, None)]
    return s


def save_real(raw, feats):
    """Append one real human session: raw events (for retraining) and features (CSV)."""
    with open(RAW_FILE, "a") as fh:
        fh.write(json.dumps({"moves": raw.get("moves", []), "keys": raw.get("keys", []),
                             "clicks": raw.get("clicks", []), "scrolls": raw.get("scrolls", []),
                             "label": 0, "saved": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")
    new = not os.path.exists(CSV_FILE)
    with open(CSV_FILE, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["saved"] + FEATURES + ["label"])
        w.writerow([time.strftime("%Y-%m-%d %H:%M:%S")] + [round(x, 6) for x in feats] + [0])


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Secure Access</title>
<style>
body{font-family:Arial,sans-serif;background:#f1f3f6;display:flex;justify-content:center;padding-top:40px}
.card{background:#fff;padding:30px;border-radius:10px;width:390px;box-shadow:0 2px 10px #0002}
input[type=text],input[type=password]{width:100%;padding:10px;margin:8px 0;box-sizing:border-box}
button{width:100%;padding:11px;margin-top:8px;border:0;border-radius:6px;color:#fff;font-size:15px;cursor:pointer}
#go,#reg{background:#2563eb}#bot{background:#6b7280}#sel{background:#9333ea}
.tabs{display:flex;margin:6px 0 14px}
.tab{flex:1;padding:10px;text-align:center;cursor:pointer;border-bottom:3px solid #e5e7eb;color:#555}
.tab.on{border-color:#2563eb;color:#2563eb;font-weight:bold}
.pane{display:none}.pane.on{display:block}
.save{background:#f0fdf4;border:1px solid #bbf7d0;border-radius:6px;padding:10px;margin-top:10px;font-size:13px}
.err{background:#fee2e2;border:1px solid #fca5a5;color:#991b1b;border-radius:6px;padding:10px;margin-bottom:10px;font-size:14px}
.good{background:#dcfce7;border:1px solid #86efac;color:#166534;border-radius:6px;padding:10px;margin:8px 0;font-size:14px}
.hp{position:absolute;left:-9999px;top:-9999px;height:0;width:0;overflow:hidden}
details{margin-top:14px;font-size:13px;color:#555}
</style></head><body><div class="card">
<h2>Secure Access</h2>
{% if failed_msg %}<div class="err"><b>Login failed.</b> {{ failed_msg }}
 <br><small><a href="/dashboard/{{ failed_id }}">View behaviour analysis</a></small></div>{% endif %}
<div class="tabs"><div class="tab on" id="t1">Sign in</div><div class="tab" id="t2">New user</div></div>

<div class="pane on" id="p1">
 <div id="notice"></div>
 <p style="font-size:13px;color:#555;margin:0">Move your mouse and type normally. Demo account: <b>demo</b> / <b>Demo@123</b></p>
 <input type="text" id="u" placeholder="Username" autocomplete="off">
 <input type="password" id="p" placeholder="Password" autocomplete="off">
 <div class="hp" aria-hidden="true"><label>Leave this empty
  <input type="text" id="hp" name="contact_me_by_fax_only" tabindex="-1" autocomplete="off"></label></div>
 <label class="save" style="display:block"><input type="checkbox" id="save">
  I am a real person: save this attempt as training data<br>
  <small>Real sessions saved so far: <b>{{ n_real }}</b></small></label>
 <button id="go">Sign in</button>
</div>

<div class="pane" id="p2">
 <p style="font-size:13px;color:#555;margin:0">Create a new account, then sign in with it.</p>
 <input type="text" id="ru" placeholder="Choose a username (3-20 letters, numbers, _)" autocomplete="off">
 <input type="password" id="rp" placeholder="Choose a password (at least 6 characters)" autocomplete="off">
 <input type="password" id="rp2" placeholder="Confirm password" autocomplete="off">
 <button id="reg">Create account</button>
 <div id="rmsg"></div>
 <p><small>Saved in the database (users.db). Passwords are hashed, never plain text.</small></p>
</div>

<details><summary>Demo tools (for presentation)</summary>
 <button id="bot">Simulate bot attack (straight line)</button>
 <button id="sel">Simulate Selenium bot (human-like moves)</button>
</details>
<p><a href="/users">User database</a> &nbsp;|&nbsp; <a href="/history">History</a> &nbsp;|&nbsp; <a href="/data">Training data</a></p></div>
<script>
let moves=[], keys=[], clicks=[], scrolls=[], down={}, t0=performance.now();
const now=()=>(performance.now()-t0)/1000;
const $=id=>document.getElementById(id);
function resetCapture(){moves=[];keys=[];clicks=[];scrolls=[];down={};t0=performance.now();}
document.addEventListener('mousemove',e=>moves.push([now(),e.clientX,e.clientY]));
document.addEventListener('mousedown',()=>clicks.push(now()));
document.addEventListener('wheel',e=>scrolls.push([now(),e.deltaY]),{passive:true});
document.addEventListener('keydown',e=>{if(!e.repeat&&down[e.code]===undefined)down[e.code]=now();});
document.addEventListener('keyup',e=>{if(down[e.code]!==undefined){keys.push([down[e.code],now()]);delete down[e.code];}});
function tab(n){
 $('t1').className='tab'+(n===1?' on':'');$('t2').className='tab'+(n===2?' on':'');
 $('p1').className='pane'+(n===1?' on':'');$('p2').className='pane'+(n===2?' on':'');
 resetCapture();}
$('t1').onclick=()=>tab(1);$('t2').onclick=()=>tab(2);
$('reg').onclick=async()=>{
 const r=await (await fetch('/register',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({username:$('ru').value,password:$('rp').value,confirm:$('rp2').value})})).json();
 if(r.ok){const name=$('ru').value.trim().toLowerCase();
  $('ru').value='';$('rp').value='';$('rp2').value='';$('rmsg').innerHTML='';
  tab(1);$('u').value=name;$('p').value='';
  $('notice').innerHTML='<div class="good">'+r.message+'</div>';}
 else{$('rmsg').innerHTML='<div class="err" style="margin-top:8px">'+r.message+'</div>';}};
const signals=()=>({webdriver:!!navigator.webdriver,
 headless:/HeadlessChrome/i.test(navigator.userAgent),
 honeypot:$('hp').value.trim()!==''});
async function send(m,k,c,s,src,save,sig,cred){
 const cr=cred||{username:$('u').value,password:$('p').value};
 const res=await fetch('/score',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({moves:m,keys:k,clicks:c,scrolls:s,source:src,save:save,signals:sig,
   username:cr.username,password:cr.password})});
 const r=await res.json();
 if(r.decision==='challenge'){window.location='/challenge/'+r.id;}
 else if(r.final.indexOf('LOGIN FAILED')===0){window.location='/?failed='+r.id;}
 else{window.location='/dashboard/'+r.id;}}
$('go').onclick=()=>send(moves,keys,clicks,scrolls,'real user',$('save').checked,signals());
$('bot').onclick=()=>{
 let m=[],k=[];for(let i=0;i<40;i++){m.push([i*0.01,100+i*15,200+i*8]);}
 for(let i=0;i<10;i++){k.push([i*0.05,i*0.05+0.01]);}
 send(m,k,[0.45],[],'simulated bot',false,{webdriver:false,headless:false,honeypot:false},{username:'admin',password:'123456'});};
$('sel').onclick=()=>{
 let m=[],k=[],t=0,kt=0;
 for(let i=0;i<70;i++){const s=i/69,e=10*s*s*s-15*s*s*s*s+6*s*s*s*s*s;t+=0.005+Math.random()*0.01;
  m.push([t,100+e*600+Math.random()*3,200+Math.sin(e*3)*60+e*200+Math.random()*3]);}
 for(let i=0;i<10;i++){kt+=0.1+Math.random()*0.2;k.push([kt,kt+0.06+Math.random()*0.06]);}
 send(m,k,[0.5,2.5],[],'simulated Selenium bot',false,{webdriver:true,headless:false,honeypot:true},{username:'demo',password:'Demo@123'});};
</script></body></html>"""

REGISTER = """<!doctype html><html><head><meta charset="utf-8"><title>Create account</title>
<style>body{font-family:Arial,sans-serif;background:#f1f3f6;display:flex;justify-content:center;padding-top:60px}
.card{background:#fff;padding:30px;border-radius:10px;width:380px;box-shadow:0 2px 10px #0002}
input{width:100%;padding:10px;margin:8px 0;box-sizing:border-box}
button{width:100%;padding:11px;margin-top:8px;border:0;border-radius:6px;background:#2563eb;color:#fff;font-size:15px;cursor:pointer}
#msg{margin-top:12px;padding:10px;border-radius:6px;display:none}.ok{background:#dcfce7}.bad{background:#fee2e2}
small{color:#555}</style></head><body><div class="card">
<h2>Create account</h2>
<input id="u" type="text" placeholder="Username (3-20 letters, numbers, _)">
<input id="p" type="password" placeholder="Password (at least 6 characters)">
<input id="p2" type="password" placeholder="Confirm password">
<button id="go">Create account</button><div id="msg"></div>
<p><small>Accounts are saved in the database users.db. Passwords are hashed, never plain text.</small></p>
<p><a href="/">Back to login</a></p></div>
<script>
document.getElementById('go').onclick=async()=>{
 const m=document.getElementById('msg');
 const r=await (await fetch('/register',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({username:u.value,password:p.value,confirm:p2.value})})).json();
 m.style.display='block';m.className=r.ok?'ok':'bad';m.textContent=r.message;};
</script></body></html>"""

CHALLENGE = """<!doctype html><html><head><meta charset="utf-8"><title>Verification</title>
<style>body{font-family:Arial,sans-serif;background:#f1f3f6;display:flex;justify-content:center;padding-top:80px}
.card{background:#fff;padding:30px;border-radius:10px;width:400px;box-shadow:0 2px 10px #0002;text-align:center}
.q{font-size:28px;font-weight:bold;margin:18px 0}
input{padding:10px;font-size:18px;width:120px;text-align:center}
button{padding:11px 22px;border:0;border-radius:6px;background:#2563eb;color:#fff;font-size:15px;cursor:pointer;margin-left:6px}
#msg{margin-top:14px;color:#b91c1c;min-height:20px}small{color:#555}</style></head><body><div class="card">
<h2>Quick verification</h2>
<p>Your behaviour looked uncertain (risk score {{ '%.2f'|format(risk) }}, between 0.35 and 0.70),
so please prove you are a person.</p>
<div class="q">{{ q }}</div>
<input id="a" type="text" inputmode="numeric" autofocus><button id="ok">Verify</button>
<div id="msg"></div><small>Attempts left: <span id="left">3</span></small></div>
<script>
async function check(){
 const r=await (await fetch('/challenge/{{ rid }}',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({answer:document.getElementById('a').value})})).json();
 if(r.done){window.location='/dashboard/{{ rid }}';return;}
 document.getElementById('msg').textContent='Wrong answer, try again.';
 document.getElementById('left').textContent=r.left;document.getElementById('a').value='';}
document.getElementById('ok').onclick=check;
document.getElementById('a').addEventListener('keydown',e=>{if(e.key==='Enter')check();});
</script></body></html>"""

RESULT = """<!doctype html><html><head><meta charset="utf-8"><title>Detection Dashboard</title>
<style>
*{box-sizing:border-box}
body{font-family:Arial,sans-serif;background:#eef1f6;margin:0;color:#111}
.top{background:#111827;color:#fff;padding:14px 28px;display:flex;justify-content:space-between;align-items:center}
.top a{color:#93c5fd;text-decoration:none;margin-left:18px;font-size:14px}
.wrap{max-width:1150px;margin:auto;padding:22px}
.grid{display:grid;gap:16px;margin-bottom:16px}
.g4{grid-template-columns:repeat(auto-fit,minmax(210px,1fr))}
.g2{grid-template-columns:repeat(auto-fit,minmax(430px,1fr))}
.card{background:#fff;border-radius:10px;padding:18px;box-shadow:0 2px 8px #0001}
.card h3{margin:0 0 10px 0;font-size:16px}
.kpi small{color:#666;display:block;margin-bottom:6px}
.kpi .big{font-size:30px;font-weight:bold}
.allow{color:#16a34a}.challenge{color:#ca8a04}.block{color:#dc2626}
.verdict{border-left:8px solid}
.verdict.allow{border-color:#16a34a}.verdict.challenge{border-color:#ca8a04}.verdict.block{border-color:#dc2626}
canvas{width:100%;border:1px solid #e5e7eb;border-radius:6px;background:#fafafa}
.bar{height:22px;border-radius:11px;background:linear-gradient(90deg,#16a34a 35%,#ca8a04 35% 70%,#dc2626 70%);position:relative;margin:14px 0 6px}
.pin{position:absolute;top:-6px;width:4px;height:34px;background:#111}
small{color:#555}
table{border-collapse:collapse;width:100%;font-size:13px}
th,td{border-bottom:1px solid #e5e7eb;padding:7px 8px;text-align:left;vertical-align:top}
th{background:#f3f4f6}.h{color:#16a34a;font-weight:bold}.b{color:#dc2626;font-weight:bold}
.ev{display:flex;align-items:center;margin:6px 0;font-size:13px}
.ev .nm{width:200px}.ev .track{flex:1;height:14px;background:#f3f4f6;border-radius:7px;position:relative}
.ev .mid{position:absolute;left:50%;top:0;bottom:0;width:2px;background:#9ca3af}
.ev .fill{position:absolute;top:0;bottom:0;border-radius:7px}
.calc{background:#f8fafc;border-radius:8px;padding:10px;margin:8px 0;font-size:14px;line-height:1.6}
.btn{display:inline-block;background:#2563eb;color:#fff;padding:10px 18px;border-radius:6px;text-decoration:none}
.note{padding:12px 16px;border-radius:8px;margin-bottom:16px;font-size:14px}
.ok{background:#dcfce7}.warn{background:#fef9c3}.bad{background:#fee2e2}
</style></head><body>
<div class="top"><b>Bot Detection Dashboard</b>
 <div><a href="/">New login</a><a href="/history">History</a><a href="/data">Training data</a></div></div>
<div class="wrap">
{% if r.saved == 'saved' %}<div class="note ok">Saved as real human training data. Total real sessions: <b>{{ r.real_total }}</b>. Use the Training data page to retrain the model.</div>{% endif %}
{% if r.saved == 'too_little' %}<div class="note warn">Not saved: too little data. Move the mouse more and type at least 3 keys, then try again.</div>{% endif %}
{% if r.saved == 'rejected' %}<div class="note warn">Not saved as human data: the automation checks flagged this session.</div>{% endif %}
<div class="note {{ 'ok' if r.final.startswith('ACCESS GRANTED') else ('bad' if (r.final.startswith('ACCESS DENIED') or r.final.startswith('LOGIN FAILED')) else 'warn') }}">
 Final outcome: <b>{{ r.final }}</b> &nbsp; {{ r.final_msg }}</div>

<div class="grid g4">
 <div class="card kpi verdict {{ r.decision }}"><small>Final verdict</small>
  <div class="big {{ r.decision }}">{{ r.label }}</div><small>{{ r.message }}</small></div>
 <div class="card kpi"><small>Bot risk score</small><div class="big">{{ '%.1f'|format(r.risk*100) }}%</div>
  <small>Decision: {{ r.decision|upper }}</small></div>
 <div class="card kpi"><small>Human-like signals</small><div class="big h">{{ human_count }} / {{ rows|length }}</div>
  <small>measurements closer to a human</small></div>
 <div class="card kpi"><small>Bot-like signals</small><div class="big b">{{ bot_count }} / {{ rows|length }}</div>
  <small>measurements closer to a bot</small></div>
</div>

<div class="grid g2">
 <div class="card"><h3>How the decision is made</h3>
  <div class="bar"><div class="pin" style="left:{{ (r.risk*100)|round(1) }}%"></div></div>
  <small>below 0.35 = ALLOW (human) &nbsp;|&nbsp; 0.35 to 0.70 = CHALLENGE &nbsp;|&nbsp; above 0.70 = BLOCK (bot)</small>
  <p>The {{ model_name }} model combines all {{ rows|length }} measurements below: model risk <b>{{ '%.3f'|format(r.model_risk) }}</b>.
  The automation checks give rule risk <b>{{ '%.2f'|format(r.rule_risk) }}</b>. The final risk is the higher of the two:
  <b>{{ '%.3f'|format(r.risk) }}</b>. Source: <b>{{ r.source }}</b>, time {{ r.time }}.</p></div>
 <div class="card"><h3>Strongest evidence</h3>
  <small>Green (right) = looks human &nbsp;|&nbsp; Red (left) = looks bot. Length = how far your value is closer to that side, in standard deviations.</small>
  {% for e in top %}<div class="ev"><div class="nm">{{ e.label }}</div>
   <div class="track"><div class="mid"></div>
   <div class="fill" style="background:{{ '#16a34a' if e.ev>=0 else '#dc2626' }};
     {% if e.ev>=0 %}left:50%;width:{{ e.w }}%{% else %}right:50%;width:{{ e.w }}%{% endif %}"></div></div></div>{% endfor %}
 </div>
</div>

<div class="grid g2">
 <div class="card"><h3>Automation checks</h3>
  <small>Rules that run next to the model and can raise the risk on their own.</small>
  <table>{% for c in r.checks %}<tr><td>{{ c.name }}</td>
   <td class="{{ 'b' if c.hit else 'h' }}">{{ 'TRIGGERED' if c.hit else 'Clean' }}</td><td><small>{{ c.detail }}</small></td></tr>{% endfor %}</table></div>
 <div class="card"><h3>Credential check and verification</h3>
  <p><b>User:</b> {{ r.user or '(none entered)' }}<br><b>Credentials:</b> {{ r.cred_text }}</p>
  {% if r.challenge == 'passed' %}<p class="h">Passed. The user answered the question correctly, so access was granted.</p>
  {% elif r.challenge == 'failed' %}<p class="b">Failed. Three wrong answers, so access was denied.</p>
  {% elif r.decision == 'challenge' %}<p>This session needs verification. <a href="/challenge/{{ r.id }}">Open the challenge</a></p>
  {% else %}<p><small>Not needed. A challenge is only shown when the risk is between 0.35 and 0.70.</small></p>{% endif %}</div>
</div>

<div class="grid g2">
 <div class="card"><h3>Cursor path</h3>
  <canvas id="path" width="800" height="300"></canvas>
  <small>Blue = your mouse line, dashed grey = perfectly straight route, green dot = start, red dot = end.
  Humans curve and wobble, simple bots go straight.</small></div>
 <div class="card"><h3>Mouse speed over time</h3>
  <canvas id="speed" width="800" height="300"></canvas>
  <small>Humans speed up, slow down and pause. Bots keep one steady speed (flat line).</small></div>
</div>

<div class="grid g2">
 <div class="card"><h3>Typing rhythm (gap between key presses)</h3>
  <canvas id="keys" width="800" height="260"></canvas>
  <small>Each bar = time between two key presses. Uneven gaps = human. Equal gaps = bot.</small></div>
 <div class="card"><h3>Key hold time (how long each key is pressed)</h3>
  <canvas id="holds" width="800" height="260"></canvas>
  <small>Each bar = how long one key was held down. Humans vary, bots press for a fixed tiny time.</small></div>
</div>

<div class="grid g2">
 <div class="card"><h3>Calculations with your numbers</h3>
  <div class="calc"><b>Path straightness</b><br>straight distance / path length
   = {{ '%.1f'|format(v.straightness*v.path_length) }} px / {{ '%.1f'|format(v.path_length) }} px = <b>{{ '%.3f'|format(v.straightness) }}</b>
   <small>(human avg {{ '%.3f'|format(ha.straightness) }}, bot avg {{ '%.3f'|format(ba.straightness) }})</small></div>
  <div class="calc"><b>Speed consistency</b><br>std of speed / mean speed
   = {{ '%.0f'|format(v.std_speed) }} / {{ '%.0f'|format(v.mean_speed) }} = <b>{{ '%.2f'|format(v.speed_cv) }}</b>
   <small>(human avg {{ '%.2f'|format(ha.speed_cv) }}, bot avg {{ '%.2f'|format(ba.speed_cv) }})</small></div>
  <div class="calc"><b>Typing rhythm</b><br>keystrokes = {{ v.key_count|int }}, average gap = {{ '%.3f'|format(v.key_mean_interval) }} s,
   variation = {{ '%.3f'|format(v.key_std_interval) }} s <small>(bots: almost zero variation)</small></div>
  <div class="calc"><b>Key hold and flight time</b><br>hold = release - press = <b>{{ '%.3f'|format(v.key_hold_mean) }} s</b> on average
   (variation {{ '%.3f'|format(v.key_hold_std) }} s)<br>flight = next press - previous release = <b>{{ '%.3f'|format(v.key_flight_mean) }} s</b>
   <small>(human hold avg {{ '%.3f'|format(ha.key_hold_mean) }} s, bot avg {{ '%.3f'|format(ba.key_hold_mean) }} s)</small></div>
  <div class="calc"><b>Clicks and scrolling</b><br>{{ v.click_count|int }} clicks, average gap {{ '%.2f'|format(v.click_mean_interval) }} s;
   {{ v.scroll_count|int }} scroll events, {{ '%.0f'|format(v.scroll_distance) }} px scrolled</div>
  <div class="calc"><b>Pauses</b><br>{{ v.n_pauses|int }} pauses longer than 0.15 s = {{ '%.1f'|format(v.pause_ratio*100) }}% of the time</div>
 </div>
 <div class="card"><h3>Training data</h3>
  <p>Real human sessions saved so far: <b>{{ real_total }}</b></p>
  <p><small>On the login page, tick "I am a real person" before signing in to save a session.
  After saving some, open the Training data page and click <b>Retrain model</b>.
  The model then learns from real behaviour, not only simulated behaviour.</small></p>
  <a class="btn" href="/data">Open training data page</a></div>
</div>

<div class="card"><h3>All {{ rows|length }} calculations considered</h3>
 <small>Your value is compared with the average human and average bot from the training data.</small>
 <table><tr><th>Measurement</th><th>Your value</th><th>Human avg</th><th>Bot avg</th><th>Looks like</th><th>How it is calculated</th></tr>
 {% for g, items in groups %}
  <tr><td colspan="6" style="background:#eef2ff"><b>{{ g }}</b></td></tr>
  {% for it in items %}<tr><td>{{ it.label }} {{ it.unit }}</td><td>{{ it.val }}</td><td>{{ it.h }}</td><td>{{ it.b }}</td>
  <td class="{{ it.cls }}">{{ it.who }}</td><td><small>{{ it.how }}</small></td></tr>{% endfor %}
 {% endfor %}</table>
</div>
<br><a class="btn" href="/">Try again</a>
</div>

<script>
const D = {{ series|tojson }};
function setup(id){const c=document.getElementById(id);return [c,c.getContext('2d')];}
function axes(ctx,c,txt){ctx.fillStyle='#555';ctx.font='12px Arial';ctx.fillText(txt,8,14);}
function empty(ctx,c,msg){ctx.fillStyle='#888';ctx.font='15px Arial';ctx.fillText(msg,20,c.height/2);}
function bars(id,K,color,title,xl){const [c,ctx]=setup(id);
 if(K.length<1){empty(ctx,c,'No data recorded');return;}
 const m=Math.max(...K)||1,L=45,B=28,T=22,w=(c.width-L-15)/K.length;
 ctx.fillStyle=color;
 K.forEach((g,i)=>{const h=(g/m)*(c.height-B-T);ctx.fillRect(L+i*w+2,c.height-B-h,Math.max(w-4,2),h);});
 axes(ctx,c,title+'  max '+m.toFixed(2));ctx.fillText(xl,c.width-150,c.height-8);}
(function(){const [c,ctx]=setup('path'),P=D.path;
 if(P.length<2){empty(ctx,c,'No mouse movement recorded');return;}
 const xs=P.map(p=>p[0]),ys=P.map(p=>p[1]);
 const x0=Math.min(...xs),x1=Math.max(...xs),y0=Math.min(...ys),y1=Math.max(...ys);
 const pad=25,sx=(c.width-2*pad)/Math.max(x1-x0,1),sy=(c.height-2*pad)/Math.max(y1-y0,1),s=Math.min(sx,sy);
 const X=x=>pad+(x-x0)*s,Y=y=>pad+(y-y0)*s;
 ctx.setLineDash([6,5]);ctx.strokeStyle='#9ca3af';ctx.beginPath();
 ctx.moveTo(X(P[0][0]),Y(P[0][1]));ctx.lineTo(X(P[P.length-1][0]),Y(P[P.length-1][1]));ctx.stroke();
 ctx.setLineDash([]);ctx.strokeStyle='#2563eb';ctx.lineWidth=2;ctx.beginPath();
 P.forEach((p,i)=>i?ctx.lineTo(X(p[0]),Y(p[1])):ctx.moveTo(X(p[0]),Y(p[1])));ctx.stroke();
 ctx.fillStyle='#16a34a';ctx.beginPath();ctx.arc(X(P[0][0]),Y(P[0][1]),6,0,7);ctx.fill();
 ctx.fillStyle='#dc2626';ctx.beginPath();ctx.arc(X(P[P.length-1][0]),Y(P[P.length-1][1]),6,0,7);ctx.fill();
})();
(function(){const [c,ctx]=setup('speed'),S=D.speed;
 if(S.length<2){empty(ctx,c,'No speed data');return;}
 const t1=S[S.length-1][0]||1,vm=Math.max(...S.map(s=>s[1]))||1,L=45,B=28,T=22,R=15;
 const X=t=>L+(t/t1)*(c.width-L-R),Y=v=>c.height-B-(v/vm)*(c.height-B-T);
 ctx.strokeStyle='#ddd';ctx.beginPath();ctx.moveTo(L,T);ctx.lineTo(L,c.height-B);ctx.lineTo(c.width-R,c.height-B);ctx.stroke();
 ctx.strokeStyle='#7c3aed';ctx.lineWidth=2;ctx.beginPath();
 S.forEach((s,i)=>i?ctx.lineTo(X(s[0]),Y(s[1])):ctx.moveTo(X(s[0]),Y(s[1])));ctx.stroke();
 axes(ctx,c,'speed (px/s)  max '+Math.round(vm));ctx.fillText('time (s) -> '+t1.toFixed(2),c.width-150,c.height-8);
})();
bars('keys',D.key_gaps,'#0891b2','gap between keys (s)','key press number ->');
bars('holds',D.holds,'#f59e0b','key hold time (s)','key number ->');
</script></body></html>"""

DASH = """<!doctype html><html><head><meta charset="utf-8"><title>History</title>
<style>body{font-family:Arial;padding:30px}table{border-collapse:collapse}
td,th{border:1px solid #ccc;padding:6px 12px}</style></head><body>
<h2>History of attempts</h2><a href="/">Back</a><br><br>
<table><tr><th>Time</th><th>User</th><th>Source</th><th>Risk</th><th>Result</th><th>Final outcome</th><th>Saved as data</th><th>Details</th></tr>
{% for a in attempts %}<tr><td>{{a.time}}</td><td>{{a.user}}</td><td>{{a.source}}</td>
<td>{{'%.3f'|format(a.risk)}}</td><td>{{a.label}} ({{a.decision}})</td>
<td>{{a.final}}</td><td>{{ 'yes' if a.saved=='saved' else '-' }}</td>
<td><a href="/dashboard/{{a.id}}">view dashboard</a></td></tr>{% endfor %}
</table></body></html>"""

DATA = """<!doctype html><html><head><meta charset="utf-8"><title>Training data</title>
<style>body{font-family:Arial;background:#eef1f6;margin:0;color:#111}
.wrap{max-width:900px;margin:auto;padding:25px}
.card{background:#fff;border-radius:10px;padding:20px;margin-bottom:16px;box-shadow:0 2px 8px #0001}
table{border-collapse:collapse;width:100%;font-size:14px}th,td{border-bottom:1px solid #e5e7eb;padding:7px;text-align:left}
th{background:#f3f4f6}button,.btn{background:#2563eb;color:#fff;border:0;border-radius:6px;padding:10px 18px;font-size:15px;cursor:pointer;text-decoration:none;display:inline-block}
#msg{margin-top:12px;padding:12px;border-radius:6px;display:none;background:#dcfce7}
small{color:#555}</style></head><body><div class="wrap">
<a href="/">Back to login</a>
<h2>Training data</h2>
<div class="card"><h3>Real human sessions</h3>
 <p>Saved so far: <b>{{ n_real }}</b> &nbsp; <small>(files: real_raw.jsonl and real_sessions.csv)</small></p>
 <p><small>Collect more by ticking "I am a real person" on the login page before signing in.
 Ask friends to do it too. Aim for 20 or more.
 With 10 or more, 30% are kept aside to test how often real people are wrongly flagged as bots.</small></p>
 {% if n_real %}<a class="btn" style="background:#6b7280" href="/download/real_sessions.csv">Download CSV</a>{% endif %}
 <br><br><button id="rt">Retrain model now</button>
 <div id="msg"></div></div>
{% if clean_rows %}<div class="card"><h3>Data cleaning report (last training run)</h3>
 <table><tr><th>Source</th><th>Step</th><th>Count</th></tr>
 {% for c in clean_rows %}<tr><td>{{ c.Source }}</td><td>{{ c.Step }}</td><td>{{ c.Count }}</td></tr>{% endfor %}</table></div>{% endif %}
<div class="card"><h3>Current model results (simulated test set)</h3>
 <p>Active model: <b>{{ model_name }}</b></p>
 <table><tr>{% for c in cols %}<th>{{ c }}</th>{% endfor %}</tr>
 {% for row in rows %}<tr>{% for c in cols %}<td>{{ row[c] }}</td>{% endfor %}</tr>{% endfor %}</table></div>
</div>
<script>
document.getElementById('rt').onclick=async()=>{
 const b=document.getElementById('rt'),m=document.getElementById('msg');
 b.disabled=true;b.textContent='Training... please wait';
 try{const r=await (await fetch('/retrain',{method:'POST'})).json();
  m.style.display='block';m.innerHTML=r.message;}catch(e){m.style.display='block';m.textContent='Retraining failed: '+e;}
 b.disabled=false;b.textContent='Retrain model now';};
</script></body></html>"""


def fmt(x):
    return f"{x:.4g}" if abs(x) < 1000 else f"{x:.0f}"


@app.route("/")
def index():
    fr = store.get(request.args.get("failed", ""))
    return render_template_string(PAGE, n_real=real_count(),
                                  failed_msg=fr["cred_actual"] if fr else None,
                                  failed_id=fr["id"] if fr else None)


@app.route("/score", methods=["POST"])
def score():
    d = request.get_json()
    raw = {k: d.get(k, []) for k in ("moves", "keys", "clicks", "scrolls")}
    sig = d.get("signals", {}) or {}
    m, downs, ups, feats, model_risk, rule_risk, checks = analyse(raw, sig)
    risk = max(model_risk, rule_risk)
    decision, label, msg = decide(risk)
    saved = "off"
    if d.get("save"):
        if rule_risk > 0:
            saved = "rejected"
        elif len(raw["moves"]) >= 10 and len(raw["keys"]) >= 3:
            save_real(raw, feats)
            saved = "saved"
        else:
            saved = "too_little"
    user = (d.get("username") or "").strip().lower()
    cred_ok, cred_actual = check_login(user, d.get("password") or "")      # step 1: authentication
    if not cred_ok:
        finals = {decision: ("LOGIN FAILED", cred_actual)}
        cred_text = cred_actual
    elif decision == "allow":                                              # step 2: behaviour check
        finals = {"allow": ("ACCESS GRANTED", cred_actual)}
        cred_text = cred_actual
    elif decision == "block":
        finals = {"block": ("ACCESS DENIED", "Password was correct, but the behaviour looks like a bot.")}
        cred_text = cred_actual + " (but access was denied because of bot behaviour)"
    else:
        finals = {"challenge": ("CHALLENGE PENDING", "Password correct. Waiting for the verification question.")}
        cred_text = cred_actual
    rid = str(int(time.time() * 1000))
    rec = {"id": rid, "time": time.strftime("%H:%M:%S"),
           "source": d.get("source", "user"), "risk": risk, "model_risk": model_risk,
           "rule_risk": rule_risk, "checks": checks, "decision": decision,
           "label": label, "message": msg, "feats": feats,
           "series": series(m, downs, ups), "saved": saved, "real_total": real_count(),
           "final": finals[decision][0], "final_msg": finals[decision][1],
           "challenge": None, "tries": 0, "user": user,
           "cred_ok": cred_ok, "cred_actual": cred_actual, "cred_text": cred_text}
    if decision == "challenge" and cred_ok:
        x, y = random.randint(3, 9), random.randint(3, 9)
        rec["q"], rec["ans"] = f"What is {x} + {y}?", x + y
    store[rid] = rec
    attempts.insert(0, rec)
    log_attempt(user, cred_ok, risk, decision, rec["final"])
    return jsonify(id=rid, risk=risk, decision=decision, label=label, message=msg,
                   final=rec["final"])


@app.route("/challenge/<rid>", methods=["GET", "POST"])
def challenge(rid):
    r = store.get(rid)
    if not r or r["decision"] != "challenge" or not r["cred_ok"]:
        return redirect("/dashboard/" + rid if r else "/")
    if request.method == "GET":
        if r["challenge"]:
            return redirect("/dashboard/" + rid)
        return render_template_string(CHALLENGE, q=r["q"], risk=r["risk"], rid=rid)
    if r["challenge"]:
        return jsonify(done=True)
    try:
        ok = int(str(request.get_json().get("answer", "")).strip()) == r["ans"]
    except ValueError:
        ok = False
    r["tries"] += 1
    if ok:
        r["challenge"] = "passed"
        r["final"], r["final_msg"] = "ACCESS GRANTED", "Passed the challenge. " + r["cred_actual"]
        log_attempt(r["user"], True, r["risk"], "challenge", "ACCESS GRANTED (passed challenge)")
        return jsonify(done=True)
    if r["tries"] >= 3:
        r["challenge"], r["final"] = "failed", "ACCESS DENIED"
        r["final_msg"] = "Failed the verification challenge three times."
        log_attempt(r["user"], True, r["risk"], "challenge", "ACCESS DENIED (failed challenge)")
        return jsonify(done=True)
    return jsonify(done=False, left=3 - r["tries"])


USERS_PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>User database</title>
<style>body{font-family:Arial;background:#eef1f6;margin:0}.wrap{max-width:1000px;margin:auto;padding:25px}
.card{background:#fff;border-radius:10px;padding:20px;margin-bottom:16px;box-shadow:0 2px 8px #0001}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid #e5e7eb;padding:7px;text-align:left}
th{background:#f3f4f6}code{font-size:11px;color:#555}</style></head><body><div class="wrap">
<a href="/">Back to login</a>
<div class="card"><h2>User database (users.db)</h2>
<p><small>Passwords are stored as one-way hashes, so even the database owner cannot read them.</small></p>
<table><tr><th>ID</th><th>Username</th><th>Password (hash)</th><th>Created</th><th>Last login</th><th>Failed logins</th></tr>
{% for u in users %}<tr><td>{{u.id}}</td><td>{{u.username}}</td><td><code>{{u.password_hash[:34]}}...</code></td>
<td>{{u.created}}</td><td>{{u.last_login or '-'}}</td><td>{{u.failed_logins}}</td></tr>{% endfor %}</table></div>
<div class="card"><h2>Login log (last 30)</h2>
<table><tr><th>Time</th><th>Username</th><th>Password correct</th><th>Bot risk</th><th>Behaviour decision</th><th>Outcome</th></tr>
{% for l in logs %}<tr><td>{{l.time}}</td><td>{{l.username or '-'}}</td><td>{{ 'yes' if l.credentials_ok else 'no' }}</td>
<td>{{l.bot_risk}}</td><td>{{l.decision}}</td><td>{{l.outcome}}</td></tr>{% endfor %}</table></div>
</div></body></html>"""


@app.route("/users")
def users_page():
    with db() as c:
        users = c.execute("SELECT * FROM users ORDER BY id").fetchall()
        logs = c.execute("SELECT * FROM login_log ORDER BY id DESC LIMIT 30").fetchall()
    return render_template_string(USERS_PAGE, users=users, logs=logs)


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "GET":
        return render_template_string(REGISTER)
    d = request.get_json() or {}
    user, pw = (d.get("username") or "").strip().lower(), d.get("password") or ""
    if not re.fullmatch(r"[a-z0-9_]{3,20}", user):
        return jsonify(ok=False, message="Username must be 3-20 letters, numbers or underscores.")
    if len(pw) < 6:
        return jsonify(ok=False, message="Password must be at least 6 characters.")
    if pw != d.get("confirm"):
        return jsonify(ok=False, message="The two passwords do not match.")
    if not add_user(user, pw):
        return jsonify(ok=False, message="That username is already taken.")
    return jsonify(ok=True, message=f"Account '{user}' created. You can now sign in.")


@app.route("/result/<rid>")
def old_result(rid):
    return redirect("/dashboard/" + rid)


@app.route("/dashboard/<rid>")
def result(rid):
    r = store.get(rid)
    if not r:
        return redirect("/")
    v = dict(zip(FEATURES, r["feats"]))
    rows, hc, bc, evs, groups = [], 0, 0, [], []
    for g, names in GROUPS:
        items = []
        for n in names:
            label, unit, how = INFO[n]
            h, b = HUMAN_AVG.get(n), BOT_AVG.get(n)
            ev = evidence(n, v[n])
            evs.append({"label": label, "ev": ev})
            if h is None:
                who, cls = "-", ""
            elif ev >= 0:
                who, cls = "Human", "h"; hc += 1
            else:
                who, cls = "Bot", "b"; bc += 1
            it = {"label": label, "unit": f"({unit})" if unit else "", "val": fmt(v[n]),
                  "h": fmt(h) if h is not None else "-", "b": fmt(b) if b is not None else "-",
                  "who": who, "cls": cls, "how": how}
            items.append(it); rows.append(it)
        groups.append((g, items))
    evs = sorted(evs, key=lambda e: -abs(e["ev"]))[:8]
    mx = max([abs(e["ev"]) for e in evs] + [1e-9])
    for e in evs:
        e["w"] = round(min(abs(e["ev"]) / mx, 1) * 50, 1)
    ha = {k: HUMAN_AVG.get(k, 0) for k in FEATURES}
    ba = {k: BOT_AVG.get(k, 0) for k in FEATURES}
    return render_template_string(RESULT, r=r, v=v, ha=ha, ba=ba, groups=groups, rows=rows,
                                  human_count=hc, bot_count=bc, series=r["series"], top=evs,
                                  model_name=MODEL_NAME, real_total=real_count())


@app.route("/dashboard")
@app.route("/history")
def history():
    return render_template_string(DASH, attempts=attempts)


@app.route("/data")
def data():
    res = pd.read_csv("results.csv") if os.path.exists("results.csv") else pd.DataFrame()
    cl = pd.read_csv("cleaning_report.csv") if os.path.exists("cleaning_report.csv") else pd.DataFrame()
    return render_template_string(DATA, n_real=real_count(), model_name=MODEL_NAME,
                                  cols=list(res.columns), rows=res.to_dict("records"),
                                  clean_rows=cl.to_dict("records"))


@app.route("/download/real_sessions.csv")
def download():
    if not os.path.exists(CSV_FILE):
        abort(404)
    return send_file(CSV_FILE, as_attachment=True)


@app.route("/retrain", methods=["POST"])
def retrain():
    try:
        out = pipeline.run(verbose=False)
        load_all()
        best = out["results"].sort_values("ROC-AUC", ascending=False).iloc[0]
        msg = (f"Done. Best model: <b>{out['best']}</b> (accuracy {best['Accuracy']:.3f}, "
               f"ROC-AUC {best['ROC-AUC']:.4f}). Real human sessions used: <b>{out['real_total']}</b>.")
        if out["real_flagged"]:
            fl = out["real_flagged"][out["best"]]
            msg += (f"<br>Real-human test: {fl} of {out['real_test']} real sessions "
                    f"wrongly flagged as bots by {out['best']}.")
        tot = lambda k: sum(c for _, s, c in out["cleaning"] if s.startswith(k))
        msg += (f"<br>Cleaning: {tot('Duplicate')} duplicates removed, {tot('Missing')} missing values filled, "
                f"{tot('Empty')} empty sessions removed, {tot('Outlier')} outlier values capped.")
        msg += "<br>The gateway is already using the new model."
        return jsonify(ok=True, message=msg)
    except Exception as e:
        return jsonify(ok=False, message=f"Retraining failed: {e}"), 500


if __name__ == "__main__":
    app.run(debug=False)
