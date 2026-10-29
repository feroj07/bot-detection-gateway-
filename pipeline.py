"""Bot detection pipeline: simulated + real data -> 27 features -> models -> evaluation.
Run:  python pipeline.py
If real_raw.jsonl exists (saved from the login page), real human sessions are included.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             roc_curve, ConfusionMatrixDisplay)

rng = np.random.default_rng(42)
RAW_FILE = "real_raw.jsonl"

FEATURES = ["duration", "n_points", "path_length", "straightness",
            "mean_speed", "std_speed", "max_speed", "speed_cv",
            "mean_acc", "std_acc", "max_acc",
            "mean_curvature", "std_curvature", "mean_angle_change",
            "n_pauses", "pause_ratio",
            "key_count", "key_mean_interval", "key_std_interval",
            # new in this version
            "key_hold_mean", "key_hold_std", "key_flight_mean", "key_flight_std",
            "click_count", "click_mean_interval", "scroll_count", "scroll_distance"]


# ---------------------------------------------------------------- features
def extract_features(t, x, y, key_times, key_up=None, clicks=None, scrolls=None):
    """t in seconds, x/y in pixels. key_times = key-down times, key_up = key-release
    times (same order), clicks = click times, scrolls = list of wheel deltas."""
    t, x, y = map(lambda a: np.asarray(a, dtype=float), (t, x, y))
    f = dict.fromkeys(FEATURES, 0.0)
    if len(t) >= 3:
        dt = np.diff(t)
        dt[dt <= 1e-4] = 1e-4
        dx, dy = np.diff(x), np.diff(y)
        d = np.hypot(dx, dy)
        v = d / dt
        a = np.diff(v) / dt[1:]
        ang = np.arctan2(dy, dx)
        dang = np.abs(np.diff(np.unwrap(ang)))
        curv = dang / (d[1:] + 1e-6)
        total = d.sum()
        straight_dist = np.hypot(x[-1] - x[0], y[-1] - y[0])
        pauses = dt > 0.15
        f.update(
            duration=t[-1] - t[0], n_points=len(t), path_length=total,
            straightness=straight_dist / (total + 1e-6),
            mean_speed=v.mean(), std_speed=v.std(), max_speed=v.max(),
            speed_cv=v.std() / (v.mean() + 1e-6),
            mean_acc=np.abs(a).mean(), std_acc=a.std(), max_acc=np.abs(a).max(),
            mean_curvature=curv.mean(), std_curvature=curv.std(),
            mean_angle_change=dang.mean(),
            n_pauses=int(pauses.sum()), pause_ratio=dt[pauses].sum() / dt.sum(),
        )
    kt = np.asarray(key_times, dtype=float)
    order = np.argsort(kt)
    kt = kt[order]
    ku = None
    if key_up is not None and len(key_up) == len(kt) and len(kt):
        ku = np.asarray(key_up, dtype=float)[order]
    f["key_count"] = len(kt)
    if len(kt) >= 2:
        ki = np.diff(kt)
        f.update(key_mean_interval=ki.mean(), key_std_interval=ki.std())
    if ku is not None:
        hold = np.clip(ku - kt, 0, None)
        f.update(key_hold_mean=hold.mean(), key_hold_std=hold.std())
        if len(kt) >= 2:
            fl = kt[1:] - ku[:-1]                      # release -> next press
            f.update(key_flight_mean=fl.mean(), key_flight_std=fl.std())
    ck = np.sort(np.asarray(clicks if clicks is not None else [], dtype=float))
    f["click_count"] = len(ck)
    if len(ck) >= 2:
        f["click_mean_interval"] = np.diff(ck).mean()
    sc = np.asarray(scrolls if scrolls is not None else [], dtype=float)
    f["scroll_count"] = len(sc)
    f["scroll_distance"] = np.abs(sc).sum() if len(sc) else 0.0
    return [f[k] for k in FEATURES]


def compress(ts, cap, new):
    """Shorten long idle gaps so waiting between actions does not distort the maths."""
    ts = np.asarray(ts, dtype=float)
    if len(ts) < 2:
        return ts
    g = np.diff(ts)
    g = np.where(g > cap, new, g)
    return np.concatenate([[ts[0]], ts[0] + np.cumsum(g)])


def prepare_session(raw):
    """Raw browser events -> (mouse array, key-down times, key-up times, features).
    Used by the web app AND by retraining, so both treat data identically."""
    m = np.array(raw.get("moves", []), dtype=float).reshape(-1, 3)
    if len(m):
        m = m[np.argsort(m[:, 0])]
        m[:, 0] = compress(m[:, 0], 0.8, 0.3)
    downs, ups, has_up = [], [], True
    for k in raw.get("keys", []):
        if isinstance(k, (list, tuple)):
            d, u = k[0], (k[1] if len(k) > 1 else None)
        else:
            d, u = k, None
        downs.append(float(d))
        if u is None:
            has_up = False
            ups.append(float(d))
        else:
            ups.append(float(u))
    order = np.argsort(downs) if downs else []
    downs = np.array(downs, dtype=float)[order] if len(downs) else np.array([])
    ups = np.array(ups, dtype=float)[order] if len(ups) else np.array([])
    cd = compress(downs, 2.0, 0.6) if len(downs) else downs
    cu = ups + (cd - downs) if (len(downs) and has_up) else None
    clicks = compress(sorted(float(c) for c in raw.get("clicks", [])), 3.0, 1.0)
    scrolls = [s[1] if isinstance(s, (list, tuple)) else s for s in raw.get("scrolls", [])]
    feats = extract_features(m[:, 0], m[:, 1], m[:, 2], cd, cu, clicks, scrolls)
    return m, cd, cu, feats


# ---------------------------------------------------------- simulated data
def bezier_path(n, start, end, bend):
    s = np.linspace(0, 1, n)
    mid = (np.array(start) + np.array(end)) / 2
    ctrl = mid + np.array([-(end[1] - start[1]), end[0] - start[0]]) * bend
    p = ((1 - s)[:, None] ** 2) * start + (2 * (1 - s) * s)[:, None] * ctrl \
        + (s[:, None] ** 2) * end
    return p[:, 0], p[:, 1]


def human_holds(nk):
    return np.clip(rng.normal(rng.uniform(0.07, 0.14), rng.uniform(0.015, 0.045), nk),
                   0.03, 0.3)


def human_session():
    start = rng.uniform(50, 600, 2); end = rng.uniform(300, 1100, 2)
    hurried = rng.random() < 0.25                       # fast, sparse, straighter
    n = rng.integers(12, 40) if hurried else rng.integers(35, 130)
    s = np.linspace(0, 1, n)
    prof = 10 * s**3 - 15 * s**4 + 6 * s**5            # bell-shaped speed
    px, py = bezier_path(n, start, end,
                         rng.uniform(-0.05, 0.05) if hurried else rng.uniform(-0.25, 0.25))
    idx = prof * (n - 1)
    x = np.interp(idx, np.arange(n), px) + rng.normal(0, rng.uniform(0.5, 2.5), n)
    y = np.interp(idx, np.arange(n), py) + rng.normal(0, rng.uniform(0.5, 2.5), n)
    dt = rng.gamma(4, rng.uniform(0.003, 0.006), n)
    dt[rng.random(n) < 0.04] += rng.uniform(0.15, 0.6)   # hesitation
    t = np.cumsum(dt)
    nk = 0 if rng.random() < 0.08 else rng.integers(6, 22)
    downs = np.cumsum(rng.lognormal(np.log(rng.uniform(0.09, 0.3)), 0.45, nk))
    ups = downs + human_holds(nk)
    clicks = np.sort(rng.uniform(0, t[-1] + (downs[-1] if nk else 1), rng.integers(1, 4))) \
        if rng.random() > 0.05 else []
    scrolls = rng.choice([-100, 100, -120, 120], rng.integers(1, 6)) \
        if rng.random() < 0.2 else []
    return t, x, y, downs, ups, clicks, scrolls


def bot_session(kind):
    start = rng.uniform(50, 600, 2); end = rng.uniform(300, 1100, 2)
    n = rng.integers(10, 110)
    scrolls = []
    if kind == "linear":                    # straight, constant speed, robotic keys
        s = np.linspace(0, 1, n)
        x = start[0] + (end[0] - start[0]) * s + rng.normal(0, 0.2, n)
        y = start[1] + (end[1] - start[1]) * s + rng.normal(0, 0.2, n)
        t = np.cumsum(np.full(n, rng.uniform(0.004, 0.02)))
        nk = rng.integers(6, 22)
        downs = np.cumsum(rng.normal(rng.uniform(0.03, 0.09), 0.004, nk).clip(0.005))
        ups = downs + rng.normal(rng.uniform(0.005, 0.02), 0.001, nk).clip(0.001)
        clicks = [t[-1] + rng.uniform(0, 0.05)]
    elif kind == "smart":                   # curved + fake noise, human-like timing
        n = rng.integers(25, 110)
        sl = np.linspace(0, 1, n)
        w = rng.uniform(0.3, 0.9)                       # how human-like the speed is
        prof = w * (10 * sl**3 - 15 * sl**4 + 6 * sl**5) + (1 - w) * sl
        px, py = bezier_path(n, start, end, rng.uniform(-0.2, 0.2))
        idx = prof * (n - 1)
        x = np.interp(idx, np.arange(n), px) + rng.normal(0, rng.uniform(1, 3), n)
        y = np.interp(idx, np.arange(n), py) + rng.normal(0, rng.uniform(1, 3), n)
        t = np.cumsum(rng.gamma(5, rng.uniform(0.002, 0.004), n))
        nk = rng.integers(6, 22)
        downs = np.cumsum(rng.lognormal(np.log(rng.uniform(0.08, 0.28)), 0.35, nk))
        if rng.random() < 0.5:                          # half copy human key holds
            ups = downs + human_holds(nk)
        else:
            ups = downs + rng.uniform(0.03, 0.06, nk)
        clicks = np.sort(rng.uniform(0, t[-1] + downs[-1], rng.integers(1, 3)))
        if rng.random() < 0.1:
            scrolls = rng.choice([-100, 100], rng.integers(1, 3))
    else:                                   # "jumpy": few teleport-like points
        n = rng.integers(3, 9)
        s = np.sort(rng.random(n)); s[0], s[-1] = 0, 1
        x = start[0] + (end[0] - start[0]) * s
        y = start[1] + (end[1] - start[1]) * s
        t = np.cumsum(rng.uniform(0.01, 0.12, n))
        downs, ups = np.array([]), np.array([])         # no typing at all
        clicks = [t[-1]] if rng.random() < 0.5 else []
    return t, x, y, downs, ups, clicks, scrolls


def make_dataset(n_human=1500, n_bot=1500):
    rows = []
    for _ in range(n_human):
        rows.append(extract_features(*human_session()) + [0, "human", "sim"])
    kinds = ["linear", "smart", "jumpy"]
    for i in range(n_bot):
        k = kinds[i % 3]
        rows.append(extract_features(*bot_session(k)) + [1, k, "sim"])
    return pd.DataFrame(rows, columns=FEATURES + ["label", "type", "source"])


# --------------------------------------------------------------- real data
def load_real(path=RAW_FILE):
    rows = []
    if os.path.exists(path):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                rows.append(prepare_session(raw)[3] + [0, "real_human", "real"])
    return pd.DataFrame(rows, columns=FEATURES + ["label", "type", "source"])


def real_count(path=RAW_FILE):
    if not os.path.exists(path):
        return 0
    with open(path) as fh:
        return sum(1 for line in fh if line.strip())


# ---------------------------------------------------------- data cleaning
def inject_log_faults(df):
    """Mimic real logging problems so the cleaning stage has something to fix:
    missing values, corrupted timestamps (absurd speeds) and duplicated rows."""
    d = df.copy()
    n = len(d)
    for r in rng.choice(n, int(0.007 * n), replace=False):            # missing cells
        cols = list(rng.choice(FEATURES, rng.integers(1, 4), replace=False))
        d.loc[d.index[r], cols] = np.nan
    bad = rng.choice(n, int(0.005 * n), replace=False)                # corrupted timestamps
    for c in ["mean_speed", "std_speed", "max_speed", "mean_acc", "std_acc", "max_acc"]:
        d.loc[d.index[bad], c] = d.loc[d.index[bad], c] * 60
    dup = d.sample(int(0.01 * n), random_state=1)                     # logger retries
    return pd.concat([d, dup], ignore_index=True)


def clean_data(df, name, log=print):
    """Duplicates -> infinite/missing values -> empty sessions. Returns (clean df, report)."""
    rep = [(name, "Rows before cleaning", len(df))]
    d = df.copy()
    d[FEATURES] = d[FEATURES].replace([np.inf, -np.inf], np.nan)
    n_dup = int(d.duplicated(subset=FEATURES + ["label"]).sum())
    d = d.drop_duplicates(subset=FEATURES + ["label"])
    rep.append((name, "Duplicate sessions removed", n_dup))
    n_missing = int(d[FEATURES].isna().sum().sum())
    d[FEATURES] = d[FEATURES].fillna(d[FEATURES].median())
    rep.append((name, "Missing/infinite values filled (median)", n_missing))
    empty = (d["n_points"] == 0) & (d["key_count"] == 0)
    n_empty = int(empty.sum())
    d = d[~empty]
    rep.append((name, "Empty sessions removed", n_empty))
    rep.append((name, "Rows after cleaning", len(d)))
    for _, step, cnt in rep:
        log(f"  [{name}] {step}: {cnt}")
    return d.reset_index(drop=True), rep


# ------------------------------------------------------------------- main
def run(verbose=True):
    global rng
    rng = np.random.default_rng(42)
    log = print if verbose else (lambda *a, **k: None)

    log("Generating simulated sessions...")
    sim = inject_log_faults(make_dataset())
    real = load_real()
    log(f"Raw sessions: {len(sim)} simulated | {len(real)} real human sessions from the login page")
    log("\n=== Data cleaning ===")
    sim, rep1 = clean_data(sim, "simulated", log)
    real, rep2 = clean_data(real, "real", log) if len(real) else (real, [])
    report = rep1 + rep2
    pd.concat([sim, real], ignore_index=True).to_csv("dataset.csv", index=False)

    sim_tr, sim_te = train_test_split(sim, test_size=0.25, stratify=sim["label"],
                                      random_state=42)
    if len(real) >= 10:
        real_tr, real_te = train_test_split(real, test_size=0.3, random_state=42)
    else:
        real_tr, real_te = real, real.iloc[0:0]
        if len(real):
            log("Fewer than 10 real sessions: all used for training, no separate real test.")
    # give real sessions more weight by repeating them (about 5% of the training set)
    rep = int(max(1, min(20, 0.05 * len(sim_tr) / max(len(real_tr), 1)))) if len(real_tr) else 0
    train = pd.concat([sim_tr] + [real_tr] * rep, ignore_index=True)
    # outlier handling: cap extreme values at the 1st / 99th percentile of the training data
    low, high = train[FEATURES].quantile(0.01), train[FEATURES].quantile(0.99)
    n_clip = int(((train[FEATURES] < low) | (train[FEATURES] > high)).sum().sum())
    log(f"  [all] Outlier values capped at 1st-99th percentile: {n_clip}")
    report.append(("all", "Outlier values capped (1st-99th percentile)", n_clip))
    train, sim_te, real_te = train.copy(), sim_te.copy(), real_te.copy()
    for d in (train, sim_te, real_te):
        d[FEATURES] = d[FEATURES].clip(lower=low, upper=high, axis=1)
    pd.DataFrame(report, columns=["Source", "Step", "Count"]).to_csv("cleaning_report.csv", index=False)
    labels = ["Duplicates removed", "Missing values filled", "Empty sessions removed", "Outliers capped"]
    vals = [sum(c for s_, st, c in report if st.startswith(k)) for k in
            ("Duplicate", "Missing", "Empty", "Outlier")]
    plt.figure(figsize=(6, 4)); plt.bar(labels, vals, color="#2563eb")
    plt.xticks(rotation=20, ha="right"); plt.ylabel("Count"); plt.title("Data cleaning summary")
    for i, v_ in enumerate(vals):
        plt.text(i, v_, str(v_), ha="center", va="bottom")
    plt.tight_layout(); plt.savefig("cleaning_summary.png", dpi=150); plt.close()
    Xtr, ytr = train[FEATURES], train["label"]
    Xte, yte, ttest = sim_te[FEATURES], sim_te["label"], sim_te["type"]

    models = {
        "Logistic Regression": make_pipeline(StandardScaler(),
                                             LogisticRegression(max_iter=3000)),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }

    results, probs = [], {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        p = m.predict_proba(Xte)[:, 1]
        pred = (p >= 0.5).astype(int)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        probs[name] = p
        results.append({
            "Model": name,
            "Accuracy": accuracy_score(yte, pred),
            "Precision": precision_score(yte, pred),
            "Recall": recall_score(yte, pred),
            "F1": f1_score(yte, pred),
            "ROC-AUC": roc_auc_score(yte, p),
            "False Positive Rate": fp / (fp + tn),
        })
    res = pd.DataFrame(results).round(4)
    res.to_csv("results.csv", index=False)
    log(f"\n=== Results (simulated test set, {len(Xte)} sessions, {len(FEATURES)} features) ===")
    log(res.to_string(index=False))

    best_name = res.sort_values("ROC-AUC", ascending=False).iloc[0]["Model"]
    best = models[best_name]
    log("\nBest model:", best_name)

    pred_best = (probs[best_name] >= 0.5).astype(int)
    log("\n=== Detection rate per bot type (best model) ===")
    for k in ["linear", "smart", "jumpy"]:
        mask = (ttest == k).values
        log(f"{k:>7}: {pred_best[mask].mean():.3f}")

    real_report = None
    if len(real_te):
        log("\n=== Real human test sessions (never used for training) ===")
        real_report = {}
        for name, m in models.items():
            flagged = int((m.predict_proba(real_te[FEATURES])[:, 1] >= 0.5).sum())
            real_report[name] = flagged
            log(f"{name:>20}: {flagged} of {len(real_te)} real humans wrongly flagged as bot")

    joblib.dump({"model": best, "features": FEATURES, "name": best_name,
                 "clip": {"low": low.to_dict(), "high": high.to_dict()}}, "model.joblib")

    plt.figure(figsize=(6, 5))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(yte, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(yte, p):.3f})")
    plt.plot([0, 1], [0, 1], "k--"); plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate"); plt.title("ROC Curves"); plt.legend()
    plt.tight_layout(); plt.savefig("roc_curves.png", dpi=150); plt.close()

    ConfusionMatrixDisplay.from_predictions(
        yte, pred_best, display_labels=["Human", "Bot"], cmap="Blues")
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout(); plt.savefig("confusion_matrix.png", dpi=150); plt.close()

    rf = models["Random Forest"]
    imp = pd.Series(rf.feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(7, 8)); imp.plot.barh()
    plt.title("Feature Importance (Random Forest)")
    plt.tight_layout(); plt.savefig("feature_importance.png", dpi=150); plt.close()
    log("\nSaved: model.joblib, results.csv, dataset.csv, roc_curves.png, "
        "confusion_matrix.png, feature_importance.png, cleaning_report.csv, cleaning_summary.png")
    return {"cleaning": report, "best": best_name, "results": res, "real_total": len(real),
            "real_test": len(real_te), "real_flagged": real_report}


if __name__ == "__main__":
    run()
