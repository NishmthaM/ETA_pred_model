# XGBoost next-station delay model for MULTI-JOURNEY data (journey_id + journey_date columns)
import numpy as np, pandas as pd, joblib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

FILE_PATH = "railway_eta_synthetic_500_journeys.xlsx"
SEED = 42
df = pd.read_excel(FILE_PATH)

def hhmm(v):
    if pd.isna(v): return np.nan
    p = str(v).split(":"); return int(p[0])*60 + int(p[1])

df["sched_min"] = df["Scheduled_arrival"].apply(hhmm)
df["preceding_delay_min"] = df["preceding_train_delay_min"].apply(hhmm)
df["journey_date"] = pd.to_datetime(df["journey_date"])
df["order"] = np.arange(len(df))
df = df.sort_values(["journey_date", "journey_id", "order"]).reset_index(drop=True)

g = df.groupby("journey_id", sort=False)          # one group = one journey
df["delay_lag1"] = g["current_delay_min"].shift(1)
df["delay_lag2"] = g["current_delay_min"].shift(2)
df["delay_change"] = df["current_delay_min"] - df["delay_lag1"]
df["delay_trend2"] = df["current_delay_min"] - df["delay_lag2"]
df["cong_lag1"] = g["congestion"].shift(1)
df["mean_delay_so_far"] = g["current_delay_min"].transform(lambda s: s.expanding().mean())
df["stations_done"] = g.cumcount()
df["sched_sin"] = np.sin(2*np.pi*df["sched_min"]/1440)
df["sched_cos"] = np.cos(2*np.pi*df["sched_min"]/1440)
df["month"] = df["journey_date"].dt.month
df["weekday"] = df["journey_date"].dt.weekday

features = ["current_delay_min","delay_lag1","delay_lag2","delay_change","delay_trend2",
            "mean_delay_so_far","stations_done","speed_kmph","distance_from_origin_km",
            "distance_to_next_km","scheduled_travel_min","dwell_time_min","preceding_delay_min",
            "congestion","cong_lag1","rain_mm","sched_sin","sched_cos","month","weekday"]

data = df.dropna(subset=["target_delay_min"]).reset_index(drop=True)

# split BY JOURNEY (chronological): last 20% of journeys are the unseen test set
ids = data["journey_id"].drop_duplicates().tolist()
cut = int(len(ids)*0.8)
train_ids, test_ids = set(ids[:cut]), set(ids[cut:])
tr = data[data.journey_id.isin(train_ids)]; te = data[data.journey_id.isin(test_ids)]
print(f"train journeys {len(train_ids)} ({len(tr)} rows) | test journeys {len(test_ids)} ({len(te)} rows)")

y_tr = tr["target_delay_min"] - tr["current_delay_min"]   # predict the CHANGE
# early-stopping validation = last 15% of TRAIN journeys (never the test set)
vcut = int(len(train_ids)*0.85); tr_ids = ids[:cut]
fit = tr[tr.journey_id.isin(tr_ids[:vcut])]; val = tr[tr.journey_id.isin(tr_ids[vcut:])]

model = XGBRegressor(n_estimators=2000, learning_rate=0.03, max_depth=4, min_child_weight=5,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                     early_stopping_rounds=50, random_state=SEED)
model.fit(fit[features], fit["target_delay_min"]-fit["current_delay_min"],
          eval_set=[(val[features], val["target_delay_min"]-val["current_delay_min"])], verbose=False)
print("best iteration:", model.best_iteration)

pred = te["current_delay_min"].values + model.predict(te[features])
pred = np.clip(pred, 0, None)                     # delay cannot be negative
act = te["target_delay_min"].values
base = te["current_delay_min"].values

def rep(n, a, p):
    print(f"{n:<22} MAE {mean_absolute_error(a,p):5.2f} min | RMSE {np.sqrt(mean_squared_error(a,p)):5.2f} | "
          f"R2 {r2_score(a,p):.3f} | within ±2 min {np.mean(np.abs(a-p)<=2)*100:5.1f}%")
print("\nUNSEEN TEST JOURNEYS")
rep("Persistence baseline", act, base)
rep("XGBoost", act, pred)

model.save_model("railway_eta_xgb.json"); joblib.dump(features, "railway_xgb_features.pkl")
pd.Series(model.feature_importances_, index=features).sort_values().plot.barh(figsize=(7,6), title="Feature importance")
plt.tight_layout(); plt.savefig("xgb_feature_importance.png", dpi=200); plt.close()

one = te[te.journey_id == te.journey_id.iloc[0]]
plt.figure(figsize=(10,4.5))
plt.plot(one["station_code"], one["target_delay_min"], "o-", label="Actual")
plt.plot(one["station_code"], np.clip(one["current_delay_min"]+model.predict(one[features]),0,None), "x--", label="XGBoost")
plt.xticks(rotation=60); plt.ylabel("Next-station delay (min)"); plt.legend(); plt.grid(True); plt.tight_layout()
plt.savefig("xgb_actual_vs_predicted.png", dpi=200)