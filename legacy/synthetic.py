import numpy as np, pandas as pd, datetime as dt
rng = np.random.default_rng(42)
tpl = pd.read_excel('railway_eta_synthetic_500_journeys.xlsx')
n = len(tpl)
sched = [pd.Timestamp(str(t)).hour*60+pd.Timestamp(str(t)).minute for t in tpl.Scheduled_arrival]
def T(m):
    m = int(m) % 1440
    return dt.time(m//60, m%60)
N_J = 500
dates = pd.date_range('2025-01-01', periods=N_J, freq='D')
rows = []
for j, d in enumerate(dates):
    month = d.month
    monsoon = 1.0 if month in (6,7,8,9) else 0.25 if month in (5,10) else 0.03
    rain = rng.exponential(12*monsoon) if rng.random() < (0.85 if monsoon==1 else 0.3*monsoon+0.05) else 0.0
    rain = round(min(rain, 120), 1)
    weekend = d.weekday() >= 5
    net = rng.gamma(2.0, 1.8)                     # network-wide disruption of the day
    delay = 0.0
    prev_actual_travel = None
    for i in range(n):
        # preceding train (running ahead on same line): correlated with same-day disruption & own delay
        prec = max(0.0, 0.6*delay + 0.5*net + rng.normal(0, 2.0))
        prec = min(int(round(prec)), 120)
        cong_score = 0.06*prec + 0.05*rain**0.7 + (0.5 if 16*60 <= sched[i] <= 19*60 else 0) - (0.3 if weekend else 0) + rng.normal(0, 0.45)
        cong = int(np.clip(round(cong_score), 0, 3))
        if i == 0: cong = 0
        dwell = 0 if i == 0 else int(np.clip(round(1 + (delay > 15)*rng.random() + rng.normal(0, .25)), 0, 3))
        cur = int(round(delay))
        actual = sched[i] + cur
        depart = actual + dwell
        dist_next = int(tpl.distance_to_next_km[i]); trav = int(tpl.scheduled_travel_min[i])
        # speed: km covered on previous section / actual minutes taken
        if i == 0: speed = 0.0
        else:
            dprev = tpl.distance_from_origin_km[i]-tpl.distance_from_origin_km[i-1]
            speed = float(np.clip(dprev / max(prev_actual_travel,1) * 60 * rng.normal(1, .05), 5, 120))
        rows.append(dict(
            journey_id=f"J{j+1:04d}", journey_date=d.date(), train_id=10108,
            station_full_name=tpl.station_full_name[i], station_code=tpl.station_code[i],
            station_zone=tpl.station_zone[i], Station_address=tpl.Station_address[i],
            speed_kmph=round(speed,2), Scheduled_arrival=T(sched[i]), Actual_Arrival_time=T(actual),
            Scheduled_departure=T(sched[i]+ (0 if i in (0,n-1) else 1)), **{"Dep. Time":T(depart)},
            train_type='MEMU', current_delay_min=cur,
            distance_from_origin_km=int(tpl.distance_from_origin_km[i]), distance_to_next_km=dist_next,
            scheduled_travel_min=trav, dwell_time_min=dwell,
            preceding_train_delay_min=T(prec), congestion=cong, rain_mm=rain, target_delay_min=np.nan))
        # evolve delay to next station
        if i < n-1:
            slack = 0.10*trav                      # timetable padding
            inc = (1.7*cong + 0.5 + 0.035*rain + 0.15*net*(cong>0) - slack*0.35 + (dwell-1)*1.2
                   + 0.05*(prec-delay) + rng.normal(0, 1.6))
            if delay > 20: inc -= 0.06*(delay-20)     # recovery / catch-up limits
            nd = max(0.0, delay + inc)
            prev_actual_travel = trav + (nd-delay) - 0  # minutes taken on this section
            prev_actual_travel = max(prev_actual_travel, 3)
            delay = nd
    # target = next station's delay
    idx = range(len(rows)-n, len(rows))
    dl = [rows[k]['current_delay_min'] for k in idx]
    for a,k in enumerate(idx):
        rows[k]['target_delay_min'] = float(dl[a+1]) if a < n-1 else np.nan
df = pd.DataFrame(rows)
cols = list(tpl.columns)
cols = ['journey_id','journey_date'] + cols[:-1] + ['rain_mm','target_delay_min']
df = df[cols]
df.to_excel('railway_eta_synthetic_500_journeys.xlsx', index=False)
print(df.shape); print(df.current_delay_min.describe()); print(df.groupby('station_code',sort=False).current_delay_min.mean().round(1).to_dict())
print(df.congestion.value_counts().sort_index().to_dict())