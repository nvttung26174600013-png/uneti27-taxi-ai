import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import KFold

os.chdir(os.path.dirname(os.path.abspath(__file__)))

class SmoothedTargetEncoder:
    def __init__(self, group_Specs):
        self.group_specs = group_Specs
        self.maps = {}
        self.global_mean = 6.48

    def fit_transform_oof(self, df, y_log, n_splits=5):
        self.global_mean = float(np.mean(y_log))
        out = pd.DataFrame(index=df.index)
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        temp = df.copy()
        temp['_target_'] = y_log
        for cols, feat_name, weight in self.group_specs:
            out[feat_name] = np.nan
            for tr_idx, val_idx in kf.split(df):
                tr_slice = temp.iloc[tr_idx]
                stats = tr_slice.groupby(cols)['_target_'].agg(['count', 'mean'])
                smoothed = (stats['count'] * stats['mean'] + weight * self.global_mean) / (stats['count'] + weight)
                val_keys = temp.iloc[val_idx].set_index(cols).index
                out.iloc[val_idx, out.columns.get_loc(feat_name)] = val_keys.map(smoothed).fillna(self.global_mean).values
            full_stats = temp.groupby(cols)['_target_'].agg(['count', 'mean'])
            self.maps[feat_name] = (cols, (full_stats['count'] * full_stats['mean'] + weight * self.global_mean) / (full_stats['count'] + weight))
            out[feat_name] = out[feat_name].astype(np.float32)
        return out

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        for feat_name, (cols, mapping_series) in self.maps.items():
            keys = df.set_index(cols).index
            out[feat_name] = keys.map(mapping_series).fillna(self.global_mean).values.astype(np.float32)
        return out

setattr(sys.modules['__main__'], 'SmoothedTargetEncoder', SmoothedTargetEncoder)

def build_features(df):
    d = pd.DataFrame(index=df.index)
    d['vendor_id'] = df['vendor_id'].astype(np.float32)
    d['pickup_month'] = df['pickup_month'].astype(np.float32)
    d['pickup_dayofweek'] = df['pickup_dayofweek'].astype(np.float32)
    d['pickup_hour'] = df['pickup_hour'].astype(np.float32)
    d['pickup_minute_bucket'] = df['pickup_minute_bucket'].astype(np.float32)
    d['passenger_count'] = df['passenger_count'].astype(np.float32)
    d['pickup_lon_approx'] = df['pickup_lon_approx'].astype(np.float32)
    d['pickup_lat_approx'] = df['pickup_lat_approx'].astype(np.float32)
    d['dropoff_lon_approx'] = df['dropoff_lon_approx'].astype(np.float32)
    d['dropoff_lat_approx'] = df['dropoff_lat_approx'].astype(np.float32)
    d['straight_line_km_approx'] = df['straight_line_km_approx'].astype(np.float32)
    d['store_and_fwd_flag_num'] = (df['store_and_fwd_flag'] == 'Y').astype(np.float32)
    
    dlon = d['dropoff_lon_approx'] - d['pickup_lon_approx']
    dlat = d['dropoff_lat_approx'] - d['pickup_lat_approx']
    dlon_km = dlon * 84.30
    dlat_km = dlat * 111.32
    
    d['dlon_km'] = dlon_km
    d['dlat_km'] = dlat_km
    d['haversine_grid_km'] = np.sqrt(dlon_km**2 + dlat_km**2)
    d['manhattan_grid_km'] = np.abs(dlon_km) + np.abs(dlat_km)
    
    theta = np.radians(-29.0)
    rot_x = dlon_km * np.cos(theta) - dlat_km * np.sin(theta)
    rot_y = dlon_km * np.sin(theta) + dlat_km * np.cos(theta)
    d['rot_manhattan_km'] = np.abs(rot_x) + np.abs(rot_y)
    
    d['bearing'] = np.arctan2(dlon_km, dlat_km)
    d['center_lon'] = (d['pickup_lon_approx'] + d['dropoff_lon_approx']) * 0.5
    d['center_lat'] = (d['pickup_lat_approx'] + d['dropoff_lat_approx']) * 0.5
    d['same_grid_flag'] = ((dlon == 0) & (dlat == 0)).astype(np.float32)
    d['log_straight_km'] = np.log1p(d['straight_line_km_approx'])
    d['log_rot_manhattan_km'] = np.log1p(d['rot_manhattan_km'])
    
    landmarks = {
        'jfk': (-73.78, 40.64),
        'lga': (-73.87, 40.77),
        'ewr': (-74.18, 40.69),
        'midtown': (-73.98, 40.75)
    }
    for name, (lon_lm, lat_lm) in landmarks.items():
        d[f'pickup_dist_{name}'] = np.sqrt((d['pickup_lon_approx'] - lon_lm)**2 + (d['pickup_lat_approx'] - lat_lm)**2)
        d[f'dropoff_dist_{name}'] = np.sqrt((d['dropoff_lon_approx'] - lon_lm)**2 + (d['dropoff_lat_approx'] - lat_lm)**2)
        
    exact_hour = d['pickup_hour'] + d['pickup_minute_bucket'] / 60.0
    d['exact_hour'] = exact_hour
    d['hour_sin'] = np.sin(2.0 * np.pi * exact_hour / 24.0)
    d['hour_cos'] = np.cos(2.0 * np.pi * exact_hour / 24.0)
    d['dow_sin'] = np.sin(2.0 * np.pi * d['pickup_dayofweek'] / 7.0)
    d['dow_cos'] = np.cos(2.0 * np.pi * d['pickup_dayofweek'] / 7.0)
    d['is_weekend'] = (d['pickup_dayofweek'] >= 5).astype(np.float32)
    d['is_rush_hour'] = (((d['pickup_hour'] >= 7) & (d['pickup_hour'] <= 10)) | 
                         ((d['pickup_hour'] >= 16) & (d['pickup_hour'] <= 19))).astype(np.float32) * (1.0 - d['is_weekend'])
    return d

st.set_page_config(page_title="UNETI27 - Dự đoán thời gian Taxi NYC", page_icon="🚕", layout="wide")

@st.cache_resource
def load_artifact():
    return joblib.load('uneti27_taxi_model.pkl')

artifact = load_artifact()
model = artifact['model']
encoder = artifact['encoder']
feature_cols = artifact['feature_cols']

def calc_approx_km(lon1, lat1, lon2, lat2):
    dlon_km = (lon2 - lon1) * 84.30
    dlat_km = (lat2 - lat1) * 111.32
    raw_km = np.sqrt(dlon_km**2 + dlat_km**2)
    return float(np.round(raw_km * 2.0) / 2.0)

def prepare_single_input(df_in):
    base_feat = build_features(df_in)
    te_feat = encoder.transform(df_in)
    full_feat = pd.concat([base_feat, te_feat], axis=1)
    return full_feat[feature_cols]

st.title("🚕 ỨNG DỤNG AI DỰ ĐOÁN THỜI GIAN DI CHUYỂN TAXI NEW YORK")
st.markdown("**Đội thi:** UNETI27 | **Cuộc thi:** Olympic Trí tuệ Nhân tạo UNETI 2026 - Vòng 2")
st.divider()

preset = st.radio(
    "Chọn chế độ nhập liệu (Hỗ trợ nạp nhanh kịch bản mẫu hoặc tự tùy chỉnh):",
    [
        "Tùy chỉnh thủ công",
        "Kịch bản 1: Chuyến đi ngắn nội đô Midtown giờ cao điểm chiều (Thứ 5, 17h30)",
        "Kịch bản 2: Chuyến đi dài từ Manhattan ra Sân bay JFK ban đêm (Thứ 6, 22h15)"
    ],
    horizontal=False
)

if "Kịch bản 1" in preset:
    def_vendor, def_pass, def_month, def_dow, def_hour, def_min = 2, 3, 5, 3, 17, 30
    def_plon, def_plat, def_dlon, def_dlat, def_km, def_flag = -73.98, 40.75, -73.96, 40.78, 3.5, "N"
elif "Kịch bản 2" in preset:
    def_vendor, def_pass, def_month, def_dow, def_hour, def_min = 1, 1, 5, 4, 22, 15
    def_plon, def_plat, def_dlon, def_dlat, def_km, def_flag = -73.98, 40.75, -73.78, 40.64, 21.0, "N"
else:
    def_vendor, def_pass, def_month, def_dow, def_hour, def_min = 1, 1, 5, 4, 19, 15
    def_plon, def_plat, def_dlon, def_dlat, def_km, def_flag = -74.01, 40.72, -73.98, 40.76, 6.0, "N"

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("1. Thông tin chuyến xe")
    vendor_id = st.selectbox("Nhà vận tải (vendor_id)", [1, 2], index=[1, 2].index(def_vendor))
    passenger_count = st.slider("Số hành khách (passenger_count)", 1, 6, def_pass)
    store_and_fwd_flag = st.selectbox("Cờ lưu/đồng bộ (store_and_fwd_flag)", ["N", "Y"], index=["N", "Y"].index(def_flag))

with col2:
    st.subheader("2. Thời điểm đón khách")
    pickup_month = st.selectbox("Tháng đón (pickup_month)", [1, 2, 3, 4, 5, 6], index=def_month - 1)
    dow_map = {0: "0 - Thứ Hai", 1: "1 - Thứ Ba", 2: "2 - Thứ Tư", 3: "3 - Thứ Năm", 4: "4 - Thứ Sáu", 5: "5 - Thứ Bảy", 6: "6 - Chủ Nhật"}
    pickup_dayofweek = st.selectbox("Thứ trong tuần (pickup_dayofweek)", list(dow_map.keys()), format_func=lambda x: dow_map[x], index=def_dow)
    pickup_hour = st.slider("Giờ đón khách (pickup_hour: 0 - 23h)", 0, 23, def_hour)
    pickup_minute_bucket = st.select_slider("Mốc phút (pickup_minute_bucket)", options=[0, 15, 30, 45], value=def_min)

with col3:
    st.subheader("3. Tọa độ Điểm đón & Điểm đến")
    pickup_lon_approx = st.number_input("Kinh độ đón (pickup_lon_approx)", value=def_plon, step=0.01, format="%.2f")
    pickup_lat_approx = st.number_input("Vĩ độ đón (pickup_lat_approx)", value=def_plat, step=0.01, format="%.2f")
    dropoff_lon_approx = st.number_input("Kinh độ đến (dropoff_lon_approx)", value=def_dlon, step=0.01, format="%.2f")
    dropoff_lat_approx = st.number_input("Vĩ độ đến (dropoff_lat_approx)", value=def_dlat, step=0.01, format="%.2f")
    auto_km = calc_approx_km(pickup_lon_approx, pickup_lat_approx, dropoff_lon_approx, dropoff_lat_approx)
    straight_line_km_approx = st.number_input("Khoảng cách chim bay ước tính (km)", min_value=0.0, max_value=50.0, value=def_km if preset != "Tùy chỉnh thủ công" else auto_km, step=0.5)

st.divider()

if st.button("🚀 DỰ ĐOÁN THỜI GIAN DI CHUYỂN BẰNG MÔ HÌNH AI", type="primary", use_container_width=True):
    input_df = pd.DataFrame([{
        'vendor_id': vendor_id,
        'pickup_month': pickup_month,
        'pickup_dayofweek': pickup_dayofweek,
        'pickup_hour': pickup_hour,
        'pickup_minute_bucket': pickup_minute_bucket,
        'passenger_count': passenger_count,
        'pickup_lon_approx': pickup_lon_approx,
        'pickup_lat_approx': pickup_lat_approx,
        'dropoff_lon_approx': dropoff_lon_approx,
        'dropoff_lat_approx': dropoff_lat_approx,
        'straight_line_km_approx': straight_line_km_approx,
        'store_and_fwd_flag': store_and_fwd_flag
    }])
    
    X_in = prepare_single_input(input_df)
    pred_log = float(np.clip(model.predict(X_in)[0], np.log1p(60), np.log1p(7191)))
    pred_sec = float(np.expm1(pred_log))
    mins = int(pred_sec // 60)
    secs = int(round(pred_sec % 60))
    avg_speed = straight_line_km_approx / (pred_sec / 3600.0) if pred_sec > 0 else 0.0

    r1, r2, r3 = st.columns(3)
    r1.metric("⏱️ Thời gian dự đoán (Đơn vị chuẩn)", f"{pred_sec:.1f} giây")
    r2.metric("🕒 Quy đổi Phút & Giây", f"{mins} phút {secs} giây")
    r3.metric("🚗 Vận tốc chim bay ước tính", f"{avg_speed:.1f} km/h")
    
    st.success(f"Mô hình AI đã thực thi dự đoán trực tiếp (Raw output ln(1+y) = {pred_log:.4f} -> {pred_sec:.2f} giây).")