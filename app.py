from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import joblib

app = Flask(__name__)

# Tải mô hình và các đối tượng tiền xử lý
model = joblib.load('stacking_model.pkl')
selector = joblib.load('feature_selector.pkl')
scaler = joblib.load('scaler.pkl')
selected_features = joblib.load('selected_features.pkl')

# Định nghĩa các đặc trưng đầu vào mà người dùng cần nhập
# Các đặc trưng gốc cần để tính các đặc trưng tương tác
# Từ script huấn luyện, các đặc trưng gốc là tất cả cột trừ 'HeartDisease' và 'DiseaseLevel'
# Đặc trưng tương tác: BP_Cholesterol, Age_BP
# Người dùng nhập: Age, Gender, BloodPressure, Cholesterol, HeartRate, QuantumPatternFeature

input_features = ['Age', 'Gender', 'BloodPressure', 'Cholesterol', 'HeartRate', 'QuantumPatternFeature']

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        try:
            # Lấy dữ liệu người dùng nhập
            user_data = {}
            for feature in input_features:
                val = request.form.get(feature)
                if val is None or val == '':
                    return render_template('index.html', error=f'Thiếu giá trị cho {feature}', prediction=None)
                user_data[feature] = float(val)

            # Tạo dataframe từ dữ liệu người dùng
            df = pd.DataFrame([user_data])

            # Tính các đặc trưng tương tác
            df['BP_Cholesterol'] = df['BloodPressure'] * df['Cholesterol']
            df['Age_BP'] = df['Age'] * df['BloodPressure']

            # Chọn các đặc trưng theo thứ tự selected_features
            X = df[selected_features]

            # Áp dụng chuẩn hóa
            X_scaled = scaler.transform(X)

            # Dự đoán
            pred = model.predict(X_scaled)[0]
            pred_proba = model.predict_proba(X_scaled)[0]

            # Ánh xạ nhãn dự đoán sang tên
            labels = {0: 'Không bệnh', 1: 'Nhẹ', 2: 'Trung bình', 3: 'Nặng'}
            mapped_label = labels.get(pred, 'Không xác định')
            confidence = max(pred_proba) * 100
            prediction = f"{mapped_label} - Độ tin cậy: {confidence:.0f}%"

            # Convert pred_proba to list for JSON serialization
            pred_proba_list = pred_proba.tolist()

            return render_template('index.html', prediction=prediction, error=None, severity_level=mapped_label, user_data=user_data, pred_proba=pred_proba_list)

            return render_template('index.html', prediction=prediction, error=None, severity_level=mapped_label, user_data=user_data)
        except Exception as e:
            return render_template('index.html', error=str(e), prediction=None)

    return render_template('index.html', prediction=None, error=None, user_data={}, pred_proba=[])

if __name__ == '__main__':
    app.run(debug=True)
