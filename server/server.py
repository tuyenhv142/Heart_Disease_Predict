from flask import Flask, request, jsonify
from flask_cors import CORS
import util

app = Flask(__name__)
CORS(app)

@app.route('/predict_heart_disease', methods=['POST'])
def predict_heart():
    try:
        # Lấy dữ liệu JSON
        data = request.get_json()

        # Kiểm tra xem dữ liệu JSON có tồn tại không
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400

        # Trích xuất các tham số
        age = float(data['age'])
        sex = float(data['sex'])
        cp = float(data['cp'])
        trestbps = float(data['trestbps'])
        chol = float(data['chol'])
        fbs = float(data['fbs'])
        restecg = float(data['restecg'])
        thalach = float(data['thalach'])
        exang = float(data['exang'])
        thal = float(data['thal'])

        # Gọi hàm dự đoán
        prediction = util.get_predict_heart_disease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, thal)

        return jsonify({'predict': prediction})
    except KeyError as e:
        return jsonify({'error': f'Missing parameter: {str(e)}'}), 400
    except ValueError:
        return jsonify({'error': 'Invalid input. Ensure all inputs are numeric.'}), 400

if __name__ == "__main__":
    print("Starting Python Flask Server For Heart Disease Prediction...")
    util.load_saved()
    app.run()
