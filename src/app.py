from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Lấy đường dẫn ảnh từ request
        image_path = request.json.get('image_path')
        if not image_path:
            return jsonify({'error': 'Thiếu đường dẫn ảnh'}), 400
        
        # Chạy lệnh predict bằng subprocess
        command = [
            "python", "./src/predict.py",
            "-I", image_path,
            "-S", "S",
            "-C", "model.pt"
        ]
        result = subprocess.run(command, capture_output=True, text=True)

        # Trả về kết quả dự đoán
        if result.returncode == 0:
            return jsonify({'prediction': result.stdout.strip()})
        else:
            return jsonify({'error': result.stderr.strip()}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
