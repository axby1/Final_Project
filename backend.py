

from flask import Flask, request, jsonify
from your_python_script import process_url  # Import your Python script or function here

app = Flask(__name__)

@app.route('/classify', methods=['POST'])
def classify_url():
    data = request.get_json()
    url = data.get('url')

    # Process the URL using your Python script or function
    category = process_url(url)

    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(port=3000)
