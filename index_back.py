#this is the backend server for index.html file inside templates folder

from flask import Flask, render_template, request, jsonify
from optimised_algo import main 

app = Flask(__name__)
"""In a Flask application, the templates folder is a conventional location where Flask expects to find your HTML templates.
 This convention helps Flask organize and locate templates automatically. When Flask renders a template using render_template,
it looks for the specified template file inside the templates folder.
"""

@app.route('/')
def index():
    return render_template('index.html')

"""
here /process_url is an example of a route (or endpoint) in your Flask application. 
In the provided example, it's a route that handles POST requests and is designed to process a URL sent from the frontend.
"""

@app.route('/process_url', methods=['POST'])

def process_url():
    data = request.get_json()
    url = data.get('url', '')

    # Process the URL (add your processing logic here)
    #result = f"Processed URL: {url}"
    result=main(url)

    return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
