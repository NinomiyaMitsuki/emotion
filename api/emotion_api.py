import json
from flask import Flask, request
from .utils import json_response, JSON_MIME_TYPE


app = Flask(__name__)


@app.route('/emotion', methods=['POST'])
def emotion_analysis():
    if request.content_type != JSON_MIME_TYPE:
        error = json.dumps({'error': 'Invalid Content Type'})
        return json_response(error, 400)

    data = request.json
    if not all([data.get('text')]):
        error = json.dumps({'error': 'Missing field/s (text)'})
        return json_response(error, 400)

    # TODO 调用EMOTION方法
    data['score'] = 0.99
    data['result'] = ''

    content = json.dumps(data)
    return content, 200, {'Content-Type': JSON_MIME_TYPE}
