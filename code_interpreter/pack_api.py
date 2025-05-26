from flask import Flask, request, jsonify
from code_interpreter import code_interpreter
app = Flask(__name__)

@app.route('/api/code_interpreter', methods=['POST'])
def api_some_function():
    data = request.json
    param1 = data.get('code')
    
    result = code_interpreter(param1)
    
    return jsonify({'result': result})


if __name__ == '__main__':
    from gevent import pywsgi
    server = pywsgi.WSGIServer(('0.0.0.0',4200),app)
    server.serve_forever()
    # app.run(host='0.0.0.0', port=4200)
    # print("ok")