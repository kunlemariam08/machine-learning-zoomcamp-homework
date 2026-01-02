from flask import Flask

app = Flask('ping')

@app.route('/ping')
def ping():
    return 'pong'

