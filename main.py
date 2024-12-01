from flask import Flask, request, render_template, jsonify
from assistant_tool import AssistantTool

app = Flask(__name__)
assistant_tool = AssistantTool()

@app.route('/')
def hello_worl():
    return render_template(template_name_or_list='index.html')


@app.route('/answer', methods=['POST'])
def  answer():
    data = request.json or {}
    user_query = data.get('query', 'No query provided')
    answer = assistant_tool.query_response(user_query)
    return jsonify({"message": answer})