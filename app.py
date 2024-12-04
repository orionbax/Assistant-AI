from flask import Flask, request, render_template, jsonify, url_for
# import assistant_tool
from assistant_tool import AssistantTool
import os 
import tempfile
import shutil

# Create a singleton instance outside of the Flask app
assistant_tool = AssistantTool()

# Initialize Flask app
app = Flask(__name__, static_folder='static')


# Add basic configuration
app.config.update(
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max file size
    UPLOAD_FOLDER=tempfile.gettempdir()
)

@app.route('/start', methods=['GET'])
def start():
    """Return the sample data dictionary."""
    return jsonify({})


@app.route('/')
def chat_interface():
    """Render the chat interface."""
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    """Process user query and return response."""
    try:
        data = request.json
        user_query = data.get('message')
        
        if not user_query:
            return jsonify({
                "type": "podcast",
                "answer": "No query provided"
            }), 400
            
        response = assistant_tool.query_response(user_query)   
        return response
        
    except Exception as e:
        return jsonify({
            "type": "podcast",
            "answer": "An error occurred while processing your request"
        }), 500

@app.route('/indices', methods=['GET'])
def list_indices():
    """Return list of available indices."""
    try:
        indices = assistant_tool.list_indices()
        return jsonify({'indices': indices}), 200
    except Exception as e:
        return jsonify({'error': f'Error retrieving indices: {str(e)}'}), 500




@app.route('/upload', methods=['POST'])
def upload_files():
    try:
        index = request.form.get('index').lower()
        if not index:
            return jsonify({'message': 'Index is required'}), 400

        if 'files' not in request.files:
            return jsonify({'message': 'No files provided'}), 400

        files = request.files.getlist('files')
        if not files:
            return jsonify({'message': 'No files selected'}), 400

        temp_dir = tempfile.mkdtemp()
        try:
            for file in files:
                if file.filename:
                    file.save(os.path.join(temp_dir, file.filename))

            assistant_tool.add_vector_to_index(temp_dir, index)
            
            return jsonify({'message': 'Files uploaded and processed successfully'}), 200
        finally:
            shutil.rmtree(temp_dir)

    except Exception as e:
        return jsonify({'message': f'Error processing upload: {str(e)}'}), 500

@app.route('/refine')
def refine_content():
    """Render the content refinement interface."""
    return render_template('index.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
