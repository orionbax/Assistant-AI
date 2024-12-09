from flask import Flask, request, render_template, jsonify, url_for, session
from assistant_tool import AssistantTool
import os 
import tempfile
import shutil
from uuid import uuid4
from datetime import timedelta

# Create a singleton instance outside of the Flask app
assistant_tool = AssistantTool()

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.secret_key = os.urandom(24)  # Required for session management

# Add these configurations if needed
app.config.update(
    SESSION_COOKIE_SECURE=True,  # For HTTPS only
    PERMANENT_SESSION_LIFETIME=timedelta(days=1),  # Session timeout
    SESSION_COOKIE_SAMESITE='Lax'  # Cookie security
)

@app.route('/start', methods=['GET'])
def start():
    """Initialize a new session for the user."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid4())
    return jsonify({'user_id': session['user_id']})

@app.route('/')
def chat_interface():
    """Render the chat interface."""
    if 'user_id' not in session:
        session['user_id'] = str(uuid4())
    return render_template('index.html')

@app.route('/answer', methods=['POST'])
def answer():
    """Process user query and return response."""
    try:
        if 'user_id' not in session:
            session['user_id'] = str(uuid4())
            
        data = request.json
        user_query = data.get('message')
        
        if not user_query:
            return jsonify({
                "type": "podcast",
                "answer": "No query provided"
            }), 400
            
        response = assistant_tool.query_response(user_query, session['user_id'])   
        return response
        
    except Exception as e:
        return jsonify({
            "type": "podcast",
            "answer": "An error occurred while processing your request"
        }), 500

@app.route('/indices', methods=['POST'])
def list_indices():
    """Return list of available indices."""
    try:
        indices = assistant_tool.list_indices()
        print(indices)
        return jsonify({'indices': indices}), 200
    except Exception as e:
        return jsonify({'error': f'Error retrieving indices: {str(e)}'}), 500


# Return persona
@app.route('/personas', methods=['POST'])
def get_personas():
    """Return list of available personas."""
    try:
        personas = assistant_tool.get_personas()
        return jsonify({'personas': personas}), 200
    except Exception as e:
        return jsonify({'error': f'Error retrieving personas: {str(e)}'}), 500

@app.route('/output_types', methods=['POST'])
def get_output_types():
    """Return a list of output types"""
    try:
        output_types = assistant_tool.get_output_types()
        return jsonify({'output_types': output_types})
    except Exception as err:
        return jsonify([f'error in fetching output types! {err}'])
    

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

@app.route('/refine_content')
def refine_content():
    """Render the content refinement interface."""
    return render_template('refine_content.html')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
