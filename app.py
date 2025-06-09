import os
import logging
from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from datetime import datetime
import asyncio
from chat_graph import ChatGraph

# Configure logging
logging.basicConfig(level=logging.DEBUG)

app = Flask(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

# Predefined user credentials
USERS = {
    '1948': {"user": "client", "company": "ainflows", "budget": 5},
    '1917': {"user": "demo", "company": "ainflows", "budget": 5}
}

# Available LLM models
AVAILABLE_MODELS = {
    'gemini-2.0-flash-exp': 'Gemini 2.0 Flash (Default)',
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-4o': 'GPT-4o',
    'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet'
}

@app.route('/')
def index():
    """Main route - check authentication and serve appropriate content"""
    if 'authenticated' not in session:
        return render_template('index.html', show_login=True)
    
    user_data = session.get('user_data', {})
    return render_template('index.html', 
                         show_login=False, 
                         user_data=user_data,
                         available_models=AVAILABLE_MODELS)

@app.route('/login', methods=['POST'])
def login():
    """Handle user login"""
    password = request.form.get('password', '').strip()
    
    if password in USERS:
        session['authenticated'] = True
        session['user_data'] = USERS[password]
        session['chat_history'] = []
        session['selected_model'] = 'gemini-2.0-flash-exp'  # Default model
        
        return jsonify({
            'success': True,
            'user': USERS[password]['user'],
            'company': USERS[password]['company']
        })
    else:
        return jsonify({'success': False, 'error': 'Invalid password'}), 401

@app.route('/logout')
def logout():
    """Handle user logout"""
    session.clear()
    return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        selected_model = data.get('model', session.get('selected_model', 'gemini-2.0-flash-exp'))
        
        if not message:
            return jsonify({'error': 'Message cannot be empty'}), 400
        
        # Update selected model in session
        session['selected_model'] = selected_model
        
        # Get chat history from session
        chat_history = session.get('chat_history', [])
        
        # Initialize chat graph
        chat_graph = ChatGraph(model=selected_model)
        
        # Create asyncio event loop for running async code
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Process message through langGraph
            response = loop.run_until_complete(
                chat_graph.process_message(message, chat_history)
            )
        finally:
            loop.close()
        
        # Add to chat history
        chat_history.append({
            'user': message,
            'bot': response,
            'timestamp': datetime.now().isoformat(),
            'model': selected_model
        })
        
        # Keep only last 50 exchanges to prevent session from getting too large
        if len(chat_history) > 50:
            chat_history = chat_history[-50:]
        
        session['chat_history'] = chat_history
        
        return jsonify({
            'response': response,
            'model_used': selected_model
        })
        
    except Exception as e:
        logging.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/new_chat', methods=['POST'])
def new_chat():
    """Start a new chat session"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    session['chat_history'] = []
    return jsonify({'success': True})

@app.route('/get_chat_history')
def get_chat_history():
    """Get current chat history"""
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    return jsonify({
        'history': session.get('chat_history', []),
        'selected_model': session.get('selected_model', 'gemini-2.0-flash-exp')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
