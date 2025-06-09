import os
import logging
import asyncio
from quart import Quart, render_template, request, jsonify, session, redirect, url_for, Response
from datetime import datetime
from chat_graph import ChatGraph

logging.basicConfig(level=logging.INFO)
app = Quart(__name__)
app.secret_key = os.environ.get("SESSION_SECRET", "dev-secret-key-change-in-production")

USERS = {
    '1948': {"user": "client", "company": "Umniah", "budget": 5, "persona": "You are a customer service AI for Umniah. Answer all questions as an Umniah representative, providing Umniah-specific help."},
    '1917': {"user": "demo", "company": "Trade Republic", "budget": 5, "persona": "You are a customer service AI for Trade Republic. Answer all questions as a Trade Republic representative, providing Trade Republic-specific help."}
}

AVAILABLE_MODELS = {
    'gemini-2.0-flash-exp': 'Gemini 2.0 Flash (Default)',
    'gpt-4o-mini': 'GPT-4o Mini',
    'gpt-4o': 'GPT-4o',
    'claude-3-5-sonnet-20241022': 'Claude 3.5 Sonnet'
}

@app.route('/')
async def index():
    if 'authenticated' not in session:
        return await render_template('index.html', show_login=True)
    user_data = session.get('user_data', {})
    return await render_template('index.html', 
                         show_login=False, 
                         user_data=user_data,
                         available_models=AVAILABLE_MODELS)

@app.route('/login', methods=['POST'])
async def login():
    data = await request.form
    password = data.get('password', '').strip()
    user = USERS.get(password)
    if user:
        session['authenticated'] = True
        session['user_data'] = user
        session['chat_history'] = []
        session['selected_model'] = 'gemini-2.0-flash-exp'
        return jsonify({
            'success': True,
            'user': user['user'],
            'company': user['company']
        })
    return jsonify({'success': False, 'error': 'Invalid password'}), 401

@app.route('/logout')
async def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/chat', methods=['POST'])
async def chat():
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401

    data = await request.get_json()
    message = data.get('message', '').strip()
    selected_model = data.get('model', session.get('selected_model', 'gemini-2.0-flash-exp'))
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    session['selected_model'] = selected_model
    chat_history = session.get('chat_history', [])
    user_data = session.get('user_data', {})
    company = user_data.get('company', 'the company')
    persona = user_data.get('persona', '')
    chat_graph = ChatGraph(model=selected_model, company=company, persona=persona)
    response = await chat_graph.process_message(message, chat_history)
    chat_history.append({
        'user': message,
        'bot': response,
        'timestamp': datetime.now().isoformat(),
        'model': selected_model
    })
    session['chat_history'] = chat_history[-50:]
    return jsonify({
        'response': response,
        'model_used': selected_model
    })

@app.route('/chat_stream', methods=['POST'])
async def chat_stream():
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    data = await request.get_json()
    message = data.get('message', '').strip()
    selected_model = data.get('model', session.get('selected_model', 'gemini-2.0-flash-exp'))
    if not message:
        return jsonify({'error': 'Message cannot be empty'}), 400
    session['selected_model'] = selected_model
    chat_history = session.get('chat_history', [])
    user_data = session.get('user_data', {})
    company = user_data.get('company', 'the company')
    persona = user_data.get('persona', '')
    chat_graph = ChatGraph(model=selected_model, company=company, persona=persona)
    async def streamer():
        async for chunk in chat_graph.stream_message(message, chat_history):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
        chat_history.append({
            'user': message,
            'bot': '',  # Let frontend fill
            'timestamp': datetime.now().isoformat(),
            'model': selected_model
        })
        session['chat_history'] = chat_history[-50:]
    return Response(streamer(), content_type="text/event-stream")

@app.route('/new_chat', methods=['POST'])
async def new_chat():
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    session['chat_history'] = []
    return jsonify({'success': True})

@app.route('/get_chat_history')
async def get_chat_history():
    if 'authenticated' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    return jsonify({
        'history': session.get('chat_history', []),
        'selected_model': session.get('selected_model', 'gemini-2.0-flash-exp')
    })