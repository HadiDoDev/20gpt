from flask import Flask, request, jsonify
import subprocess

app = Flask(__name__)

@app.route('/run_command', methods=['POST'])
def run_command():
    try:
        data = request.json

        user_id = data['user_id']
        total_rials = data['total_rials']
        chat_modes = ','.join(data['chat_modes'])

        command = f"docker exec -it 20gpt_bot python3 bot/increase_credit.py --user_id={user_id} --total_rials={total_rials} --chat_modes={chat_modes}"

        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)

        return jsonify({'output': result.stdout, 'error': result.stderr}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
