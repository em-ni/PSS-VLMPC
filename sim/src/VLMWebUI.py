# src/VLMWebUI.py
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import webbrowser

class VLMWebHandler(BaseHTTPRequestHandler):
    """HTTP request handler for the VLM web UI."""
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == '/' or self.path == '/index.html':
            self.send_response(200)
            self.send_header('Content-type', 'text/html')
            self.end_headers()
            self.wfile.write(self.get_html().encode())
        elif self.path == '/commands':
            # Get recent commands and responses
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            history = getattr(self.server, 'history', [])
            self.wfile.write(json.dumps(history[-20:]).encode())  # Last 20 items
        else:
            self.send_response(404)
            self.end_headers()
    
    def do_POST(self):
        """Handle POST requests."""
        if self.path == '/command':
            content_length = int(self.headers['Content-Length'])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            command = data.get('command', '').strip()
            if command:
                # Add to command queue
                self.server.command_queue.put(command)
                
                # Add to history
                if not hasattr(self.server, 'history'):
                    self.server.history = []
                self.server.history.append({
                    'type': 'command',
                    'text': command,
                    'timestamp': time.strftime('%H:%M:%S')
                })
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({'status': 'ok'}).encode())
            else:
                self.send_response(400)
                self.end_headers()
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress log messages."""
        pass
    
    def get_html(self):
        """Return the HTML for the UI."""
        return '''
<!DOCTYPE html>
<html>
<head>
    <title>VLM Robot Controller</title>
    <meta charset="utf-8">
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #2b2b2b;
            color: white;
            margin: 0;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
            color: #4CAF50;
        }
        .instructions {
            background-color: #3a3a3a;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }
        .history {
            background-color: #1e1e1e;
            border: 1px solid #555;
            border-radius: 5px;
            height: 300px;
            overflow-y: auto;
            padding: 10px;
            margin-bottom: 20px;
            font-family: 'Courier New', monospace;
            font-size: 20px;
        }
        .input-container {
            display: flex;
            gap: 10px;
            margin-bottom: 20px;
        }
        #commandInput {
            flex: 1;
            padding: 10px;
            background-color: #3a3a3a;
            border: 1px solid #555;
            border-radius: 3px;
            color: white;
            font-size: 20px;
        }
        #sendButton {
            padding: 10px 20px;
            background-color: #4CAF50;
            border: none;
            border-radius: 3px;
            color: white;
            cursor: pointer;
            font-size: 20px;
        }
        #sendButton:hover {
            background-color: #45a049;
        }
        .command {
            color: #4CAF50; 
        }
        .response {
            color: #FFA500;
        }
        .status {
            color: #87CEEB;
        }
        .timestamp {
            color: #888;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>PSS VLMPC Soft Robot</h1>
        
        <div class="instructions">
            Examples: "move right", "touch the red target", "avoid purple obstacle", "stop"
        </div>
        
        <div id="history" class="history"></div>
        
        <div class="input-container">
            <input type="text" id="commandInput" placeholder="Enter robot command..." />
            <button id="sendButton">Send</button>
        </div>
        
        <div style="text-align: center; color: #888; font-size: 18px;">
            Press Enter to send command | Commands are processed in real-time
        </div>
    </div>

    <script>
        const commandInput = document.getElementById('commandInput');
        const sendButton = document.getElementById('sendButton');
        const history = document.getElementById('history');
        
        // Focus on input
        commandInput.focus();
        
        // Send command function
        function sendCommand() {
            const command = commandInput.value.trim();
            if (!command) return;
            
            fetch('/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: command})
            })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'ok') {
                    commandInput.value = '';
                    updateHistory();
                }
            })
            .catch(error => console.error('Error:', error));
        }
        
        // Event listeners
        sendButton.addEventListener('click', sendCommand);
        commandInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                sendCommand();
            }
        });
        
        // Update history
        function updateHistory() {
            fetch('/commands')
            .then(response => response.json())
            .then(data => {
                history.innerHTML = '';
                data.forEach(item => {
                    const div = document.createElement('div');
                    const className = item.type;
                    div.innerHTML = `<span class="timestamp">[${item.timestamp}]</span> <span class="${className}">${item.text}</span>`;
                    history.appendChild(div);
                });
                history.scrollTop = history.scrollHeight;
            })
            .catch(error => console.error('Error:', error));
        }
        
        // Auto-update history every 2 seconds
        setInterval(updateHistory, 2000);
        updateHistory();
    </script>
</body>
</html>
'''

class VLMWebUI:
    """
    Web-based UI for VLM commands. Opens a web browser with a simple interface.
    """
    def __init__(self, command_queue, port=8765):
        self.command_queue = command_queue
        self.port = port
        self.running = False
        self.server = None
        self.server_thread = None
        
    def start_ui(self):
        """Start the web UI server."""
        if self.running:
            return
            
        self.running = True
        
        # Try multiple ports if the default is in use
        ports_to_try = [self.port, self.port + 1, self.port + 2, self.port + 3, self.port + 4]
        server_started = False
        
        for port in ports_to_try:
            try:
                # Create HTTP server
                self.server = HTTPServer(('localhost', port), VLMWebHandler)
                self.server.command_queue = self.command_queue
                self.server.history = []
                self.port = port  # Update to the actual port used
                server_started = True
                break
            except OSError as e:
                if e.errno == 98:  # Address already in use
                    print(f"Port {port} is in use, trying next port...")
                    continue
                else:
                    raise e
        
        if not server_started:
            print(f"Failed to start web UI: No available ports in range {ports_to_try[0]}-{ports_to_try[-1]}")
            self.running = False
            return
        
        try:
            
            # Start server in separate thread
            self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
            self.server_thread.start()
            
            # Open browser
            url = f"http://localhost:{self.port}"
            print(f"VLM Web UI started at: {url}")
            
            # # Try multiple methods to open browser
            # browser_opened = False
            
            # # Method 1: Try webbrowser module
            # try:
            #     webbrowser.open(url)
            #     browser_opened = True
            #     print("Browser opened automatically")
            # except Exception as e:
            #     print(f"webbrowser.open failed: {e}")
            
            # # Method 2: Try system commands if webbrowser failed
            # if not browser_opened:
            #     import subprocess
            #     try:
            #         # Try different commands based on the system
            #         commands = [
            #             ['xdg-open', url],  # Linux
            #             ['open', url],      # macOS
            #             ['start', url],     # Windows
            #             ['firefox', url],   # Firefox directly
            #             ['google-chrome', url],  # Chrome directly
            #             ['chromium-browser', url],  # Chromium
            #         ]
                    
            #         for cmd in commands:
            #             try:
            #                 subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            #                 browser_opened = True
            #                 print(f"Browser opened using: {cmd[0]}")
            #                 break
            #             except (subprocess.CalledProcessError, FileNotFoundError):
            #                 continue
                            
            #     except Exception as e:
            #         print(f"System command browser opening failed: {e}")
            
            # If all methods failed, provide manual instructions
            # if not browser_opened:
            #     print(f"Please manually open your web browser and navigate to:")
            #     print(f"    {url}")
                
        except Exception as e:
            print(f"Failed to start web UI: {e}")
            self.running = False
            
    def add_response(self, response):
        """Add a VLM response to the history."""
        if self.server and hasattr(self.server, 'history'):
            self.server.history.append({
                'type': 'response',
                'text': f"{response}",
                'timestamp': time.strftime('%H:%M:%S')
            })
            
    def add_status_update(self, status):
        """Add a status update to the history."""
        if self.server and hasattr(self.server, 'history'):
            self.server.history.append({
                'type': 'status',
                'text': f"{status}",
                'timestamp': time.strftime('%H:%M:%S')
            })
            
    def is_running(self):
        """Check if UI is running."""
        return self.running
        
    def stop(self):
        """Stop the web UI."""
        self.running = False
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.server_thread:
            self.server_thread.join(timeout=1.0)
        print("Web UI stopped")
