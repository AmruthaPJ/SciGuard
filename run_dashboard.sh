#!/bin/bash

# 1. Start the Flask Backend in the background
echo "🚀 Starting Sci-Guard Backend..."
python app.py &

# 2. Wait for the server to initialize
sleep 3

# 3. Open the Dashboard in the default browser
echo "🌐 Opening Forensic Dashboard..."
open frontend/index.html

echo "✅ Sci-Guard is now running."
echo "Press Ctrl+C in this terminal to stop the backend later."

# Keep the script running so the user can kill the background process with Ctrl+C
wait
