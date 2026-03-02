#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"
LOGO_PATH="$DIR/visual/bmw-logo.png"

if [ -f "$LOGO_PATH" ]; then
    ICON_CMD="with icon POSIX file \"$LOGO_PATH\""
else
    ICON_CMD="with icon note"
fi

response=$(osascript -e "display dialog \"BMW AI Assistant Launcher\n\nChoose a launch mode:\" buttons {\"Cancel\", \"Run + Ingest\", \"Run\"} default button \"Run\" $ICON_CMD with title \"BMW AI Launcher\"")

if [[ "$response" == *"button returned:Cancel"* ]]; then
    osascript -e 'tell application "Terminal" to close first window' & exit
fi

if ! command -v uv &> /dev/null; then
    osascript -e 'display alert "Error: uv not found" message "Please install uv (astral-sh/uv) to run this application." as critical'
    exit 1
fi

echo "Starting application..."

if [[ "$response" == *"button returned:Run + Ingest"* ]]; then
    echo "♻️  Rebuilding Knowledge Base..."
    uv run python main.py --restart-ingestion
else
    # Standard Run
    uv run python main.py
fi

echo "Closing..."
osascript -e 'tell application "Terminal" to close first window' & exit