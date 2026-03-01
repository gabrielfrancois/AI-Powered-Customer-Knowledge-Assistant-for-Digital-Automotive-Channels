#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$DIR"

response=$(osascript -e 'display dialog "Run BMW AI Assistant?\n\nEnsure Homebrew and uv are installed." buttons {"Cancel", "Run"} default button "Run" with icon note with title "BMW AI Launcher"')

if [[ "$response" != *"button returned:Run"* ]]; then
    osascript -e 'tell application "Terminal" to close first window' & exit
fi


if ! command -v uv &> /dev/null; then
    osascript -e 'display alert "Error: uv not found" message "Please install uv (astral-sh/uv) to run this application." as critical'
    exit 1
fi

echo "Starting application..."
uv run python main.py

echo "Closing..."
osascript -e 'tell application "Terminal" to close first window' & exit
