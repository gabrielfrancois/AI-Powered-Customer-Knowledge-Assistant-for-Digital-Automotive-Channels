import os
import sys
import shutil
import subprocess
import time
import argparse
from pathlib import Path

from helper_function.prints import *

# Define paths
ROOT_DIR = Path(__file__).parent
VECTOR_DB_PATH = ROOT_DIR / "data" / "vectorstore"
APP_PATH = ROOT_DIR / "src" / "app.py"

def clean_vector_db():
    """Deletes the existing vector database."""
    if VECTOR_DB_PATH.exists():
        print(orange(f"Removing existing vector database at {VECTOR_DB_PATH}..."))
        shutil.rmtree(VECTOR_DB_PATH)
    else:
        print("ℹ️  No existing database found to delete.")

def run_ingestion(force_restart: bool = False):
    """Runs the ingestion script to build the vector database."""
    print(orange("Checking Knowledge Base status..."))
    
    # Check if we need to run ingestion
    db_exists = VECTOR_DB_PATH.exists() and os.listdir(VECTOR_DB_PATH)
    
    if force_restart or not db_exists:
        if force_restart:
            clean_vector_db()
            print(orange("Restarting ingestion process (Fresh Build)..."))
        else:
            print(red("Database missing. Starting initial ingestion..."))
            
        try:
            # We use subprocess to run it exactly like "uv run -m src.rag.ingest"
            subprocess.run(["uv", "run", "-m", "src.rag.ingest"], check=True)
            print(green("Ingestion complete. Database is ready."))
        except subprocess.CalledProcessError as e:
            print(red(f"Ingestion failed: {e}"))
            sys.exit(1)
    else:
        print(green("Vector Database exists. Skipping ingestion."))

def launch_app():
    """Launches the Streamlit application."""
    print(green("Launching BMW AI Assistant..."))
    print("Press Ctrl+C to stop the server.")
    
    try:
        # Run streamlit via uv
        subprocess.run(["uv", "run", "streamlit", "run", str(APP_PATH)], check=True)
    except KeyboardInterrupt:
        print(orange("\nApp stopped by user."))

def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="BMW AI Assistant Launcher")
    parser.add_argument(
        "--restart-ingestion", 
        action="store_true", 
        help="Force delete and rebuild the vector database."
    )
    return parser.parse_args()

if __name__ == "__main__":
    print("BMW AI Assistant Launcher")
    print("============================")
    
    args = parse_arguments()
    
    run_ingestion(force_restart=args.restart_ingestion)
    
    time.sleep(1) # Small pause for UX
    launch_app()