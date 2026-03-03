import os
import sys
import shutil
import subprocess
import time
import argparse
from pathlib import Path

# Import your color functions
from helper_function.prints import orange, red, green

# Define paths
ROOT_DIR = Path(__file__).parent
VECTOR_DB_PATH = ROOT_DIR / "data" / "vectorstore"
APP_PATH = ROOT_DIR / "src" / "app.py"

def clean_vector_db():
    """Deletes the existing vector database."""
    if VECTOR_DB_PATH.exists():
        print(orange(f"🧹 Removing existing vector database at {VECTOR_DB_PATH}..."))
        shutil.rmtree(VECTOR_DB_PATH)
    else:
        print("No existing database found to delete.")

def run_ingestion(force_restart: bool = False):
    """Runs the ingestion script."""
    print(orange("🔍 Checking Knowledge Base status..."))
    
    db_exists = VECTOR_DB_PATH.exists() and os.listdir(VECTOR_DB_PATH)
    
    if force_restart or not db_exists:
        if force_restart:
            clean_vector_db()
            print(orange("Restarting ingestion process..."))
        else:
            print(red("Database missing. Starting initial ingestion..."))
            
        try:
            env = os.environ.copy()
            subprocess.run(["uv", "run", "-m", "src.rag.ingest"], check=True, env=env)
            print(green("Ingestion complete."))
        except subprocess.CalledProcessError as e:
            print(red(f"Ingestion failed: {e}"))
            sys.exit(1)
    else:
        print(green("Vector Database exists. Skipping ingestion."))

def launch_app():
    """Launches Streamlit simply."""
    print(green("Launching BMW AI Assistant..."))
    print("Press Ctrl+C to stop the server.")
    
    try:
        env = os.environ.copy()          
        env["TOKENIZERS_PARALLELISM"] = "false"  
        subprocess.run(
            [
                "uv", "run", "streamlit", "run", str(APP_PATH),
                "--server.fileWatcherType", "none",  
                "--server.headless", "false"         # Ensure it pops open
            ], 
            check=True,
            env=env
        )
    except KeyboardInterrupt:
        print(green("\nApp stopped by user."))

def parse_arguments():
    parser = argparse.ArgumentParser(description="BMW AI Assistant Launcher")
    parser.add_argument("--restart-ingestion", action="store_true", help="Rebuild DB.")
    return parser.parse_args()

if __name__ == "__main__":
    print("BMW AI Assistant Launcher")
    print("============================")
    
    args = parse_arguments()
    
    run_ingestion(force_restart=args.restart_ingestion)
    
    time.sleep(1) 
    launch_app()