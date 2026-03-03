import csv
import uuid
from collections import Counter

import pandas as pd
from datetime import datetime
from pathlib import Path

from src import config
from helper_function.prints import *

class AnalyticsManager:
    def __init__(self):
        self.logs_dir = config.LOGS_DIR
        self.interactions_file = config.INTERACTIONS_FILE
        self.feedback_file = config.FEEDBACK_FILE
        self._initialize_storage() # Ensure directory and files exist

    def _initialize_storage(self):
        """Creates CSV files with headers if they don't exist."""
        if not config.ENABLE_ANALYTICS:
            return

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Security check, define what should be on the csv data
        interaction_schema = [
            "message_id", "session_id", "timestamp", "user_query", 
            "answer", "sources", "latency", "success_flag", 
            "category", "input_tokens", "output_tokens"
        ]
        
        feedback_schema = [
            "feedback_id", "timestamp", "message_id", 
            "thumb_up_down", "comment", "related_sources"
        ]
        
        def validate_and_setup(filepath, expected_columns):
            if not filepath.exists(): # If file doesn't exist -> create it
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(expected_columns)
                return
            try: # If file exists -> check if the schema is right
                df = pd.read_csv(filepath, nrows=0)
                existing_cols = list(df.columns)
                
                if existing_cols != expected_columns:
                    print(red(f"Schema mismatch in {filepath.name}."))
                    print(f"   Expected: {expected_columns}")
                    print(f"   Found:    {existing_cols}")
                    
                    # Fallback
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    backup_path = filepath.parent / f"{filepath.stem}_backup_{timestamp}.csv"
                    filepath.rename(backup_path)
                    print(orange(f"Fallback:  Old file backed up to: {backup_path.name}"))
                    
                    # Create Fresh File
                    with open(filepath, "w", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow(expected_columns)
                    print(green(f"Created fresh {filepath.name} with correct schema."))
                else:
                    print(green(f"All the given database matches, continue filling {filepath}..."))
            
            except Exception as e: # If file is corrupted, recreate it
                print(red(f"Error reading {filepath.name}: {e}. Recreating file."))
                with open(filepath, "w", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow(expected_columns)

        validate_and_setup(self.interactions_file, interaction_schema)
        validate_and_setup(self.feedback_file, feedback_schema)
    
    def _estimate_tokens(self, text: str) -> int:
        """
        Estimates tokens based on char count (Speed optimization).
        Rule of thumb: 1 token ~= 4 chars in English.
        """
        if not text: return 0
        return len(text) // 4

    def _categorize_intent(self, query: str) -> str:
        """
        Simple heuristic clustering/classification of user intent.
        (BERT classifier might be a great IDEA, in case of huge bunch of data).
        """
        q = query.lower()
        if any(w in q for w in ["warranty", "guarantee", "cover", "years"]):
            return "Warranty & Legal"
        elif any(w in q for w in ["charge", "charging", "battery", "range", "km", "electric"]):
            return "Technical / EV Specs"
        elif any(w in q for w in ["price", "cost", "buy", "lease", "finance"]):
            return "Sales & Configurator"
        elif any(w in q for w in ["service", "repair", "maintenance", "tires", "oil"]):
            return "After-Sales Service"
        else:
            return "General / Unclassified"

    def log_interaction(self, session_id: str, query: str, answer: str, sources: list, context_text: str, latency: float) -> str:
        """
        Logs a standard Q&A interaction.
        Returns the message_id to link with future feedback.
        """
        if not config.ENABLE_ANALYTICS:
            return None

        message_id = str(uuid.uuid4()) # unique id
        timestamp = datetime.now().isoformat()
        
        # Success Flag: Did the model say "I don't know"?
        success_flag = "FALSE" if "i don't know" in answer.lower() else "TRUE"
        
        input_tokens = self._estimate_tokens(query) + self._estimate_tokens(context_text)
        output_tokens = self._estimate_tokens(answer)
        
        category = self._categorize_intent(query)
        sources_str = " | ".join(sources) # Flatten list for CSV

        with open(self.interactions_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                message_id, 
                session_id,  
                timestamp, 
                query, 
                answer, 
                sources_str, 
                f"{latency:.4f}", 
                success_flag, 
                category, 
                input_tokens, 
                output_tokens
            ])
            
        return message_id

    def log_feedback(self, message_id: str, thumb_score: int, comment: str, sources: list):
        """
        Logs user feedback.
        thumb_score: 1 (Up) or 0 (Down)
        """
        if not config.ENABLE_FEEDBACK_COLLECTION:
            return

        feedback_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        sources_str = " | ".join(sources)

        with open(self.feedback_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                feedback_id, timestamp, message_id, 
                thumb_score, comment, sources_str
            ])
            print(orange(f"Feedback logged for {message_id}"))

    def get_dashboard_metrics(self):
        """
        Generates DataFrames and Metrics for the Streamlit Dashboard.
        """
        if not self.interactions_file.exists():
            return None, None, None, None, None
        df_int = pd.read_csv(self.interactions_file)
        
        total_in = df_int["input_tokens"].sum()
        total_out = df_int["output_tokens"].sum()
        est_cost = (
            (total_in / 1_000_000 * config.COST_PER_1M_INPUT_TOKENS) + 
            (total_out / 1_000_000 * config.COST_PER_1M_OUTPUT_TOKENS)
        )
        
        session_depth = df_int.groupby("session_id").size().mean()
        
        all_cited = []
        for s in df_int["sources"].dropna():
            if isinstance(s, str) and s.strip():
                all_cited.extend(s.split(" | ")) # SOURCE DIVERSITY
            
        source_counts = pd.Series(all_cited).value_counts().reset_index()
        source_counts.columns = ["Source", "Usage"]
        
        total_citations = source_counts["Usage"].sum()
        source_counts["Share"] = (source_counts["Usage"] / total_citations) * 100
        
        total = len(df_int)
        if total > 0:
            failed = len(df_int[df_int["success_flag"].astype(str).str.lower() == "false"]) # success flag for answer rate
            no_answer_rate = (failed / total) * 100
        else:
            no_answer_rate = 0

        cat_counts = df_int["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        df_feed = pd.DataFrame()
        if self.feedback_file.exists():
            df_feed = pd.read_csv(self.feedback_file)
            
        bad_sources = []
        if not df_feed.empty:
            thumbs_down = df_feed[df_feed["thumb_up_down"] == 0] # thumbs down = 0 
            for s in thumbs_down["related_sources"].dropna(): # avoid undefined nan
                if isinstance(s, str) and s.strip():
                    bad_sources.extend(s.split(" | "))
        bad_counts = pd.Series(bad_sources).value_counts()
 
        good_sources = []
        if not df_feed.empty:
            thumbs_up = df_feed[df_feed["thumb_up_down"] == 1]
            for s in thumbs_up["related_sources"].dropna():
                good_sources.extend(s.split(" | "))
        good_counts = pd.Series(good_sources).value_counts()

        source_stats = pd.DataFrame({
            "Thumbs Up": good_counts,
            "Thumbs Down": bad_counts
        }).fillna(0) # Fill NaN with 0 if a source has only ups or only downs
        
        source_stats["Total Feedback"] = source_stats["Thumbs Up"] + source_stats["Thumbs Down"]
        source_stats["Approval Rate"] = (source_stats["Thumbs Up"] / source_stats["Total Feedback"]) * 100
        
        # Ascending (0% approval is worst) 
        problematic_sources = source_stats.sort_values(
            ["Approval Rate", "Total Feedback"], 
            ascending=[True, False]
        ).reset_index()
        
        problematic_sources = problematic_sources.rename(columns={"index": "Source File"})

        metrics = {
            "total_queries": total,
            "avg_latency": df_int["latency"].mean(),
            "no_answer_rate": no_answer_rate,
            "est_cost": est_cost,
            "session_depth": session_depth
        }
        return metrics, cat_counts, source_counts, problematic_sources