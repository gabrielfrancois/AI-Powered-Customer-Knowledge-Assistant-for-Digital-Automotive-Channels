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

        if not self.interactions_file.exists():
            with open(self.interactions_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["message_id", "timestamp", "user_query", "answer", "sources", "latency (seconds)y", "success_flag", "category"])

        if not self.feedback_file.exists():
            with open(self.feedback_file, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["feedback_id", "timestamp", "message_id", "thumb_up_down", "comment", "related_sources"])

    def _categorize_intent(self, query: str) -> str:
        """
        Simple heuristic clustering/classification of user intent.
        In production, this would be a BERT classifier.
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

    def log_interaction(self, query: str, answer: str, sources: list, latency: float) -> str:
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
        
        category = self._categorize_intent(query)
        sources_str = " | ".join(sources) # Flatten list for CSV

        with open(self.interactions_file, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                message_id, timestamp, query, answer, 
                sources_str, f"{latency:.4f}", success_flag, category
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
            return None, None, None

        df_int = pd.read_csv(self.interactions_file)
        
        # No Answer Rate
        total = len(df_int)
        if total > 0:
            failed = len(df_int[df_int["success_flag"] == False]) # Assuming boolean conversion on read
            if df_int["success_flag"].dtype == object: 
                failed = len(df_int[df_int["success_flag"].str.lower() == "false"]) # Handle string 'FALSE' if pandas didn't convert automatically
            no_answer_rate = (failed / total) * 100
        else:
            no_answer_rate = 0

        # Get Performance by Category
        cat_counts = df_int["category"].value_counts().reset_index()
        cat_counts.columns = ["Category", "Count"]

        # Problematic Sources (from Feedback file, thumbs down)
        df_feed = pd.DataFrame()
        if self.feedback_file.exists():
            df_feed = pd.read_csv(self.feedback_file)
            
        all_sources = []
        for s in df_int["sources"].dropna():
            all_sources.extend(s.split(" | "))
        source_usage = pd.Series(all_sources).value_counts()

        # Count times each source got a Thumbs Down
        bad_sources = []
        if not df_feed.empty:
            thumbs_down = df_feed[df_feed["thumb_up_down"] == 0]
            for s in thumbs_down["related_sources"].dropna():
                bad_sources.extend(s.split(" | "))
        source_complaints = pd.Series(bad_sources).value_counts()

        # Calculate Rate (Complaints / Usage)
        source_stats = pd.DataFrame({
            "Total Citations": source_usage,
            "Negative Feedback": source_complaints
        }).fillna(0)
        
        # Calculate Rejection %
        source_stats["Rejection Rate"] = (source_stats["Negative Feedback"] / source_stats["Total Citations"]) * 100
        
        # Filter to show only sources that actually have complaints
        problematic_sources = source_stats[source_stats["Negative Feedback"] > 0].sort_values("Rejection Rate", ascending=False).reset_index()
        problematic_sources = problematic_sources.rename(columns={"index": "Source File"})

        metrics = {
            "total_queries": total,
            "avg_latency": df_int["latency"].mean(),
            "no_answer_rate": no_answer_rate
        }

        return metrics, cat_counts, problematic_sources