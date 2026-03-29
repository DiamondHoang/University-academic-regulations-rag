import json
import os
import re
import sys
import asyncio
import pandas as pd
from tqdm import tqdm
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    context_precision,
    context_recall,
    answer_correctness
)
from ragas.embeddings import HuggingFaceEmbeddings
import time
from openai import AsyncOpenAI

from uni_rag import UniversityRAG
from loader.doc_loader import RegulationDocumentLoader
from config import Config
from ragas.llms import llm_factory


# =========================
# CONFIGURATION
# =========================
class EvalConfig:
    """Centralized configuration for RAG evaluation"""
    DATASET_PATH = "dataset.json"
    OUTPUT_DIR = "rag_eval_results"
    TOP_K = 7  # Number of documents to retrieve per question

    NUM_QUESTIONS = 40  # Set to an integer to limit, None for all

    # File paths
    RAG_RESULTS_FILE = f"{OUTPUT_DIR}/rag_answers.json"
    FULL_RESULTS_FILE = f"{OUTPUT_DIR}/full_results.csv"
    EVAL_CHECKPOINT_FILE = f"{OUTPUT_DIR}/eval_checkpoint.json"
    SUMMARY_FILE = f"{OUTPUT_DIR}/summary.csv"

    # Batching and retry settings
    EVAL_BATCH_SIZE = 20
    SLEEP_BETWEEN_BATCHES = 0.5
    RAG_RETRY_ATTEMPTS = 3
    EVAL_RETRY_ATTEMPTS = 2
    # Delay between individual answer generation calls (seconds).
    # Helps avoid Groq rate-limit spikes.
    ANSWER_DELAY = 3

    # Groq API settings (OpenAI-compatible)
    GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
    GROQ_BASE_URL = "https://api.groq.com/openai/v1"
    RAGAS_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    EMBEDDING_MODEL = "BAAI/bge-m3"

    @classmethod
    def ensure_output_dir(cls):
        """Create output directory if it doesn't exist"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


# =========================
# API CLIENTS
# =========================
class APIClients:
    """Lazy singleton for shared API clients (only needed for evaluation phase)"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize Groq-backed OpenAI-compatible client and RAGAS wrappers"""
        self.async_client = AsyncOpenAI(
            api_key=EvalConfig.GROQ_API_KEY,
            base_url=EvalConfig.GROQ_BASE_URL,
            timeout=60.0,
            max_retries=3,
        )
        self.ragas_llm = llm_factory(client=self.async_client, model=EvalConfig.RAGAS_MODEL)
        self.embeddings = HuggingFaceEmbeddings(model=EvalConfig.EMBEDDING_MODEL)


# =========================
# DATA MANAGEMENT
# =========================
class DataManager:
    """Handles loading and saving of datasets and results"""

    @staticmethod
    def load_dataset(limit=None):
        """Load evaluation dataset with optional limit"""
        with open(EvalConfig.DATASET_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)
        if limit and limit > 0:
            dataset = dataset[:limit]
        return dataset

    @staticmethod
    def load_rag_answers():
        """Load previously generated RAG answers"""
        if os.path.exists(EvalConfig.RAG_RESULTS_FILE):
            with open(EvalConfig.RAG_RESULTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    @staticmethod
    def save_rag_answers(rag_answers):
        """Save RAG answers to file"""
        with open(EvalConfig.RAG_RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump(rag_answers, f, ensure_ascii=False, indent=2)

    @staticmethod
    def results_exist():
        """Check if evaluation results already exist"""
        return os.path.exists(EvalConfig.FULL_RESULTS_FILE)


# =========================
# CITATION / FOOTER STRIPPER
# =========================
def strip_citations_and_footer(text: str) -> str:
    """Remove inline citation markers and source footer from a generated answer.

    Removes:
        - Inline citations: [1], [2], etc.
        - Source footer block starting with 'Nguồn tham khảo:' or 'References:'
    """
    # Remove source footer (everything from the footer header onwards)
    text = re.sub(r'\n*(Nguồn tham khảo|References)\s*:\s*\n.*', '', text, flags=re.DOTALL)
    # Remove inline citation markers like [1], [2], [12]
    text = re.sub(r'\[\d+\]', '', text)
    # Clean up extra spaces before punctuation and collapse multiple spaces
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    text = re.sub(r'  +', ' ', text)
    return text.strip()


# =========================
# RAG INITIALIZATION
# =========================
class RAGInitializer:
    """Handles RAG system initialization"""

    @staticmethod
    def initialize():
        """Initialize RAG system with documents"""
        loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
        rag = UniversityRAG()

        documents = loader.load_documents()
        rag.build_vectorstore(documents, force_rebuild=False)

        return rag


# =========================
# ANSWER GENERATOR  (Phase 1)
# =========================
class RAGAnswerGenerator:
    """Phase 1: Generate and cache RAG answers from the dataset."""

    def __init__(self, rag: UniversityRAG):
        self.rag = rag

    def _query(self, question: str):
        """Run a single RAG query and return (contexts, clean_answer)."""
        try:
            retrieved_docs = asyncio.run(
                self.rag.retriever.retrieve(question, k=EvalConfig.TOP_K)
            )

            if not retrieved_docs:
                return [], "Tôi không tìm thấy thông tin liên quan."

            ranked_docs = retrieved_docs[:EvalConfig.TOP_K]
            contexts = [doc.page_content for doc in ranked_docs]

            result = asyncio.run(
                self.rag.response_generator.agenerate(
                    query=question,
                    documents=ranked_docs,
                    conversation_history="",
                )
            )

            raw_answer = result.get("answer", "")
            clean_answer = strip_citations_and_footer(raw_answer)
            return contexts, clean_answer

        except Exception as e:
            return [], f"Lỗi xử lý: {str(e)[:80]}"

    def generate(self, dataset):
        """Generate answers for all dataset entries, with resume support.

        Already-processed question IDs (from a previous run) are skipped.
        Results are saved incrementally after each question.
        """
        rag_answers = DataManager.load_rag_answers()
        processed_ids = {sample["id"] for sample in rag_answers}

        pending = [qa for qa in dataset if qa["id"] not in processed_ids]
        if not pending:
            print("All answers already cached. Nothing to generate.")
            return

        print(f"Generating answers for {len(pending)} questions (skipping {len(processed_ids)} cached)")

        try:
            for qa in tqdm(pending, desc="Generating RAG answers"):
                contexts, answer = self._query(qa["question"])

                rag_answers.append({
                    "id": qa["id"],
                    "question": qa["question"],
                    "answer": answer,
                    "ground_truth": qa["answer"],
                    "contexts": contexts,
                })

                DataManager.save_rag_answers(rag_answers)
                time.sleep(EvalConfig.ANSWER_DELAY)

        except KeyboardInterrupt:
            DataManager.save_rag_answers(rag_answers)
            print("\nInterrupted — partial results saved.")
            raise

        DataManager.save_rag_answers(rag_answers)
        print(f"Done. Results saved to {EvalConfig.RAG_RESULTS_FILE}")


# =========================
# METRICS EVALUATOR  (Phase 2)
# =========================
class MetricsEvaluator:
    """Phase 2: Evaluate the cached rag_answers.json using RAGAS metrics."""

    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = [
            faithfulness,
            context_precision,
            context_recall,
            answer_correctness,
        ]

    def evaluate_batch(self, batch, attempt=0):
        """Evaluate a single batch; returns a DataFrame of metric scores."""
        max_retries = EvalConfig.EVAL_RETRY_ATTEMPTS
        try:
            batch_results = evaluate(
                batch,
                metrics=self.metrics,
                llm=self.llm,
                embeddings=self.embeddings,
            )
            return batch_results.to_pandas()

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                time.sleep(wait_time)
            return pd.DataFrame({
                metric.name: [None] * len(batch) for metric in self.metrics
            })

    def evaluate_dataset(self, dataset, batch_size=None):
        """Evaluate entire dataset in batches with checkpoint support."""
        if batch_size is None:
            batch_size = EvalConfig.EVAL_BATCH_SIZE

        all_results = []
        total_samples = len(dataset)

        # Resume from checkpoint if available
        checkpoint_data = {}
        if os.path.exists(EvalConfig.EVAL_CHECKPOINT_FILE):
            with open(EvalConfig.EVAL_CHECKPOINT_FILE, "r") as f:
                checkpoint_data = json.load(f)

        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch_indices = range(i, batch_end)

            batch_ids = [dataset[idx]["id"] for idx in batch_indices]
            if all(bid in checkpoint_data for bid in batch_ids):
                batch_df = pd.DataFrame([checkpoint_data[bid] for bid in batch_ids])
                all_results.append(batch_df)
                continue

            batch_data = dataset.select(batch_indices)
            batch_results_df = self.evaluate_batch(batch_data)

            for idx, row in batch_results_df.iterrows():
                checkpoint_data[batch_ids[idx]] = row.to_dict()

            with open(EvalConfig.EVAL_CHECKPOINT_FILE, "w") as f:
                json.dump(checkpoint_data, f)

            all_results.append(batch_results_df)

            if batch_end < total_samples:
                time.sleep(EvalConfig.SLEEP_BETWEEN_BATCHES)

        return pd.concat(all_results, ignore_index=True)


# =========================
# RESULTS ANALYZER
# =========================
class ResultsAnalyzer:
    """Analyzes and saves evaluation results"""

    @staticmethod
    def save_results(df, dataset):
        """Save full results with question IDs"""
        df["id"] = dataset["id"]
        df.to_csv(EvalConfig.FULL_RESULTS_FILE, index=False)

    @staticmethod
    def generate_summary(df):
        """Generate and save summary statistics"""
        summary = pd.DataFrame({"overall": df.mean(numeric_only=True)})
        summary.to_csv(EvalConfig.SUMMARY_FILE)
        return summary

    @staticmethod
    def print_summary(df):
        """Print mean metric scores to stdout"""
        print("\n===== Evaluation Summary =====")
        for col in df.select_dtypes("number").columns:
            print(f"  {col}: {df[col].mean():.4f}")
        print("==============================\n")


# =========================
# PHASE ORCHESTRATORS
# =========================

def run_phase1(limit=None):
    """Phase 1 — Generate RAG answers and save to rag_answers.json."""
    EvalConfig.ensure_output_dir()
    dataset = DataManager.load_dataset(limit=limit)

    print(f"=== Phase 1: Answer Generation ===")
    print(f"Dataset size: {len(dataset)} questions")

    rag = RAGInitializer.initialize()
    generator = RAGAnswerGenerator(rag)
    generator.generate(dataset)


def run_phase2():
    """Phase 2 — Load rag_answers.json and evaluate with RAGAS + Groq."""
    EvalConfig.ensure_output_dir()

    if not os.path.exists(EvalConfig.RAG_RESULTS_FILE):
        print(f"ERROR: {EvalConfig.RAG_RESULTS_FILE} not found. Run Phase 1 first.")
        return

    print(f"=== Phase 2: RAGAS Evaluation ===")
    rag_data = DataManager.load_rag_answers()
    print(f"Loaded {len(rag_data)} answers from {EvalConfig.RAG_RESULTS_FILE}")

    if DataManager.results_exist():
        print(f"Results already exist at {EvalConfig.FULL_RESULTS_FILE}")
        df = pd.read_csv(EvalConfig.FULL_RESULTS_FILE)
        ResultsAnalyzer.print_summary(df)
        return

    eval_dataset = Dataset.from_list(rag_data)
    clients = APIClients()

    evaluator = MetricsEvaluator(clients.ragas_llm, clients.embeddings)
    df = evaluator.evaluate_dataset(eval_dataset)

    ResultsAnalyzer.save_results(df, eval_dataset)
    ResultsAnalyzer.generate_summary(df)
    ResultsAnalyzer.print_summary(df)
    print(f"Full results saved to {EvalConfig.FULL_RESULTS_FILE}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    # Usage:
    #   python eval.py phase1          -- generate answers only
    #   python eval.py phase2          -- evaluate cached answers
    #   python eval.py                 -- run both phases sequentially
    phase = sys.argv[1] if len(sys.argv) > 1 else "all"

    if phase == "phase1":
        run_phase1(limit=EvalConfig.NUM_QUESTIONS)
    elif phase == "phase2":
        run_phase2()
    else:
        run_phase1(limit=EvalConfig.NUM_QUESTIONS)
        run_phase2()