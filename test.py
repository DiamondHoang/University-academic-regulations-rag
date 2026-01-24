import json
import os
import re
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
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

from uni_rag import UniversityRAG
from loader.doc_loader import RegulationDocumentLoader
from config import Config
from ragas.llms import llm_factory
from openai import AsyncOpenAI, RateLimitError


# =========================
# CONFIGURATION
# =========================
class EvalConfig:
    """Centralized configuration for RAG evaluation"""
    DATASET_PATH = "dataset.json"
    OUTPUT_DIR = "rag_eval_results"
    TOP_K = 5
    NUM_QUESTIONS = 50  # Set to None for all questions, or specify a number (e.g., 50)
    
    # File paths
    CHECKPOINT_FILE = f"{OUTPUT_DIR}/checkpoint.json"
    RAG_RESULTS_FILE = f"{OUTPUT_DIR}/rag_answers.json"
    FULL_RESULTS_FILE = f"{OUTPUT_DIR}/full_results.csv"
    SUMMARY_FILE = f"{OUTPUT_DIR}/summary.csv"
    
    # Batching and retry settings
    RAG_BATCH_SIZE = 10
    EVAL_BATCH_SIZE = 20
    SLEEP_BETWEEN_BATCHES = 0.5
    RAG_RETRY_ATTEMPTS = 3
    EVAL_RETRY_ATTEMPTS = 2
    RAG_SLEEP_DELAY = 0.5
    
    # API settings
    OPENROUTER_API_KEY = os.getenv('OPENROUTER_API_KEY')
    OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
    RAGAS_MODEL = "openai/gpt-4o-mini"
    EMBEDDING_MODEL = "BAAI/bge-m3"
    
    @classmethod
    def ensure_output_dir(cls):
        """Create output directory if it doesn't exist"""
        os.makedirs(cls.OUTPUT_DIR, exist_ok=True)


# =========================
# UTILITY FUNCTIONS
# =========================


# =========================
# API CLIENTS
# =========================
class APIClients:
    """Singleton for API clients"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize API clients"""
        self.async_client = AsyncOpenAI(
            api_key=EvalConfig.OPENROUTER_API_KEY,
            base_url=EvalConfig.OPENROUTER_BASE_URL,
            timeout=60.0,
            max_retries=3
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
            print(f"Limited dataset to {limit} questions")
        
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
# RAG INITIALIZATION
# =========================
class RAGInitializer:
    """Handles RAG system initialization"""
    
    @staticmethod
    def initialize():
        """Initialize RAG system with documents"""
        loader = RegulationDocumentLoader(base_path=Config.BASE_PATH)
        rag = UniversityRAG(config={"use_hybrid": True})
        
        documents = loader.load_documents()
        rag.build_vectorstore(documents, force_rebuild=False)
        
        return rag


# =========================
# RAG QUERY HANDLER
# =========================
class RAGQueryHandler:
    """Handles RAG queries with retry logic"""
    
    def __init__(self, rag):
        self.rag = rag
    
    @retry(
        stop=stop_after_attempt(EvalConfig.RAG_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, Exception)),
        before_sleep=lambda retry_state: print(f"Retry {retry_state.attempt_number}...")
    )
    def query(self, question):
        """Execute RAG query with automatic retry on rate limit"""
        try:
            intent = self.rag.query_analyzer.analyze(
                question,
                self.rag.memory.get_history()
            )
            
            search_query = intent["enhanced_query"]
            
            retrieved_docs = self.rag.retriever.retrieve(
                search_query,
                k=self.rag.config["max_retrieved_docs"],
                doc_type=None,
                regulation_type=None,
            )
            
            if not retrieved_docs:
                return {"enhanced_query": search_query}, [], "Tôi không tìm thấy thông tin liên quan."
            
            ranked_docs = self.rag.retriever.rank_by_recency(retrieved_docs)[:3]
            contexts = [doc.page_content for doc in ranked_docs]
            
            result = self.rag.response_generator.generate(
                question=question,
                documents=ranked_docs,
                conversation_history="",
                analysis_result=intent,
                clean_mode=True  # Dùng clean mode cho dataset
            )
            
            # Answer đã được clean trong response_generator._generate_clean
            return intent, contexts, result["answer"]
        except Exception as e:
            print(f"Query processing error: {type(e).__name__}: {e}")
            return {"enhanced_query": question}, [], f"Lỗi xử lý: {str(e)[:50]}"


# =========================
# RAG ANSWER GENERATOR
# =========================
class RAGAnswerGenerator:
    """Generates and caches RAG answers"""
    
    def __init__(self, rag):
        self.rag = rag
        self.query_handler = RAGQueryHandler(rag)
        self.data_manager = DataManager()
    
    def generate_answers(self, dataset):
        """Generate RAG answers with caching"""
        rag_answers = self.data_manager.load_rag_answers()
        processed_ids = {sample["id"] for sample in rag_answers}
        
        print(f"Found {len(processed_ids)} cached RAG answers")
        
        try:
            for qa in tqdm(dataset, desc="Generating RAG answers"):
                if qa["id"] in processed_ids:
                    continue
                
                try:
                    intent, contexts, answer = self.query_handler.query(qa["question"])
                    
                    if answer is None or not isinstance(contexts, list):
                        print(f"\nWarning: Invalid response for {qa['id']}, skipping...")
                        continue
                    
                    rag_answers.append({
                        "id": qa["id"],
                        "question": qa["question"],
                        "answer": str(answer),
                        "ground_truth": qa["answer"],
                        "contexts": contexts,
                    })
                    
                    self.data_manager.save_rag_answers(rag_answers)
                    time.sleep(EvalConfig.RAG_SLEEP_DELAY)
                    
                except Exception as e:
                    print(f"\nError processing question {qa['id']}: {type(e).__name__}: {e}")
                    self.data_manager.save_rag_answers(rag_answers)
                    time.sleep(2)
                    continue
        
        except KeyboardInterrupt:
            print("\nInterrupted by user. Saving progress...")
            self.data_manager.save_rag_answers(rag_answers)
            raise
        
        self.data_manager.save_rag_answers(rag_answers)
        return Dataset.from_list(rag_answers)


# =========================
# METRICS EVALUATOR
# =========================
class MetricsEvaluator:
    """Handles metrics evaluation with batching"""
    
    def __init__(self, llm, embeddings):
        self.llm = llm
        self.embeddings = embeddings
        self.metrics = [
            faithfulness,
            context_precision,
            context_recall,
            answer_correctness
        ]
    
    def evaluate_batch(self, batch, attempt=0):
        """Evaluate a single batch with retry logic"""
        max_retries = EvalConfig.EVAL_RETRY_ATTEMPTS
        
        try:
            batch_results = evaluate(
                batch, 
                metrics=self.metrics, 
                llm=self.llm, 
                embeddings=self.embeddings
            )
            return batch_results.to_pandas()
            
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 5 * (attempt + 1)
                print(f"Error in batch: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
                return self.evaluate_batch(batch, attempt + 1)
            else:
                print(f"Batch failed after {max_retries} attempts: {e}")
                return pd.DataFrame({
                    metric.name: [None] * len(batch) for metric in self.metrics
                })
    
    def evaluate_dataset(self, dataset, batch_size=None):
        """Evaluate entire dataset in batches"""
        if batch_size is None:
            batch_size = EvalConfig.EVAL_BATCH_SIZE
        
        all_results = []
        total_samples = len(dataset)
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        for i in range(0, total_samples, batch_size):
            batch_end = min(i + batch_size, total_samples)
            batch = dataset.select(range(i, batch_end))
            
            print(f"\nEvaluating batch {i//batch_size + 1}/{num_batches} ({batch_end - i} samples)")
            
            batch_results = self.evaluate_batch(batch)
            all_results.append(batch_results)
            
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
        """Save full results with metadata"""
        df["id"] = dataset["id"]
        df.to_csv(EvalConfig.FULL_RESULTS_FILE, index=False)
        print(f"\nFull results saved to {EvalConfig.FULL_RESULTS_FILE}")
    
    @staticmethod
    def generate_summary(df):
        """Generate and save summary statistics"""
        summary = pd.DataFrame({
            "overall": df.mean(numeric_only=True),
        })
        summary.to_csv(EvalConfig.SUMMARY_FILE)
        
        print("\n=== SUMMARY ===")
        print(summary)
        
        return summary


# =========================
# MAIN ORCHESTRATOR
# =========================
class EvaluationOrchestrator:
    """Orchestrates the entire evaluation process"""
    
    def __init__(self):
        EvalConfig.ensure_output_dir()
        self.data_manager = DataManager()
        self.api_clients = APIClients()
    
    def run(self):
        """Run the complete evaluation pipeline"""
        print(f"\n{'='*60}")
        print(f"RAG EVALUATION PIPELINE")
        print(f"{'='*60}")
        
        # Step 1: Load dataset
        print(f"\n[1/4] Loading dataset")
        dataset = self.data_manager.load_dataset(limit=EvalConfig.NUM_QUESTIONS)
        print(f"Dataset size: {len(dataset)} questions")
        
        # Step 2: Generate RAG answers
        print(f"\n[2/4] Generating/Loading RAG answers")
        eval_dataset = self._get_or_generate_answers(dataset)
        print(f"Total samples ready for evaluation: {len(eval_dataset)}")
        
        # Step 3: Run evaluation
        print(f"\n[3/4] Running RAGAS evaluation")
        df = self._get_or_evaluate(eval_dataset)
        
        # Step 4: Generate summary
        print(f"\n[4/4] Generating analysis")
        ResultsAnalyzer.generate_summary(df)
        
        self._print_summary()
    
    def _get_or_generate_answers(self, dataset):
        """Get cached answers or generate new ones"""
        if os.path.exists(EvalConfig.RAG_RESULTS_FILE):
            print(f"Loading cached RAG answers from {EvalConfig.RAG_RESULTS_FILE}")
            rag_data = self.data_manager.load_rag_answers()
            
            # Filter to match dataset IDs (in case we're limiting questions)
            dataset_ids = {qa["id"] for qa in dataset}
            rag_data = [r for r in rag_data if r["id"] in dataset_ids]
            
            if len(rag_data) < len(dataset):
                print(f"Need to generate {len(dataset) - len(rag_data)} more answers...")
                return self._generate_missing_answers(dataset)
            
            return Dataset.from_list(rag_data)
        else:
            print("No cached answers found. Generating from scratch...")
            return self._generate_missing_answers(dataset)
    
    def _generate_missing_answers(self, dataset):
        """Generate missing RAG answers"""
        rag = RAGInitializer.initialize()
        generator = RAGAnswerGenerator(rag)
        return generator.generate_answers(dataset)
    
    def _get_or_evaluate(self, eval_dataset):
        """Get cached evaluation or run new evaluation"""
        if self.data_manager.results_exist():
            print(f"\nEvaluation results already exist at {EvalConfig.FULL_RESULTS_FILE}")
            print("Delete this file if you want to re-run evaluation.")
            return pd.read_csv(EvalConfig.FULL_RESULTS_FILE)
        
        evaluator = MetricsEvaluator(
            self.api_clients.ragas_llm,
            self.api_clients.embeddings
        )
        df = evaluator.evaluate_dataset(eval_dataset)
        ResultsAnalyzer.save_results(df, eval_dataset)
        
        return df
    
    def _print_summary(self):
        """Print final summary"""
        print(f"\n{'='*60}")
        print(f"EVALUATION COMPLETE")
        print(f"{'='*60}")
        print(f"\nRAG answers cached at: {EvalConfig.RAG_RESULTS_FILE}")
        print(f"Evaluation results at: {EvalConfig.FULL_RESULTS_FILE}")
        print(f"Summary statistics at: {EvalConfig.SUMMARY_FILE}")
        print(f"\nTo re-run evaluation: delete the CSV files")
        print(f"To regenerate RAG answers: delete {EvalConfig.RAG_RESULTS_FILE}")
        print(f"To evaluate different number of questions: modify EvalConfig.NUM_QUESTIONS")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    orchestrator = EvaluationOrchestrator()
    orchestrator.run()