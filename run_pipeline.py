#!/usr/bin/env python
"""
Complete Pipeline Runner

This script runs the entire data preparation, analysis, and modeling pipeline
in one command. Useful for end-to-end execution.

Usage:
    python run_pipeline.py

Logs are saved to: logs/pipeline_YYYY-MM-DD_HH-MM-SS.log
"""

import subprocess
import sys
import logging
from pathlib import Path
from datetime import datetime


def setup_logging():
    """Set up logging to both console and file."""
    logs_dir = Path(__file__).parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    
    # Create log filename with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"pipeline_{timestamp}.log"
    
    # Create logger
    logger = logging.getLogger("pipeline")
    logger.setLevel(logging.DEBUG)
    
    # File handler (detailed logs)
    file_handler = logging.FileHandler(log_file, mode='w')
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    
    # Console handler (user-friendly) with UTF-8 encoding for emoji support
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setStream(sys.stdout)
    # Force UTF-8 encoding to support emojis on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    console_formatter = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger, log_file


def run_script(script_path, description, logger):
    """Run a Python script and handle errors."""
    logger.info(f"\n{'='*80}")
    logger.info(f"  {description}")
    logger.info(f"{'='*80}\n")
    
    start_time = datetime.now()
    logger.debug(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=Path(__file__).parent
    )
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    logger.debug(f"Elapsed time: {elapsed_time:.2f} seconds")
    
    if result.returncode != 0:
        logger.error(f"\n[ERROR] Error running {script_path}")
        logger.error("Pipeline stopped. Fix the error and try again.")
        logger.error(f"Exit code: {result.returncode}")
        sys.exit(1)
    
    logger.info(f"[SUCCESS] {description} completed successfully\n")


def main():
    """Run the complete pipeline."""
    logger, log_file = setup_logging()
    root_dir = Path(__file__).parent
    
    logger.info("\n")
    logger.info("CUSTOMER CHURN PREDICTION & CLV ANALYSIS - COMPLETE PIPELINE")
    logger.info("="*80 + "\n")
    logger.info(f"Pipeline started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Logs will be saved to: {log_file}\n")
    
    # Step 1: Data Preparation
    run_script(
        root_dir / "src" / "data_prep.py",
        "Step 1/5: DATA PREPARATION (load, clean, engineer features, split)",
        logger
    )
    
    # Step 2: CLV Analysis
    run_script(
        root_dir / "src" / "clv_analysis.py",
        "Step 2/5: CLV ANALYSIS (compute CLV, segment, insights)",
        logger
    )
    
    # Step 3: Model Training
    run_script(
        root_dir / "src" / "train_models.py",
        "Step 3/5: MODEL TRAINING (train 3 models, evaluate)",
        logger
    )
    
    # Step 4: Interpretability
    run_script(
        root_dir / "src" / "interpretability.py",
        "Step 4/5: INTERPRETABILITY (SHAP & feature importance)",
        logger
    )
    
    # Summary
    logger.info("\n" + "="*80)
    logger.info("PIPELINE COMPLETE!")
    logger.info("="*80)
    logger.info(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"\nAll outputs saved:")
    logger.info(f"  • Data: data/processed/{{train,val,test}}.csv")
    logger.info(f"  • Models: models/{{logistic_regression,random_forest,xgboost}}.pkl")
    logger.info(f"  • Results: models/test_results.csv")
    logger.info(f"  • Plots: figures/")
    logger.info(f"  • Importance: models/*_importance.csv")
    logger.info(f"  • Logs: {log_file}\n")
    logger.info("Next steps:")
    logger.info("  1. Review model performance: models/test_results.csv")
    logger.info("  2. Check business insights: Look at figures/churn_by_clv.png")
    logger.info("  3. Launch the app: streamlit run app.py")
    logger.info("  4. Deploy: Push to GitHub, connect to Streamlit Cloud\n")
    logger.info("For more details, see README.md")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        sys.exit(1)
