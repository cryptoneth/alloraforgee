import logging
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
import yaml
import joblib
from datetime import datetime, timedelta
import dill  # Added for pickle predict_fn
import asyncio  # Added for async worker

# Added for Allora integration
from allora_sdk.worker import AlloraWorker

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__)))

# Import project modules
from utils.helpers import (
    load_config, setup_logging, set_random_seeds, create_directories,
    validate_timezone_utc, check_data_leakage, quality_gate_check,
    save_intermediate_data, memory_usage_check
)
from data.acquisition import DataAcquisition
from data.denoising import DataDenoiser
from features.engineering import FeatureEngineer
from models.lightgbm_model import LightGBMForecaster
from models.tft_model import TFTModel
from models.nbeats_model import NBeatsWrapper
from models.tcn_model import TCNWrapper
from models.ensemble import StackingEnsemble
from evaluation.validation import WalkForwardValidator
from evaluation.metrics import MetricsCalculator
from evaluation.backtesting import EconomicBacktester
from evaluation.statistical_tests import StatisticalTester
from explainability.interpretability import ModelInterpreter

logger = logging.getLogger(__name__)


class ETHForecastingPipeline:
    """
    Complete ETH forecasting pipeline implementation.
    
    This class orchestrates the entire forecasting workflow,
    ensuring data integrity and following all project rules.
    Updated for Allora SDK latest, topic 68, and custom wallet.
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the forecasting pipeline.
        
        Args:
            config_path: Path to configuration file (add 'allora' section with api_key, mnemonic, topic_id)
        """
        self.config_path = config_path
        self.config = None
        self.data = None
        self.features = None
        self.models = {}
        self.ensemble = None
        self.results = {}
        
        # Pipeline state tracking
        self.pipeline_state = {
            'environment_setup': False,
            'data_pipeline': False,
            'baseline_model': False,
            'advanced_models': False,
            'ensemble_creation': False,
            'evaluation_analysis': False,
            'deployment': False  # Added for Allora worker
        }
        
        logger.info("ETH Forecasting Pipeline initialized with Allora integration")
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """
        Run the complete forecasting pipeline, including deployment to Allora topic 68.
        
        Returns:
            Dictionary containing all results and metrics
        """
        try:
            logger.info("=" * 80)
            logger.info("STARTING ETH FORECASTING PIPELINE WITH ALLORA DEPLOYMENT")
            logger.info("=" * 80)
            
            start_time = time.time()
            
            # Phase 1: Environment Setup
            self._phase_1_environment_setup()
            
            # Phase 2: Data Pipeline
            self._phase_2_data_pipeline()
            
            # Phase 3: Baseline Model
            self._phase_3_baseline_model()
            
            # Phase 4: Advanced Models
            self._phase_4_advanced_models()
            
            # Phase 5: Ensemble Creation
            self._phase_5_ensemble_creation()
            
            # Phase 6: Evaluation and Analysis
            self._phase_6_evaluation_analysis()
            
            # Phase 7: Deployment to Allora (new)
            self._phase_7_deployment()
            
            # Final validation
            self._final_validation()
            
            total_time = time.time() - start_time
            logger.info(f"Pipeline completed successfully in {total_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            self._handle_pipeline_failure(e)
            raise
    
    # ... (Phases 1 to 6 unchanged, just copy from original) ...

    def _phase_7_deployment(self) -> None:
        """
        Phase 7: Deployment to Allora Network (topic 68)
        
        Builds predict_fn from ensemble, pickles it, and runs worker with custom wallet.
        Uses latest Allora SDK features (nonce management, etc.).
        """
        logger.info("Phase 7: Deployment to Allora Topic 68")
        logger.info("-" * 40)
        
        try:
            if self.ensemble is None:
                raise ValueError("No ensemble available for deployment")
            
            allora_config = self.config.get('allora', {})
            api_key = allora_config.get('api_key', '<your API key>')  # From developer.allora.network
            mnemonic = allora_config.get('mnemonic', '')  # Your wallet mnemonic; blank for new
            topic_id = allora_config.get('topic_id', 68)  # Updated to 68
            
            # Define predict function for live inference (adapt to your DataAcquisition for live features)
            def predict():
                # Fetch live features (assume DataAcquisition has get_live_features; or integrate AlloraMLWorkflow)
                live_data_acq = DataAcquisition(config=self.config)
                live_data = live_data_acq.get_live_data()  # Implement this in acquisition.py if needed
                live_features = self.feature_engineer.engineer_features(live_data)  # Reuse engineer
                
                feature_cols = [col for col in live_features.columns if col not in ['timestamp', 'ETH_log_return_24h', 'ETH_direction']]
                X_live = live_features[feature_cols]
                
                # Predict with ensemble
                preds = self.ensemble.predict(X_live)
                return pd.Series(preds, index=live_features.index)
            
            # Pickle the predict function
            predict_pkl_path = os.path.join(self.config['paths']['models'], 'predict.pkl')
            with open(predict_pkl_path, "wb") as f:
                dill.dump(predict, f)
            logger.info(f"Predict function pickled to {predict_pkl_path}")
            
            # Define my_model for worker (loads pickle and runs)
            def my_model():
                tic = time.time()
                with open(predict_pkl_path, "rb") as f:
                    predict_fn = dill.load(f)
                prediction = predict_fn()
                toc = time.time()
                print(f"Predict time: {toc - tic}, Prediction: {prediction}")
                return prediction
            
            # Run worker async (with mnemonic for custom wallet)
            async def run_worker():
                worker = AlloraWorker(
                    predict_fn=my_model,
                    api_key=api_key,
                    topic_id=topic_id,  # 68
                    mnemonic=mnemonic if mnemonic else None  # If blank, creates new and saves to .allora_key
                )
                async for result in worker.run():
                    if isinstance(result, Exception):
                        print(f"Error: {str(result)}")
                    else:
                        print(f"Prediction submitted to Allora: {result.prediction}")
            
            # Run the worker
            logger.info(f"Deploying worker to topic {topic_id} with wallet mnemonic: {'provided' if mnemonic else 'new generated'}")
            asyncio.run(run_worker())
            
            self.pipeline_state['deployment'] = True
            logger.info("✓ Phase 7 completed successfully")
            
        except Exception as e:
            logger.error(f"Phase 7 failed: {str(e)}")
            raise

    # ... (Rest unchanged: _final_validation, _handle_pipeline_failure, etc.) ...

def main():
    """
    Main entry point for the ETH forecasting pipeline with Allora deployment.
    """
    try:
        # Initialize pipeline
        pipeline = ETHForecastingPipeline()
        
        # Run complete pipeline
        results = pipeline.run_complete_pipeline()
        
        print("\n" + "=" * 80)
        print("ETH FORECASTING PIPELINE COMPLETED WITH ALLORA DEPLOYMENT")
        print("=" * 80)
        print(f"Models trained: {len([m for m in pipeline.models.values() if m is not None])}")
        print(f"Ensemble created: {pipeline.ensemble is not None}")
        print(f"Results available in: {pipeline.config['paths']['reports']}")
        print("=" * 80)
        
        return results
        
    except Exception as e:
        print(f"\nPipeline failed with error: {str(e)}")
        print("Check logs for detailed error information")
        return None


if __name__ == "__main__":
    main()
