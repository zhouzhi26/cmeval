#!/usr/bin/env python3
"""
Batch Model Evaluation Script
Automatically evaluates multiple models by calling chemical_knowledge_test.py
"""

import os
import subprocess
import time
import json
from pathlib import Path
from datetime import datetime
import argparse


class BatchModelEvaluator:
    def __init__(self):
        self.models = [
            "/root/autodl-tmp/models/Qwen/Qwen3-0___6B",
            "/root/autodl-tmp/models/Qwen/Qwen3-1___7B",
            "/root/autodl-tmp/models/Qwen/Qwen3-4B",
            "/root/autodl-tmp/models/Qwen/Qwen3-8B"
        ]
        
        self.results_summary = []
        self.start_time = None
        
    def check_model_exists(self, model_path):
        """Check if model exists"""
        if not os.path.exists(model_path):
            return False
        
        # Check if there are model files
        model_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors') or f.endswith('.bin')]
        return len(model_files) > 0
    
    def get_model_name(self, model_path):
        """Extract model name from model path"""
        return os.path.basename(model_path)
    
    def run_single_evaluation(self, model_path, dataset_path, output_dir="results"):
        """Run evaluation for a single model"""
        model_name = self.get_model_name(model_path)
        print(f"\n{'='*60}")
        print(f"🚀 Starting evaluation for: {model_name}")
        print(f"📁 Model path: {model_path}")
        print(f"📊 Dataset: {os.path.basename(dataset_path)}")
        print(f"{'='*60}")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
        output_file = f"{output_dir}/{model_name}_{dataset_name}.json"
        
        # Build command
        cmd = [
            "python", "chemical_knowledge_test.py",
            "--model-path", model_path,
            "--dataset-path", dataset_path,
            "--output", output_file
        ]
        
        try:
            start_time = time.time()
            
            # Run evaluation
            print(f"⏱️  Running evaluation...")
            print(f"📝 Command: {' '.join(cmd)}")
            
            # Use real-time output instead of capture_output
            result = subprocess.run(
                cmd, 
                timeout=3600  # 1 hour timeout
            )
            
            end_time = time.time()
            duration = end_time - start_time
            
            if result.returncode == 0:
                print(f"✅ Evaluation completed successfully!")
                print(f"⏱️  Duration: {duration:.2f} seconds ({duration/60:.2f} minutes)")
                
                # Try to read result file to get metrics
                metrics = self.extract_metrics(output_file)
                
                # Record results
                evaluation_result = {
                    "model_name": model_name,
                    "model_path": model_path,
                    "dataset": dataset_name,
                    "output_file": output_file,
                    "status": "success",
                    "duration_seconds": duration,
                    "duration_minutes": duration / 60,
                    "metrics": metrics,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.results_summary.append(evaluation_result)
                
                # Print metrics
                if metrics:
                    print(f"📊 Results:")
                    print(f"   Accuracy: {metrics.get('accuracy', 'N/A'):.4f}")
                    print(f"   Precision: {metrics.get('precision', 'N/A'):.4f}")
                    print(f"   Recall: {metrics.get('recall', 'N/A'):.4f}")
                    print(f"   F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
                
                return True
                
            else:
                print(f"❌ Evaluation failed!")
                print(f"Return code: {result.returncode}")
                
                evaluation_result = {
                    "model_name": model_name,
                    "model_path": model_path,
                    "dataset": dataset_name,
                    "output_file": output_file,
                    "status": "failed",
                    "error": f"Return code: {result.returncode}",
                    "duration_seconds": duration,
                    "timestamp": datetime.now().isoformat()
                }
                
                self.results_summary.append(evaluation_result)
                return False
                
        except subprocess.TimeoutExpired:
            print(f"⏰ Evaluation timed out after 1 hour")
            
            evaluation_result = {
                "model_name": model_name,
                "model_path": model_path,
                "dataset": dataset_name,
                "output_file": output_file,
                "status": "timeout",
                "duration_seconds": 3600,
                "timestamp": datetime.now().isoformat()
            }
            
            self.results_summary.append(evaluation_result)
            return False
            
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            
            evaluation_result = {
                "model_name": model_name,
                "model_path": model_path,
                "dataset": dataset_name,
                "output_file": output_file,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
            
            self.results_summary.append(evaluation_result)
            return False
    
    def extract_metrics(self, result_file):
        """Extract metrics from result file"""
        try:
            if os.path.exists(result_file):
                with open(result_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if 'metrics' in data:
                    return data['metrics']
        except Exception as e:
            print(f"Warning: Could not extract metrics from {result_file}: {e}")
        
        return None
    
    def run_batch_evaluation(self, dataset_path, output_dir="results", skip_existing=True):
        """Run batch evaluation"""
        print("🚀 Starting batch model evaluation...")
        print(f"📊 Dataset: {dataset_path}")
        print(f"📁 Output directory: {output_dir}")
        print(f"🔢 Total models: {len(self.models)}")
        
        self.start_time = time.time()
        
        # Check dataset file
        if not os.path.exists(dataset_path):
            print(f"❌ Dataset file not found: {dataset_path}")
            return False
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Statistics
        total_models = len(self.models)
        successful_models = 0
        failed_models = 0
        
        # Display model list
        print(f"\n📋 Models to evaluate:")
        for i, model_path in enumerate(self.models, 1):
            model_name = os.path.basename(model_path)
            status = "✅" if self.check_model_exists(model_path) else "❌"
            print(f"   {i:2d}. {status} {model_name}")
        
        print(f"\n{'='*60}")
        print(f"🎯 Starting evaluation process...")
        print(f"{'='*60}")
        
        # Evaluate models one by one
        for i, model_path in enumerate(self.models, 1):
            print(f"\n{'='*60}")
            print(f"📋 Progress: {i}/{total_models} ({i/total_models*100:.1f}%)")
            print(f"⏰ Elapsed time: {(time.time() - self.start_time)/60:.1f} minutes")
            print(f"✅ Completed: {successful_models}, ❌ Failed: {failed_models}")
            
            # Estimate remaining time
            if i > 1:
                avg_time_per_model = (time.time() - self.start_time) / (i - 1)
                remaining_models = total_models - i
                estimated_remaining = avg_time_per_model * remaining_models
                print(f"⏱️  Estimated remaining time: {estimated_remaining/60:.1f} minutes")
            
            print(f"{'='*60}")
            
            # Check if model exists
            if not self.check_model_exists(model_path):
                print(f"⚠️  Model not found: {model_path}")
                failed_models += 1
                continue
            
            # Check if should skip existing results
            if skip_existing:
                model_name = self.get_model_name(model_path)
                dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
                expected_output = f"{output_dir}/{model_name}_{dataset_name}.json"
                
                if os.path.exists(expected_output):
                    print(f"⏭️  Skipping {model_name} - result file already exists")
                    successful_models += 1
                    continue
            
            # Run evaluation
            success = self.run_single_evaluation(model_path, dataset_path, output_dir)
            
            if success:
                successful_models += 1
            else:
                failed_models += 1
            
            # Add delay to avoid resource conflicts
            if i < total_models:
                print(f"⏳ Waiting 10 seconds before next evaluation...")
                time.sleep(10)
        
        # Generate summary report
        self.generate_summary_report(output_dir, total_models, successful_models, failed_models)
        
        return True
    
    def generate_summary_report(self, output_dir, total_models, successful_models, failed_models):
        """Generate summary report"""
        end_time = time.time()
        total_duration = end_time - self.start_time
        
        summary = {
            "batch_evaluation_summary": {
                "total_models": total_models,
                "successful_models": successful_models,
                "failed_models": failed_models,
                "success_rate": successful_models / total_models if total_models > 0 else 0,
                "total_duration_seconds": total_duration,
                "total_duration_minutes": total_duration / 60,
                "start_time": datetime.fromtimestamp(self.start_time).isoformat(),
                "end_time": datetime.fromtimestamp(end_time).isoformat()
            },
            "individual_results": self.results_summary
        }
        
        # Save summary report
        summary_file = f"{output_dir}/batch_evaluation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"🎉 Batch Evaluation Completed!")
        print(f"{'='*60}")
        print(f"📊 Summary:")
        print(f"   Total models: {total_models}")
        print(f"   Successful: {successful_models}")
        print(f"   Failed: {failed_models}")
        print(f"   Success rate: {successful_models/total_models*100:.1f}%")
        print(f"   Total duration: {total_duration/60:.1f} minutes")
        print(f"   Summary file: {summary_file}")
        print(f"{'='*60}")
        
        # Print failed models
        if failed_models > 0:
            print(f"\n❌ Failed models:")
            for result in self.results_summary:
                if result['status'] != 'success':
                    print(f"   - {result['model_name']}: {result['status']}")
        
        # Print successful models
        if successful_models > 0:
            print(f"\n✅ Successful models:")
            for result in self.results_summary:
                if result['status'] == 'success':
                    print(f"   - {result['model_name']}")
                    if result.get('metrics'):
                        metrics = result['metrics']
                        print(f"     F1: {metrics.get('f1_score', 'N/A'):.4f}, "
                              f"Acc: {metrics.get('accuracy', 'N/A'):.4f}")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Batch Model Evaluation Tool")
    parser.add_argument("--dataset", 
                       default="/root/autodl-tmp/output.json",
                       help="Evaluation dataset path")
    parser.add_argument("--output-dir", 
                       default="results",
                       help="Output directory")
    parser.add_argument("--no-skip", 
                       action="store_true",
                       help="Do not skip existing result files")
    
    args = parser.parse_args()
    
    print("🚀 Batch Model Evaluation Tool")
    print("="*50)
    
    # Create evaluator
    evaluator = BatchModelEvaluator()
    
    # Run batch evaluation
    success = evaluator.run_batch_evaluation(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        skip_existing=not args.no_skip
    )
    
    if success:
        print("\n✅ Batch evaluation completed!")
        return 0
    else:
        print("\n❌ Batch evaluation failed!")
        return 1


if __name__ == "__main__":
    exit(main())
