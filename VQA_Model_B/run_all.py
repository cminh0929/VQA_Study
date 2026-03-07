import os
import subprocess
import logging

try:
    import main
except ImportError:
    print("Error: Could not import main.py. Make sure you are running this from the VQA_Model_B directory.")
    exit(1)

# Set desired configuration
main.CONFIG['dataset'] = 'vqa2_full'
main.CONFIG['epochs'] = 10
main.CONFIG['batch_size'] = 32  # Keep batch size reasonable, adjust if OOM
main.CONFIG['save_every'] = 10  # Only save at the very end or adjust as needed

def run_all():
    print("=" * 60)
    print("🚀 AUTOMATED VQA EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Dataset:   {main.CONFIG['dataset']}")
    print(f"Epochs:    {main.CONFIG['epochs']}")
    print(f"Device:    {main.CONFIG['device']}")
    print("=" * 60)
    
    # 1. Always build vocab first to be safe
    print("\n[STEP 1] Building Vocabularies...")
    main.cmd_build_vocab()
    
    # 2. Train and Evaluate each model
    for model_id in range(1, 5):
        print(f"\n" + "=" * 60)
        print(f"[STEP 2] RUNNING EXPERIMENT FOR MODEL {model_id} / 4")
        print("=" * 60)
        
        main.CONFIG['model_id'] = model_id
        
        # Train
        try:
            main.cmd_train()
        except Exception as e:
            print(f"❌ Error training Model {model_id}: {e}")
            continue
            
        # Evaluate
        try:
            main.cmd_evaluate()
        except Exception as e:
            print(f"❌ Error evaluating Model {model_id}: {e}")
            continue

    # 3. Compare all models
    print("\n" + "=" * 60)
    print("[STEP 3] FINAL COMPARISON")
    print("=" * 60)
    main.cmd_compare()
    
    # 4. Generate Visualizations
    print("\n" + "=" * 60)
    print("[STEP 4] GENERATING VISUALIZATIONS")
    print("=" * 60)
    subprocess.run(["py", "-3.10", "visualize_predictions.py"])
    
    print("\n🎉 ALL EXPERIMENTS FINISHED SUCCESSFULLY! 🎉")
    print("Visualizations saved in: results/visualizations/")
    print("Predictions saved in:    results/")

if __name__ == "__main__":
    run_all()
