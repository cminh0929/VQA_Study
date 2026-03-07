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
main.CONFIG['batch_size'] = 32
main.CONFIG['save_every'] = 10

def run_all():
    print("=" * 60)
    print("AUTOMATED VQA EXPERIMENT RUNNER")
    print("=" * 60)
    print(f"Dataset : {main.CONFIG['dataset']}")
    print(f"Epochs  : {main.CONFIG['epochs']}")
    print(f"Device  : {main.CONFIG['device']}")
    print("=" * 60)

    # 1. Build vocab
    print("\n[Step 1] Building vocabularies...")
    main.cmd_build_vocab()

    # 2. Train and evaluate each model
    for model_id in range(1, 5):
        print(f"\n{'='*60}")
        print(f"[Step 2] Model {model_id} / 4")
        print("=" * 60)

        main.CONFIG['model_id'] = model_id

        try:
            main.cmd_train()
        except Exception as e:
            print(f"Error training Model {model_id}: {e}")
            continue

        try:
            main.cmd_evaluate()
        except Exception as e:
            print(f"Error evaluating Model {model_id}: {e}")
            continue

    # 3. Compare all models
    print(f"\n{'='*60}")
    print("[Step 3] Final comparison")
    print("=" * 60)
    main.cmd_compare()

    # 4. Generate visualizations
    print(f"\n{'='*60}")
    print("[Step 4] Generating visualizations")
    print("=" * 60)
    subprocess.run(["py", "-3.10", "visualize_predictions.py"])

    print("\nAll experiments finished.")
    print("Visualizations : results/visualizations/")
    print("Predictions    : results/")

if __name__ == "__main__":
    run_all()
