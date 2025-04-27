import wandb
import os

# --- Configuration ---
PROJECT_NAME = "cs336_assignment1"
RUN_NAME = "manual_log_5000iters"
ENTITY = "TaylorJohn" # Replace with your entity if different
CHECKPOINT_PATH = "spring2024-assignment1-basics/checkpoints/tinystories_test_5000iters/final_checkpoint.pt"
ARTIFACT_NAME = "tinystories_5000iters_final"

# Final metrics from the 5000-iteration run
FINAL_VAL_LOSS = 1.6469
FINAL_VAL_PERPLEXITY = 5.19

def main():
    print("Initializing W&B Run...")
    try:
        run = wandb.init(
            project=PROJECT_NAME,
            name=RUN_NAME,
            entity=ENTITY,
            job_type="manual_log"
        )
        print(f"W&B Run initialized: {run.url}")

        # Log summary metrics
        print("Logging final metrics...")
        run.summary["final_val_loss"] = FINAL_VAL_LOSS
        run.summary["final_val_perplexity"] = FINAL_VAL_PERPLEXITY
        print(f"  Logged final_val_loss: {FINAL_VAL_LOSS}")
        print(f"  Logged final_val_perplexity: {FINAL_VAL_PERPLEXITY}")

        # Log checkpoint as an artifact
        if os.path.exists(CHECKPOINT_PATH):
            print(f"Creating and logging artifact '{ARTIFACT_NAME}'...")
            artifact = wandb.Artifact(ARTIFACT_NAME, type='model')
            artifact.add_file(CHECKPOINT_PATH)
            run.log_artifact(artifact)
            print(f"  Logged checkpoint: {CHECKPOINT_PATH}")
        else:
            print(f"Warning: Checkpoint file not found at {CHECKPOINT_PATH}. Skipping artifact logging.")

        # Finish the run
        run.finish()
        print("W&B Run finished.")

    except Exception as e:
        print(f"An error occurred during W&B logging: {e}")

if __name__ == "__main__":
    main() 