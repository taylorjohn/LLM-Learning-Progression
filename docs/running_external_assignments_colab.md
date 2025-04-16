# Running External Assignments (e.g., CS336) on Google Colab

This guide provides general steps for setting up and running computationally intensive assignments, like those from Stanford's CS336 course, using Google Colab, especially when aiming to reproduce or participate in benchmarks like leaderboards.

**Disclaimer:** Specific steps might vary based on the exact assignment structure and requirements provided by the course instructors. Always refer to the official assignment handout first.

## 1. Obtain the Assignment Code

*   **Leaderboard != Code:** Repositories like the `assignment1-basics-leaderboard` only track results. You **must** obtain the actual starter code for the assignment from the official course platform (e.g., course website, Canvas, Ed).

## 2. Set Up Google Colab

*   **New Notebook:** Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
*   **GPU Runtime:**
    *   Navigate to `Runtime` -> `Change runtime type`.
    *   Select `Python 3`.
    *   Choose a `GPU` accelerator.
        *   **Free Tier:** Often provides **NVIDIA T4**. Good for initial development and debugging, but significantly slower than benchmark GPUs.
        *   **Paid Tiers (Colab Pro/Pro+):** May offer access to **NVIDIA A100** or V100. The A100 is the closest available Colab GPU to an H100, but still considerably slower. Access is not guaranteed and depends on availability.
        *   **Benchmark Reference (H100):** Note that leaderboard times (like the 1.5-hour limit mentioned for CS336 Assignment 1) are often based on high-end data center GPUs like the NVIDIA H100, which are much faster than anything typically available on Colab.

## 3. Mount Google Drive (Recommended)

*   **Persistence:** Mount your Google Drive to save code, datasets, model checkpoints, and outputs reliably across sessions.
*   **Command:**
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
*   **Organization:** Create a dedicated folder in your Drive for the assignment (e.g., `/content/drive/MyDrive/cs336_assignment1`).

## 4. Get Code into Colab

*   **Method A: Git Clone (Recommended if available):**
    ```bash
    # Navigate to your chosen directory (e.g., in Drive)
    %cd /content/drive/MyDrive/cs336_assignment1

    # Clone the assignment's starter code repository (replace with actual URL)
    !git clone <URL_TO_ASSIGNMENT_STARTER_CODE_REPO>

    # Change into the cloned directory
    %cd <ASSIGNMENT_REPO_DIRECTORY_NAME>
    ```
*   **Method B: Upload & Unzip:**
    ```bash
    # Navigate to your chosen directory (e.g., in Drive)
    %cd /content/drive/MyDrive/cs336_assignment1

    # Upload the assignment's zip file via Colab's file browser (left panel)
    # Or upload to Drive manually first

    # Unzip the file (replace with actual path and name)
    !unzip /content/drive/MyDrive/path/to/your/assignment_code.zip

    # Change into the unzipped directory
    %cd <ASSIGNMENT_DIRECTORY_NAME>
    ```

## 5. Access the Dataset

*   **Location:** Download the required dataset (e.g., the specified OpenWebText subsample) as per course instructions.
*   **Accessibility:**
    *   Upload it to your Google Drive and reference the path (e.g., `/content/drive/MyDrive/data/openwebtext`).
    *   Or, download directly within Colab if a URL is provided (using `!wget <url>`). Note that files downloaded directly to the Colab instance (not Drive) are temporary.
*   **Configuration:** Ensure your training script is configured to use the correct dataset path.

## 6. Install Dependencies

*   **`requirements.txt`:** Most Python projects include this file.
*   **Command:** Run this from the root directory of the assignment code:
    ```bash
    !pip install -r requirements.txt
    ```
    *   You might also need to install `wandb` separately if not included: `!pip install wandb`

## 7. Run Training

*   **Script & Arguments:** Follow the assignment instructions to execute the main training script. This usually involves specifying configuration files, hyperparameters, data paths, output directories, etc.
    ```bash
    # Example command (adapt based on actual assignment script)
    !python train.py --config configs/my_config.yaml --dataset_path /path/to/dataset --output_dir /path/to/outputs --use_wandb True
    ```
*   **Weights & Biases (for Leaderboards):**
    *   Log in: Run `!wandb login` in a cell and provide your API key when prompted.
    *   Ensure the training script is configured to log metrics to `wandb`, especially validation loss and wallclock time.

## 8. Monitor and Evaluate

*   **Colab Output:** Check the cell output for progress, errors, and metrics.
*   **W&B Dashboard:** Monitor training curves, resource usage, and hyperparameters live at [wandb.ai](https://wandb.ai).
*   **Validation Loss:** Track the primary metric for leaderboard comparison.
*   **Wallclock Time:** Be mindful of runtime, especially relative to any benchmark limits (remembering the Colab GPU vs. H100 difference).

## 9. Submitting to a Leaderboard (If Applicable)

*   **Follow Instructions:** Adhere strictly to the submission guidelines provided in the leaderboard repository (like the CS336 one).
*   **Required Information:** Typically includes final validation loss, a link to a learning curve plot (e.g., from W&B) showing wallclock time, and a description of your approach.
*   **Pull Request:** Submit your results via a Pull Request to the leaderboard repository. 