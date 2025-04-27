#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
import mlx.core as mx

# Attempt to import the convert function. Handle potential import errors.
# The exact location might vary slightly depending on mlx-lm version structure.
try:
    from mlx_lm.utils import convert
except ImportError:
    try:
        # Older versions might have it elsewhere
        from mlx_lm.convert import convert
    except ImportError:
        print("Error: Could not import the 'convert' function from 'mlx_lm'.")
        print("Please ensure 'mlx-lm' is installed correctly ('pip install mlx-lm').")
        exit(1)

def main(args):
    """Main function to perform model conversion."""
    hf_path = args.hf_path
    mlx_path = Path(args.mlx_path)

    print(f"Starting conversion of Hugging Face model '{hf_path}'...")
    print(f"Quantizing to 4-bit MLX format.")
    print(f"Output directory: {mlx_path}")

    # Ensure output directory exists
    mlx_path.mkdir(parents=True, exist_ok=True)

    try:
        # Perform the conversion with 4-bit quantization enabled
        # Common parameters for 4-bit quantization in mlx-lm:
        # quantize=True
        # q_bits=4
        # q_group_size=64 (common default, adjust if needed)
        convert(
            hf_path=hf_path,
            mlx_path=str(mlx_path),
            quantize=True,
            q_bits=4,
            q_group_size=64 # Explicitly set common group size
            # upload_repo=None # Set this if you want to upload to HF Hub
        )
        print("\nConversion successful!")
        print(f"Quantized MLX model saved to: {mlx_path}")

    except Exception as e:
        print(f"\nError during conversion: {e}")
        print("Please check the model path and ensure necessary dependencies are installed.")
        # Consider adding more specific error handling if needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert a Hugging Face model to 4-bit quantized MLX format."
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        required=True,
        help="Path or identifier of the Hugging Face model (e.g., 'microsoft/phi-2')."
    )
    parser.add_argument(
        "--mlx-path",
        type=str,
        required=True,
        help="Path to save the converted MLX model directory."
    )
    # Add quantization arguments if needed, but for this demo, we hardcode 4-bit
    # parser.add_argument(
    #     "-q", "--quantize", action="store_true", help="Generate a quantized model."
    # )
    # parser.add_argument(
    #     "--q-bits", type=int, default=4, help="Bits for quantization."
    # )
    # parser.add_argument(
    #     "--q-group-size", type=int, default=64, help="Group size for quantization."
    # )

    args = parser.parse_args()
    main(args) 