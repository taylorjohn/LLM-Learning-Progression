# CS336 Basics - Rust Example Structure

This repository provides a Rust project structure example that mirrors the organizational concepts of the Python-based Stanford CS336 Spring 2024 Assignment 1 ([stanford-cs336/spring2024-assignment1-basics](https://github.com/stanford-cs336/spring2024-assignment1-basics)).

**Note:** This project contains **only the structure** and placeholder code. It does **not** implement the actual machine learning algorithms or tasks from the CS336 assignment.

## Structure Overview

*   `Cargo.toml`: The manifest file for this Rust crate (package). It defines metadata, dependencies (runtime and development), similar to Python's `requirements.txt` + `setup.py`.
*   `src/`: Contains the library source code.
    *   `lib.rs`: The main library file. You would define public modules and functions here. Unit tests specific to modules can also reside within `src` files under `#[cfg(test)]`.
*   `tests/`: Contains integration tests. These tests use the library as an external crate would.
*   `data/`: (User-created) Intended location for datasets, similar to the Python assignment setup.

## Building and Testing

Ensure you have the Rust toolchain (including `cargo`) installed ([https://www.rust-lang.org/tools/install](https://www.rust-lang.org/tools/install)).

Navigate to the `cs336_basics_rs` directory in your terminal.

*   **Build the library:**
    ```bash
    cargo build
    ```
*   **Run all tests (unit and integration):**
    ```bash
    cargo test
    ```

## Data

Create the `data` directory manually if needed:

```bash
mkdir data
```

Refer to the original CS336 assignment instructions for details on which datasets to download (e.g., TinyStories, OpenWebText sample) and place them here. 