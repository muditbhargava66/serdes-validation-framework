# Installation Guide

Follow these steps to install and set up the SerDes Validation Framework on your local machine.

## Prerequisites

- Python 3.7 or higher
- Git

## Installation Steps

1. **Clone the repository:**

    ```bash
    git clone https://github.com/muditbhargava66/serdes-validation-framework.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd serdes-validation-framework
    ```

3. **Create and activate a virtual environment (optional but recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows, use `venv\Scripts\activate`
    ```

4. **Install the dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

5. **Run the tests to ensure everything is set up correctly:**

    ```bash
    python -m unittest discover -s tests
    ```

## Usage

Refer to the [usage documentation](USAGE.md) for examples and instructions on how to use the framework.

---