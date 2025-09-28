# Real-Estate Intent Recognition

A machine learning project to classify user text queries (e.g., "looking to buy a house") into predefined real estate intents. The training data is **synthetically generated using Google's Gemini 2.5 Pro**. This repository includes scripts for training, evaluating, and serving the model via a simple API.

-----

## Quick Start

1.  **Clone the repository and install dependencies:**

    ```sh
    git clone https://github.com/Erenuo/Real-Estate-Intent-Recognition.git
    cd Real-Estate-Intent-Recognition
    pip install -r requirements.txt
    ```

2.  **Run the API:**

    ```sh
    python prediction_api.py
    ```

    The API will run on `http://127.0.0.1:5000`. You can test it using `client.html`.

-----

## Usage

  - **Predict from command line**:
    ```sh
    python predict_intent.py "your text query here"
    ```

-----

