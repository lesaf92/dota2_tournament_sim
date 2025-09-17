# Dota 2 Tournament Simulator & Match Predictor

A full-stack web application that simulates an 8-team, double-elimination Dota 2 tournament. Match outcomes are predicted by a neural network trained on historical data using multiple advanced rating systems.

-----

### Project Overview

This project provides an end-to-end pipeline for predicting Dota 2 match outcomes. It includes scripts to:

1.  **Gather Data**: Fetch thousands of professional match results from the OpenDota API.
2.  **Calculate Historical Ratings**: Process matches chronologically to compute multiple historical skill ratings for each team.
3.  **Train a Model**: Train a PyTorch neural network on the generated dataset to predict win probabilities.
4.  **Simulate a Tournament**: A Flask-based web application provides a Liquidpedia-style bracket interface where users can select 8 teams and run a full tournament simulation based on the model's predictions.

-----

### Features

  - **Automated Data Pipeline**: A bash script automates the entire setup process.
  - **Historical Rating Engine**: Implements Elo (with variable K-factors) and a custom Glicko-2 engine from scratch.
  - **Neural Network Prediction**: Uses a PyTorch model to predict match outcomes based on rating differences.
  - **Interactive Web Interface**: A sleek, Liquipedia-inspired tournament bracket built with Flask, HTML, CSS, and JavaScript.
  - **Full Tournament Simulation**: Simulates a complete 8-team, double-elimination bracket, including upper and lower brackets, to determine a champion.

-----

### Understanding the Rating Systems

A key feature of this project is its use of robust rating systems to quantify team skill. Here’s how they work.

#### **The Elo Rating System**

The [**Elo system**](https://en.wikipedia.org/wiki/Elo_rating_system) is the foundation of many competitive rating systems, originally designed for chess. Its goal is to calculate the relative skill level of players in a zero-sum game.

  * **Core Concept**: Each team has a rating number. When two teams play, the winner takes points from the loser. The number of points exchanged depends on the difference in their ratings.
  * **Expected Outcome**: If a high-rated team beats a low-rated team, only a few points are exchanged, as this was the expected outcome. However, if the low-rated team causes an upset, it will gain a large number of points.
  * **The K-Factor**: The maximum number of points that can be exchanged is determined by a value called the **K-factor**.
      * A **low K-factor** (like `k=32`, used in this project) leads to smaller, more stable rating changes.
      * A **high K-factor** (like `k=64`) makes the ratings more volatile and responsive to recent results.
        This project calculates both `Elo32` and `Elo64` to feed the model a richer set of features.

#### **The Glicko-2 Rating System**

Developed by Professor Mark Glickman, [**Glicko-2**](https://en.wikipedia.org/wiki/Glicko_rating_system) is a significant improvement upon the Elo system because it introduces the concept of **rating uncertainty**. It acknowledges that we can be more or less confident in a team's rating.

Glicko-2 tracks three values for each team:

1.  **Rating (μ)**: This is the skill rating, similar to Elo. It's the system's best guess of a team's strength.
2.  **Rating Deviation (RD or φ)**: This is the measure of uncertainty. An RD is like a margin of error: a low RD means we are very confident in the team's rating (e.g., a veteran team that plays often), while a high RD means the rating is less reliable (e.g., a new team or a team that hasn't played in a long time). **A team with a high RD will see its rating change much more drastically after a match.**
3.  **Rating Volatility (σ)**: This measures the consistency of a team's performance over time. A team with surprisingly erratic results (e.g., beating strong teams but losing to weak ones) will have a high volatility, which causes their RD to increase more quickly.

In essence, Glicko-2 provides a much more nuanced view of skill by not only estimating a team's strength but also how *reliable* that estimation is.

-----

### Technology Stack

  - **Backend**: Python, Flask
  - **Machine Learning**: PyTorch
  - **Data Handling**: Pandas, NumPy
  - **Frontend**: HTML5, CSS3 (Flexbox), JavaScript (Fetch API)
  - **Orchestration**: Bash Script
  - **Data Source**: OpenDota API

-----

### Project Structure

```
/
├── app.py                   # Flask web server
├── data_generator.py        # Python script to fetch data and generate the dataset
├── train.py                 # Python script to train the PyTorch model
├── teams.py                 # Optional script to pre-fetch team data
├── run_project.sh           # Main bash script to orchestrate everything
|
├── dota2_predictor.pt       # (Generated) The trained model file
├── dota2_dataset.csv        # (Generated) The final dataset
├── teams.json               # (Generated) Team ID-to-name mapping
├── scaler.json              # (Generated) Scaler data for the model
|
├── templates/
│   └── index.html           # Frontend HTML
|
└── static/
    └── style.css            # Frontend CSS
```

-----

### Setup and Installation

1.  **Clone the Repository**

    ```bash
    git clone https://github.com/lesaf92/dota2_tournament_sim.git
    cd dota2_tournament_sim
    ```

2.  **Create a Virtual Environment**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install Dependencies**
    Install the packages using the `requirements.txt` file:

    ```bash
    pip install -r requirements.txt
    ```

-----

### Usage

The project is automated with a bash script. This is the recommended way to run the pipeline.

1.  **Make the script executable:**

    ```bash
    chmod +x run_project.sh
    ```

2.  **Run the orchestrator:**

    ```bash
    ./run_project.sh
    ```

    The script will automatically:

      - Run the data generator if `dota2_multi_rating_dataset.csv` is not found.
      - Run the training script if `dota2_predictor.pt` is not found.
      - Launch the Flask web application.

3.  **Access the Web App:** Open your browser and navigate to `http://127.0.0.1:5000`.

-----

### Acknowledgments

  - This project relies on the fantastic and free [OpenDota API](https://docs.opendota.com/).
  - The Glicko-2 rating system was designed by Professor [Mark E. Glickman](https://www.glicko.net/glicko.html).
