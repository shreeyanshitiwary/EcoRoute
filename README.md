**🔍 Overview**
EcoRoute is an AI/ML-based project that combines synthetic data generation, fuel consumption prediction, and intelligent route optimization in road networks. It aims to find the most eco-friendly, fastest, and shortest routes between locations using custom-trained models and dynamic road conditions.

The project leverages:

Random Forest Regression to predict fuel consumption

Synthetic road network generation using networkx

Multi-factor route comparison based on fuel, time, and distance

Advanced route visualizations using matplotlib

**📌 Key Features**
🚗 Fuel Consumption Model: Trained on synthetic data using realistic driving conditions.

🛣️ Custom Road Network Generator: Generates roads with varying traffic, terrain, and quality.

🧠 ML-Based Edge Scoring: Predicts fuel consumption on every route segment using multiple features.

📈 Route Comparison Engine:

Eco-friendly (min fuel)

Fastest (min time)

Shortest (min distance)

📊 Visual Comparison Dashboard: Shows map + bar charts + efficiency scores.

**🧠 Technologies Used**
Python

scikit-learn (ML model)

pandas, numpy (data handling)

networkx (road graphs)

matplotlib (visualization)

**⚙️ How It Works
Synthetic Data Generation**

Vehicle and road features like weight, engine size, incline, etc.

Computes corresponding fuel consumption using a formula.

Model Training

Trains a RandomForestRegressor to predict fuel usage per segment.

Road Network Creation

Random geometric graph with dynamic traffic, terrain, and road quality.

Route Comparison

Calculates edge-wise fuel cost and travel time.

Uses Dijkstra’s algorithm with different weights to find optimal routes.

Efficiency Score

Combines fuel, time, and distance into a normalized 0–100 score.

Visualization

Side-by-side comparison of all routes with performance metrics and visual cues.

