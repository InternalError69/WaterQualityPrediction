<div align="center">
  <h1>💧 AquaGuardian</h1>
  <p>
    <strong>AquaGuardian</strong> is an advanced AI-driven monitoring system designed to detect and predict illegal industrial waste dumping in water bodies. By leveraging machine learning and IoT sensor data, it provides real-time alerts and geospatial insights to protect urban water resources.
  </p>
  <p>
    🚀 <strong>Live Dashboard:</strong> <a href="https://water-quality-prediction-aq.streamlit.app/">https://water-quality-prediction-aq.streamlit.app/</a>
  </p>
</div>

<hr>

<h2>✨ Key Features</h2>
<ul>
  <li><strong>Real-time Spike Detection:</strong> Identifies abnormal fluctuations in pH, TDS, and Turbidity that signal contamination events.</li>
  <li><strong>Deep Learning Analysis:</strong> Employs a <strong>1D Convolutional Neural Network (CNN)</strong> to analyze temporal patterns in water quality.</li>
  <li><strong>Predictive Analytics:</strong> Forecasts potential future dumping events based on historical dumping intervals.</li>
  <li><strong>Interactive Geospatial UI:</strong>
    <ul>
      <li><strong>Lake Monitoring:</strong> Heatmaps visualizing pollution levels across various nodes (e.g., Bengaluru lakes).</li>
      <li><strong>Contamination Classification:</strong> Distinguishes between Mining, Chemical, Sewage, and Industrial runoff.</li>
    </ul>
  </li>
  <li><strong>Multi-Model Comparison:</strong> Evaluates performance across Random Forest, XGBoost, and Logistic Regression models.</li>
</ul>

<hr>

<h2>🛠 Tech Stack</h2>
<table>
  <thead>
    <tr>
      <th>Category</th>
      <th>Tools</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>Frontend</strong></td>
      <td><img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=Streamlit&logoColor=white" alt="Streamlit"></td>
    </tr>
    <tr>
      <td><strong>Machine Learning</strong></td>
      <td>
        <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat&logo=TensorFlow&logoColor=white" alt="TensorFlow">
        <img src="https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white" alt="Scikit-Learn">
      </td>
    </tr>
    <tr>
      <td><strong>Data Processing</strong></td>
      <td>
        <img src="https://img.shields.io/badge/pandas-%23150458.svg?style=flat&logo=pandas&logoColor=white" alt="Pandas">
        <img src="https://img.shields.io/badge/numpy-%23013243.svg?style=flat&logo=numpy&logoColor=white" alt="NumPy">
      </td>
    </tr>
    <tr>
      <td><strong>Visualization</strong></td>
      <td>
        <img src="https://img.shields.io/badge/Folium-77B829?style=flat&logo=Python&logoColor=white" alt="Folium">
        <img src="https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=flat&logo=Matplotlib&logoColor=black" alt="Matplotlib">
      </td>
    </tr>
  </tbody>
</table>

<hr>

<h2>🚀 Getting Started</h2>

<h3>Prerequisites</h3>
<ul>
  <li>Python 3.9+</li>
  <li>pip (Python package manager)</li>
</ul>

<h3>Installation & Launch</h3>
<ol>
  <li>
    <strong>Clone the Repo</strong>
<pre><code>git clone https://github.com/internalerror69/hackverse2k25.git
cd hackverse2k25</code></pre>
  </li>
  <li>
    <strong>Install Dependencies</strong>
<pre><code>pip install -r requirements.txt</code></pre>
  </li>
  <li>
    <strong>Launch the Dashboard</strong>
<pre><code>streamlit run hackverse_dashboard.py</code></pre>
  </li>
</ol>

<hr>

<h2>📊 How It Works</h2>
<p>The system processes raw sensor data through a specialized pipeline:</p>
<ol>
  <li><strong>Data Generation:</strong> A synthetic data engine simulates realistic sensor readings for pH, Temperature, Turbidity, and TDS, including "spike" events representing industrial dumping.</li>
  <li><strong>Preprocessing:</strong> Features are scaled using <code>StandardScaler</code>, and temporal windows are created to allow the CNN to understand sequences of data rather than just single points.</li>
  <li><strong>Inference:</strong> 
    <ul>
      <li>The <strong>1D-CNN Model</strong> extracts spatial-temporal features to classify whether a dumping event is occurring.</li>
      <li><strong>Classical ML Models</strong> (XGBoost, Random Forest) provide a baseline for accuracy and reliability.</li>
    </ul>
  </li>
  <li><strong>Prediction:</strong> The system calculates the time delta between previous spikes to estimate when the next dumping event is likely to occur.</li>
  <li><strong>Visualization:</strong> Data is rendered via Streamlit with Folium maps for geographical context and Matplotlib for historical trend analysis.</li>
</ol>

<hr>

<h2>📁 Repository Highlights</h2>
<ul>
  <li><code>hackverse_dashboard.py</code>: The core Streamlit application handling the UI, map rendering, and real-time inference logic.</li>
  <li><code>HackverseIOT.ipynb</code>: The primary research notebook containing model training, evaluation metrics, and CNN architecture definitions.</li>
  <li><code>Data_generator.ipynb</code>: Script used to create the synthetic datasets used for training and testing.</li>
  <li><code>data/</code>: Directory containing pre-trained models (<code>cnn_model.h5</code>, <code>xgboost_model.pkl</code>), training history, and scalers.</li>
  <li><code>requirements.txt</code>: Complete list of dependencies required to replicate the environment.</li>
</ul>

<hr>

<p align="center">
  <small>MIT License &copy; 2026</small>
</p>
