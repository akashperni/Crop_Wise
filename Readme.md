# CropWise 🌿

CropWise is an ML-powered application designed to recommend the most suitable crops to grow on a farm based on various input parameters. This project leverages machine learning techniques to provide precise and accurate crop recommendations, thus aiding precision agriculture.

## Features

- **Crop Recommendation**: Predicts the best crop to grow based on soil and weather parameters.
- **User Input**: Allows users to input parameters such as Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
- **Machine Learning Model**: Utilizes a trained GaussianNB model to make predictions.

## Live Demo

Check out the live demo [here](https://akash-cropwise-625097d1df01.herokuapp.com/).

## How It Works

1. **User Inputs**: Users input the values for Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.
2. **Model Prediction**: The model predicts the most suitable crop based on the provided inputs.
3. **Results**: Displays the recommended crop to the user.

## Output Screen

![Output Screen](/Users/akashperni/Desktop/project/Crop_Wise/output_screen.png)

## Installation

To run this project locally, follow these steps:

1. **Clone the repository**:
   ```sh
   git clone https://github.com/yourusername/cropwise.git
   cd cropwise

2. **Set up a virtual environment**:
   ```sh
   python3 -m venv venv
   source venv/bin/activate

3. **Install dependencies:**:
   ```sh
   pip install -r requirements.txt

4. **Run the application:**:
   ```sh
   streamlit run app.py

## File Structure
- **app.py** : Contains the Streamlit application code.
- **model.py**: Contains the code to train and save the machine learning model.
- **Procfile**: Specifies the commands to run the application on Heroku.
- **setup.sh**: Contains the setup script for configuring the Streamlit server.
- **requirements**.txt: Lists the dependencies required for the project.

## Usage
-**Open the application**:
Navigate to the application URL (e.g., http://localhost:8501 if running locally).

-**Input Parameters**: Enter the values for Nitrogen, Phosphorus, Potassium, Temperature, Humidity, pH, and Rainfall.

-**Predict Crop**:Click the 'Predict' button to get the recommended crop for your farm.

## Deployment

To deploy this application on Heroku:

1. **Create a new Heroku app**:
   ```sh
   heroku create your-app-name

2. **Push the code to Heroku**:
   ```sh
   git push heroku master

3. **Open the application**:
   ```sh
   heroku open


## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any improvements.

## Acknowledgments
- Special thanks to the creators of Streamlit and the machine learning libraries used in this project.
- Inspired by the need for precision agriculture and the benefits it can bring to the farming community.

-----------------------------------------------------------------

- **Note**: This M.L. application is for educational/demo purposes only and cannot be relied upon for actual agricultural decisions.*