
# **Mobile Price Prediction Using Machine Learning**

This repository contains the capstone 1 project for the Machine Learning Zoomcamp program. The goal of this project is to develop and deploy a classification machine learning model to predict price range of mobile phone.

---

## **Problem Description**

Bob has started his own mobile company. He wants to give tough fight to big companies like Apple,Samsung etc.

He does not know how to estimate price of mobiles his company creates. In this competitive mobile phone market you cannot simply assume things. To solve this problem he collects sales data of mobile phones of various companies.

Bob wants to find out some relation between features of a mobile phone(eg:- RAM,Internal Memory etc) and its selling price. But he is not so good at Machine Learning. So he needs your help to solve this problem.

In this problem you do not have to predict actual price but a price range indicating how high the price is

Diabetes is a chronic condition that affects millions worldwide. Early diagnosis is crucial for effective management and treatment. The objective of this project is to build a machine learning model that can predict whether a patient has diabetes based on specific medical attributes.

### Dataset
The project uses the **Mobile Phone Dataset**, which contains the following attributes:
## The Features info
21 colums
- price_range:This indicates the mobile phone's price category {0 Very low price / 1 Average price / 2 High price / 3 Very high price}

- battery_power:Battery capacity in milliampere-hours (mAh)

- clock_speed:Processor speed in GHz (CPU)

- n_cores:Number of processor cores

- Ram:RAM size (MB)

- Camera:{fc-->Front camera resolution (megapixels) / pc-->Rear camera resolution}

- four_g:Does it support 4G?

- three_g:Does it support 3G?

- wifi:WiFi support

- blue:Bluetooth Support

- px_height:High screen pixel resolution

- px_width:width screen pixel resolution

- sc_h:Screen height (cm)

- sc_w:Screen width (cm)

- int_memory:Internal memory (GB)

- mobile_wt:Mobile phone weight (grams)

- m_dep:Mobile phone thickness (cm)

- dual_sim:Dual SIM support?

- touch_screen:Is the screen Touch?

- talk_time:Number of hours of calls

- Target variable: `Outcome` (the price range of different mobile phones)

The dataset aims to provide price range for mobile phone based on these features.

---

## **Project Structure**

The repository contains the following files:

- **`readme.md`**: This file, describing the project and how to run it.
- **`mobile_price.ipynb`**: data cleaning and model selection process.
- **`diabetes.py`**: exported script from .ipynb.
- **`train.py`**: A Python script for training the machine learning model and saving it to a file.
- **`predict.py`**: A script for loading the trained model and serving predictions via a web service.
- **`requirements.txt`**: Lists all Python dependencies needed for the project.
- **`Dockerfile`**: Configuration for building a Docker image to run the service.
- **`participant.json`**: Exemple of input data for deployed service
- **`model.bin`**: logistic regression model classification file with (solver='liblinear', C=10, max_iter=1000, random_state=42)

---

## **How to Run the Project**

### Prerequisites
- Docker (for containerized deployment)

### Setup Instructions

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kunlemariam08/machine-learning-zoomcamp-homework/blob/main/capstone1
   ```

2. **Download the dataset**:
   - Dataset is available at (https://www.kaggle.com/datasets/iabhishekofficial/mobile-price-classification/data)
   - Place create `data/` directory and put dataset `train.csv` and `test.csv` inside.

3. **Deploy with Docker**:
   Build and run the Docker container:
   ```bash
   docker build -t price-prediction .
   docker run -it --rm -p 9696:9696 price-prediction
   ```

4. **Access the Application**:
   The application will be accessible at `http://localhost:9696`.
---

## **Deliverables**

1. **EDA and Model Development (`diabetes.ipynb`)**:
   - Data preparation and cleaning steps.
   - Exploratory Data Analysis (EDA), including feature importance analysis.

   Logistic regression showed the best performance. So, the logistic regression model was selected for deployment.

2. **Training Script (`train.py`)**:
   - Trains the final model and saves it as a pickle file.

3. **Prediction Script (`predict.py`)**:
   - Loads the trained model and serves predictions via a REST API.

4. **Docker Deployment**:
   - The service can be containerized using the provided `Dockerfile`.

5. **Dataset Instructions**:
   - Instructions for downloading the dataset and placing it in the `data/` folder.
---

## **Key Features**

- **Machine Learning Model**: Logistic Regression as the baseline model, with performance compared to other algorithms such as Random Forest and XGBoost.
- **REST API**: A lightweight web service to make predictions based on input data.
- **Dockerized Deployment**: Ensures compatibility and easy deployment across environments.

---

## **Acknowledgments**

- **ML Zoomcamp**: This project was created as part of the ML Zoomcamp capstone 1.
- **Dataset**: Sourced from Kaggle.

Feel free to explore the code and suggest improvements! 😊