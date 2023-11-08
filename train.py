from predict_price import estimatePrice

import numpy as np

# import pandas as pd
import csv
import sys


def load_data(filename):
    data_column1 = []
    data_column2 = []

    try:
        with open(filename, "r") as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                data_column1.append(float(row["km"]))
                data_column2.append(float(row["price"]))
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        return None, None

    return data_column1, data_column2


# Program 2: Train the Linear Regression Model




# Training function
def train_linear_regression(mileage_, price_):
    # tmpθ0 = learningRate ∗ 1/m
    # and add for all dataset (since tab start at 0 and size is m lines so last one will be m -1
    # (estimatePrice(mileage[i]) − price[i]) estimated price - actual price
    # Initialize parameters
    theta0, theta1, tmp_theta0, tmp_theta1 = 0, 0, 0, 0
    learning_rate = 0.01  # You can adjust the learning rate
    num_iterations = 200  # You can adjust the number of iterations
    mileage = np.array(mileage_)
    price = np.array(price_)
    # with original data, numbers get big very quickly !
    # we normalize our data so it's clean and easier to work with
    # we normalize with minmaxing (putting everything on a scale from 0 to 1)
    # because it's easy
    m_min_val = np.min(mileage)
    m_max_val = np.max(mileage)
    n_mil = (mileage - m_min_val) / (m_max_val - m_min_val)    
    p_min_val = np.min(price)
    p_max_val = np.max(price)
    n_pri = (price - p_min_val) / (p_max_val - p_min_val)

    m = len(n_mil)  # size of dataset
    for _ in range(num_iterations):  # everytime we train on the whole dataset
        tmp_theta0 = (learning_rate / m) * np.sum(
            estimatePrice(n_mil, theta0, theta1) - n_pri
        )
        tmp_theta1 = (learning_rate / m) * np.sum(
            (estimatePrice(n_mil, theta0, theta1) - n_pri) * n_mil
        )
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
    #dont forget to denormalize thetas or they wont work with regular data
    #denormalized_d = normalized_d * (max_d - min_d) + min_d
    theta0_ = 
    return theta0, theta1


if __name__ == "__main__":
    # Load your dataset (replace with your actual dataset loading code)
    mileage, price = load_data("data.csv")

    # Train the model
    theta0, theta1 = train_linear_regression(mileage, price)
    print(theta0, theta1)
    # Save the trained parameters to a file
    with open("parameters.txt", "w") as file:
        file.write(str(round(theta0, 6)) + "," + str(round(theta1, 6)))