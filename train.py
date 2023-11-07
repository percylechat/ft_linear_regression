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


def normalize_data(data):
    # we normalize with minmaxing (putting everything on a crale from 0 to 1)
    # because it's easy
    min_val = np.min(data)
    max_val = np.max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    return scaled_data


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
    n_mil = normalize_data(mileage)
    n_pri = normalize_data(price)
    m = len(n_mil)  # size of dataset
    for _ in range(num_iterations):  # everytime we train on the whole dataset
        tmp_theta0 = (learning_rate / m) * np.sum(
            estimatePrice(n_mil, theta0, theta1) - n_pri
        )
        tmp_theta1 = (learning_rate / m) * np.sum(
            (estimatePrice(n_mil, theta0, theta1) - n_pri) * n_mil
        )
        # print(theta0, tmp_theta0, theta1, tmp_theta1)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
        # tmp_theta0 = 0
        # tmp_theta1 = 0
        # curve = learning_rate / m
        # for i in range(int(m)):
        #     estimated_price = estimatePrice(int(mileage[i]), theta0, theta1)
        #     tmp_theta0 += estimated_price - int(price[i])
        #     tmp_theta1 += (estimated_price - int(price[i])) * int(mileage[i])
        # print(learning_rate, m)

        # tmp_theta0 *= curve
        # tmp_theta1 *= curve
        # print(tmp_theta0, tmp_theta1)
        # theta0 -= tmp_theta0
        # theta1 -= tmp_theta1
    return theta0, theta1


if __name__ == "__main__":
    # Load your dataset (replace with your actual dataset loading code)
    mileage, price = load_data("data.csv")

    # Train the model
    theta0, theta1 = train_linear_regression(mileage, price)
    print(theta0, theta1)
    # Save the trained parameters to a file
    with open("parameters.txt", "w") as file:
        file.write(str(theta0) + ", " + str(theta1))
