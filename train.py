from predict_price import estimatePrice
#import numpy as np
#import pandas as pd
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
def train_linear_regression(
    mileage, price, theta0, theta1, learning_rate, num_iterations
):
# tmpθ0 = learningRate ∗ 1/m()i=0(estimatePrice(mileage[i]) − price[i])

    x = mileage #X IS USUALLY USED AS THE DATA WE HAVE
    y = price # WHEREAS Y IS THE DATA WE PREDICT
    m = float(len(mileage))  # size of dataset
    print(m, mileage)

    for _ in range(num_iterations):  # everytime we train on the whole dataset
        tmp_theta0 = 0
        tmp_theta1 = 0
        curve = learning_rate / m
        for i in range(int(m)):
            estimated_price = estimatePrice(int(mileage[i]), theta0, theta1)
            tmp_theta0 += estimated_price - int(price[i])
            tmp_theta1 += (estimated_price - int(price[i])) * int(mileage[i])
        print(learning_rate, m)
        
        tmp_theta0 *= curve
        tmp_theta1 *= curve
        print(tmp_theta0, tmp_theta1)
        theta0 -= tmp_theta0
        theta1 -= tmp_theta1
    return theta0, theta1


if __name__ == "__main__":
    # Load your dataset (replace with your actual dataset loading code)
    mileage, price = load_data("data.csv")

    # Initialize parameters
    theta0, theta1 = 0, 0
    learning_rate = 0.1  # You can adjust the learning rate
    num_iterations = 200  # You can adjust the number of iterations

    # Train the model
    theta0, theta1 = train_linear_regression(
        mileage, price, theta0, theta1, learning_rate, num_iterations
    )
    print(theta0, theta1)
    # Save the trained parameters to a file
    with open("parameters.txt", "w") as file:
        file.write(theta0, theta1)