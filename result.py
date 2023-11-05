# tuto theorique https://towardsdatascience.com/machine-leaning-cost-function-and-gradient-descend-75821535b2ef
# tuto pratique https://towardsdatascience.com/implementing-gradient-descent-in-python-from-scratch-760a8556c31f


# Load the trained parameters from a file
def load_parameters(filename):
    try:
        with open(filename, "r") as file:
            theta0, theta1 = map(float, file.read().split())
            return theta0, theta1
    except FileNotFoundError:
        return 0, 0  # Default values if no parameters have been trained yet


# Prediction function
def estimatePrice(mileage, theta0, theta1):
    # estimatePrice(mileage) = θ0 + (θ1 ∗ mileage)
    return int(theta0) + (int(theta1) * int(mileage))


if __name__ == "__main__":
    # Load the trained parameters
    theta0, theta1 = load_parameters("parameters.txt")

    # Get user input for mileage
    mileage = float(input("Enter the mileage of the car: "))

    # Predict the price
    estimated_price = estimatePrice(mileage, theta0, theta1)
    print("The estimated price for a car with", mileage, "mileage is:", estimated_price)
