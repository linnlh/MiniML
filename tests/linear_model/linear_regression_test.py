from mini_ml.linear_model import LinearRegression

def simple_regression_test():
    X = [[1], [2]]
    y = [1, 2]

    reg = LinearRegression()
    reg.fit(X, y)

if __name__ == "__main__":
    simple_regression_test()