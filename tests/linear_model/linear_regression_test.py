from mini_ml.linear_model import LinearRegression

def simple_regression_test():
    X = [[1], [2]]
    y = [2, 4]

    reg = LinearRegression()
    reg.fit(X, y)

    X_test = [[3]]
    y_pred = reg.predict(X_test)
    print(y_pred)

if __name__ == "__main__":
    simple_regression_test()