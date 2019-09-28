import numpy as np

def cofiCostFunc(params, Y, R, num_users, num_movies,
                      num_features, lambda_=0.0):

    X = params[:num_movies*num_features].reshape(num_movies, num_features)
    Theta = params[num_movies*num_features:].reshape(num_users, num_features)

    J = 0
    X_grad = np.zeros(X.shape)
    Theta_grad = np.zeros(Theta.shape)

    squared_error = np.power(np.dot(X,Theta.T) - Y,2)

    J = (1/2.) * np.sum(squared_error * R)
    X_grad = np.dot(( np.dot(X, Theta.T) - Y ) * R, Theta)
    Theta_grad = np.dot((( np.dot(X, Theta.T) - Y ) * R).T, X)

    J = J + (lambda_/2.)*( np.sum( np.power(Theta, 2) ) + np.sum( np.power(X, 2) ) )
    X_grad = X_grad + lambda_*X
    Theta_grad = Theta_grad + lambda_*Theta


    grad = np.concatenate([X_grad.ravel(), Theta_grad.ravel()])
    return J, grad