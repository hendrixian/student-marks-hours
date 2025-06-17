from numpy import *

def compute_error_for_line_given_points(b,m,points):
    totalError = 0
    for i in range(0,len(points)): # for every points
        x = points[i,0]
        y = points[i,1]
        totalError +=(y-(m*x+b))**2
    return totalError/float(len(points)) # get the average error
 

def step_gradient(b_current,m_current,points,learningRate):
    #starting point for gradient
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    
    for i in range(0,len(points)):
        x = points[i,0]
        y = points[i,1]
        # direction with respect to b and m
        #computing partial derivatives
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * (y - ((m_current * x) + b_current))

    # update b and m from PD, adjusting accordingly to learningrate
    new_b = b_current - (learningRate * b_gradient)
    new_m = m_current - (learningRate * m_gradient)
    return [new_b,new_m]

def gradient_descent_runner(points,starting_b,starting_m,learning_rate,num_iterations):
    b = starting_b
    m = starting_m

    for i in range(num_iterations):
        b,m = step_gradient(b,m,array(points),learning_rate) # update b,m with more accurate b and m
    return [b,m]

def run():
    # 1-collect data
    points = genfromtxt('score_updated.csv',delimiter=',') 
    """
    first loop converts each line into sequences of strings
    second loop converts each string into appropriate data types
    """
    # 2-define hyperparameters
    learning_rate=0.0001 # how fast our model converge(close to best fit)? small updates each iterations
    # y = mx+b
    initial_b =0
    initial_m =0
    num_iterations = 30000

    # 3-train model
    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error={compute_error_for_line_given_points(initial_b,initial_m,points)}")
    print("Running...")
    [b,m] = gradient_descent_runner(points,initial_b,initial_m,learning_rate,num_iterations)
    print(f"After {num_iterations}, ending point  at b = {b}, m = {m}, error={compute_error_for_line_given_points(b,m,points)}")


if __name__ == '__main__':
    run()