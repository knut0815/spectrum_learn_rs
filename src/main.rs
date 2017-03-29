extern crate csv;
extern crate gl;
extern crate glutin;
extern crate libc;

/// http://mccormickml.com/2014/03/04/gradient-descent-derivation/
///
/// Performs one step of stochastic gradient descent
fn gradient_descent(m: f64,
                    b: f64,
                    points_x: &[f64],
                    points_y: &[f64],
                    learning_rate: f64,
                    iterations: u32) -> (f64, f64) {
    let mut m_current = m;
    let mut b_current = b;
    let n = points_x.len() as f64;
    let inv_n = 1.0 / n;

    // Perform gradient descent i times
    for i in 0..iterations {
        // Compute the partial derivatives of the MSE cost function with
        // respect to m and b
        let mut m_partial = 0.0;
        let mut b_partial = 0.0;
        for (x, y) in points_x.iter().zip(points_y.iter()) {
            m_partial += ((m_current * x + b_current) - y) * x;
            b_partial += ((m_current * x + b_current) - y);
        }
        m_partial *= inv_n;
        b_partial *= inv_n;

        // Update the current estimates of m and b: the gradient points in
        // the direction of increase, which is why we subtract it from the
        // current estimates
        m_current -= learning_rate * m_partial;
        b_current -= learning_rate * b_partial;
    }
        (m_current, b_current)
}

/// Computes the mean squared error
fn mse(m: f64,
       b: f64,
       points_x: &[f64],
       points_y: &[f64]) -> f64 {
    let mut calculated_error = 0.0;

    for (x, y) in points_x.iter().zip(points_y.iter()) {
        // The equation of the current regression line is: y = m * x + b
        // Find the squared distance between the point on the line corresponding
        // to x and the actual value of y
        calculated_error += (y - (m * x + b)).powf(2.0);
    }
    calculated_error / points_x.len() as f64
}

fn run(points_x: &[f64], points_y: &[f64]) {
    // (1) Collect data

    // (2) Define hyperparameters
    let m_initial = 0.0;
    let b_initial = 0.0;
    let learning_rate = 0.0001;
    let iterations = 1000;

    // (3) Train the model
    let error_initial = mse(m_initial, b_initial, points_x, points_y);
    println!("Starting gradient descent...");
    println!("Initially, m = {}, b = {}, error = {}", m_initial, b_initial, error_initial);

    let (m_final, b_final) = gradient_descent(m_initial, b_initial, points_x, points_y, learning_rate, iterations);
    let error_final = mse(m_final, b_final, points_x, points_y);

    println!("After {} iterations,  m = {}, b = {}, error = {}", iterations, m_final, b_final, error_final);
}

fn main() {
    let mut reader = csv::Reader::from_file("./src/data.csv").unwrap().has_headers(false);

    let mut v0: Vec<f64> = Vec::new();
    let mut v1: Vec<f64> = Vec::new();

    for record in reader.decode() {
        // Test score, number of hours studied
        let (c0, c1): (f64, f64) = record.unwrap();
        v0.push(c0);
        v1.push(c1);
    }

    run(&v0, &v1);
}
