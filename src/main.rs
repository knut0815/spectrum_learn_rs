extern crate csv;

/// http://mccormickml.com/2014/03/04/gradient-descent-derivation/
///
/// Performs one step of stochastic gradient descent
fn gradient_descent(m: f64,
                    b: f64,
                    points_x: &[f64],
                    points_y: &[f64],
                    learning_rate: f64) -> (f64, f64) {
    // Initialize both partial derivatives to zero
    let mut m_partial = 0.0;
    let mut b_partial = 0.0;
    let n = points_x.len() as f64;

    // Compute partial derivatives with respect to m and b, respectively
    for (x, y) in points_x.iter().zip(points_y.iter()) {
        m_partial += -(2.0 / n) * (y - (m * x + b));
        b_partial += (2.0 / n) * x * (y - (m * x + b));
    }

    // Update m and b
    (m - learning_rate * m_partial, b - learning_rate * b_partial)
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
        println!("{}, {} - accumulated error: {}", x, y, calculated_error);
    }
    calculated_error / points_x.len() as f64
}

fn run(points_x: &[f64], points_y: &[f64]) {
    // (1) Collect data

    // (2) Define hyperparameters
    let m_initial = 0.0;
    let b_initial = 0.0;
    let learning_rate = 0.0001;
    let iterations = 1;

    // (3) Train the model
    let error_initial = mse(m_initial, b_initial, points_x, points_y);
    println!("Starting gradient descent...");
    println!("Initially, m = {}, b = {}, error = {}", m_initial, b_initial, error_initial);
    let mut m_current = m_initial;
    let mut b_current = b_initial;
    for i in 0..iterations {
        (m_current, b_current) = gradient_descent(m_initial, b_initial, points_x, points_y, learning_rate);
    }

    let error_final = mse(m_current, b_current, points_x, points_y);
    println!("After {} iterations,  m = {}, b = {}, error = {}", iterations, m_current, b_current, error_final);
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
