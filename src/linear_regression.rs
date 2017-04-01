pub struct Line {
    pub m: f32,
    pub b: f32
}

pub struct Point {
    pub x: f32,
    pub y: f32,
}

pub struct LinearRegression<'a> {
    pub points: &'a [(f32, f32)],
    learning_rate: f32,
    iterations: u32,
}

impl<'a> LinearRegression<'a> {
    /// Constructs a new `LinearRegression` instance.
    pub fn from_records(pts: &'a [(f32, f32)]) -> LinearRegression<'a> {
        LinearRegression {
            points: pts,
            learning_rate: 0.0001,
            iterations: 10,
        }
    }

    /// Sets the learning rate that will be used during gradient
    /// descent.
    ///
    /// By default, the learning rate is set to 0.0001.
    pub fn learning_rate(mut self, rate: f32) -> LinearRegression<'a> {
        self.learning_rate = rate;
        self
    }

    /// Sets the number of iterations that will be performed during
    /// gradient descent.
    ///
    /// By default, 10 iterations are performed.
    pub fn iterations(mut self, iters: u32) -> LinearRegression<'a> {
        self.iterations = iters;
        self
    }

    /// Performs one iteration of the stochastic gradient descent
    /// algorithm, given a starting slope `m` and y-intercept `b`.
    ///
    /// Returns the newly calculated slope and y-intercept as a tuple.
    fn gradient_descent(&self, m: f32, b: f32) -> (f32, f32) {
        let mut m_current = m;
        let mut b_current = b;
        let n = self.points.len() as f32;
        let inv_n = 1.0 / n;

        for _ in 0..self.iterations {
            // Compute the partial derivatives of the MSE cost function with
            // respect to m and b
            let mut m_partial = 0.0;
            let mut b_partial = 0.0;
            for &(x, y) in self.points.iter() {
                m_partial += ((m_current * x + b_current) - y) * x;
                b_partial += (m_current * x + b_current) - y;
            }
            m_partial *= inv_n;
            b_partial *= inv_n;

            // Update the current estimates of m and b: the gradient points in
            // the direction of increase, which is why we subtract it from the
            // current estimates
            m_current -= self.learning_rate * m_partial;
            b_current -= self.learning_rate * b_partial;
        }
        (m_current, b_current)
    }

    /// Computes the mean squared error, given the current slope
    /// `m` and y-intercept `b`.
    fn mse(&self, m: f32, b: f32) -> f32 {
        let mut calculated_error = 0.0;

        for &(x, y) in self.points.iter() {
            // The equation of the current regression line is: y = m * x + b
            // Find the squared distance between the point on the line corresponding
            // to x and the actual value of y
            calculated_error += (y - (m * x + b)).powf(2.0);
        }
        calculated_error / self.points.len() as f32
    }

    /// Runs the linear regression algorithm with starting
    /// slope m and y-intercept b.
    pub fn run(&self, m_initial: f32, b_initial: f32) -> (f32, f32) {
        let error_initial = self.mse(m_initial, b_initial);
        println!("Starting gradient descent...");
        println!("Initially, m = {}, b = {}, error = {}", m_initial, b_initial, error_initial);

        let (m_final, b_final) = self.gradient_descent(m_initial, b_initial);
        let error_final = self.mse(m_final, b_final);

        println!("After {} iterations,  m = {}, b = {}, error = {}", self.iterations, m_final, b_final, error_final);
        (m_final, b_final)
    }
}
