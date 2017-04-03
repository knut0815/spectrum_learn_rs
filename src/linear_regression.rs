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
    pub m: f32,
    pub b: f32,
    learning_rate: f32,
}

impl<'a> LinearRegression<'a> {
    /// Constructs a new `LinearRegression` instance.
    pub fn from_records(pts: &'a [(f32, f32)]) -> LinearRegression<'a> {
        LinearRegression {
            m: 0.0,
            b: 0.0,
            points: pts,
            learning_rate: 0.0001,
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

    /// Performs one iteration of the stochastic gradient descent
    /// algorithm, given a starting slope `m` and y-intercept `b`.
    ///
    /// Returns the newly calculated slope and y-intercept as a tuple.
    pub fn step_gradient_descent(&mut self, iterations: u32) {
        let n = self.points.len() as f32;
        let inv_n = 1.0 / n;

        for _ in 0..iterations {
            // Compute the partial derivatives of the MSE cost function with
            // respect to m and b
            let mut m_partial = 0.0;
            let mut b_partial = 0.0;
            for &(x, y) in self.points.iter() {
                m_partial += ((self.m * x + self.b) - y) * x;
                b_partial += (self.m * x + self.b) - y;
            }
            m_partial *= inv_n;
            b_partial *= inv_n;

            // Update the current estimates of m and b: the gradient points in
            // the direction of increase, which is why we subtract it from the
            // current estimates
            self.m -= self.learning_rate * m_partial;
            self.b -= self.learning_rate * b_partial;
        }
    }

    /// Computes the mean squared error, given the current slope
    /// `m` and y-intercept `b`.
    pub fn mse(&self) -> f32 {
        let mut calculated_error = 0.0;

        for &(x, y) in self.points.iter() {
            // The equation of the current regression line is: y = m * x + b
            // Find the squared distance between the point on the line corresponding
            // to x and the actual value of y
            calculated_error += (y - (self.m * x + self.b)).powf(2.0);
        }
        calculated_error / self.points.len() as f32
    }
}
