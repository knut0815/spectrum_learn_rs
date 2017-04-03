extern crate csv;
extern crate gl;
extern crate glutin;

mod linear_regression;
mod program;

use linear_regression::LinearRegression;
use program::Program;

use gl::types::{GLfloat, GLenum, GLuint, GLint, GLchar, GLsizeiptr, GLboolean};
use std::mem;
use std::ptr;

fn map(v: f32, fmin: f32, fmax: f32, tmin: f32, tmax: f32) -> f32 {
    (v - fmin) * ((tmax - tmin) / (fmax - fmin)) + tmin
}

fn main() {
    let mut reader = csv::Reader::from_file("./src/data.csv").unwrap().has_headers(false);

    // Parse the CSV file while keeping track of the
    // minimum and maximum value of each feature
    let mut points: Vec<(f32, f32)> = Vec::new();
    let mut max_x = 0.0;
    let mut max_y = 0.0;
    let mut min_x = std::f32::MAX;
    let mut min_y = std::f32::MAX;
    for record in reader.decode() {
        // Test score, number of hours studied
        let p: (f32, f32) = record.unwrap();
        if p.0 > max_x { max_x = p.0; }
        if p.1 > max_y { max_y = p.1; }
        if p.0 < min_x { min_x = p.0; }
        if p.1 < min_y { min_y = p.1; }

        points.push(p);
    }

    /// Create and run the linear regression algorithm on the data set
    ///
    ///
    const MAX_ITERATIONS: u32 = 100;
    let mut lin_reg = LinearRegression::from_records(&points);
    let error_initial = lin_reg.mse();
    println!("Starting gradient descent...");
    println!("Initially, m = {}, b = {}, error = {}", lin_reg.m, lin_reg.b, error_initial);

    /// Setup the window
    ///
    ///
    let window = glutin::Window::new().unwrap();
    let window_size = window.get_inner_size();

    let mut points_to_draw: Vec<GLfloat> = Vec::new();
    for pt in &points {
        points_to_draw.push(map(pt.0 as f32, min_x as f32, max_x as f32, -1.0, 1.0));
        points_to_draw.push(map(pt.1 as f32, min_y as f32, max_y as f32, -1.0, 1.0));
    }

    // Handles to OpenGL objects
    let mut scatter_vao = 0;
    let mut scatter_vbo = 0;
    let mut line_vao = 0;
    let mut line_vbo = 0;
    let mut best_fit: Vec<f32> = vec![-1.0, -1.0 * lin_reg.m + lin_reg.b, 1.0, 1.0 * lin_reg.m + lin_reg.b];
    let mut shader_program = 0;

    unsafe {
        window.make_current();
        let window_size = window.get_inner_size().unwrap();

        gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);
        gl::Viewport(0, 0, window_size.0 as i32, window_size.1 as i32);
        gl::ClearColor(0.0, 0.0, 0.0, 1.0);

        let points_program = Program::from_file("src/points.vert", "src/points.frag");
        gl::PointSize(4.0);

        /// Create a VBO and VAO for rendering the scatter plot
        ///
        ///
        gl::GenVertexArrays(1, &mut scatter_vao);
        gl::BindVertexArray(scatter_vao);

        gl::GenBuffers(1, &mut scatter_vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, scatter_vbo);
        gl::BufferData(gl::ARRAY_BUFFER,
                       (points_to_draw.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
                       mem::transmute(&points_to_draw[0]),
                       gl::STATIC_DRAW);

        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 0, ptr::null());

        gl::BindVertexArray(0);

        /// Create a VBO and VAO for rendering the best-fit line
        ///
        ///
        gl::GenVertexArrays(1, &mut line_vao);
        gl::BindVertexArray(line_vao);

        gl::GenBuffers(1, &mut line_vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, line_vbo);
        gl::BufferData(gl::ARRAY_BUFFER,
                       (best_fit.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
                       mem::transmute(&best_fit[0]),
                       gl::STATIC_DRAW);

        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 0, ptr::null());

        gl::BindVertexArray(0);

        /// Bind the shader program that will be used to draw the
        /// scatter plot as well as the best-fit line
        ///
        ///
        points_program.bind();
    }

    let mut i: u32 = 0;
    let mut frame_count: u32 = 0;
    loop {
        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);

            // Draw scatter plot
            gl::BindVertexArray(scatter_vao);
            gl::DrawArrays(gl::POINTS, 0, points_to_draw.len() as i32);

            // Draw best-fit line
            gl::BindVertexArray(line_vao);
            gl::DrawArrays(gl::LINES, 0, best_fit.len() as i32);

            // Update data
            if frame_count % 3000 == 0 && i < MAX_ITERATIONS {
                lin_reg.step_gradient_descent(1);
                let error = lin_reg.mse();
                println!("Taking one step,  m = {}, b = {}, error = {}", lin_reg.m, lin_reg.b, error);

                best_fit = vec![-1.0, -1.0 * lin_reg.m + lin_reg.b, 1.0, 1.0 * lin_reg.m + lin_reg.b];
                gl::BindBuffer(gl::ARRAY_BUFFER, line_vbo);
                gl::BufferSubData(gl::ARRAY_BUFFER,
                                  0,
                                  (best_fit.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
                                  mem::transmute(&best_fit[0]));
                i += 1;
            }
            frame_count += 1;
        }

        window.swap_buffers();

        for event in window.poll_events() {
            match event {
                glutin::Event::Closed => return,
                _ => ()
            }
        }
    }

}
