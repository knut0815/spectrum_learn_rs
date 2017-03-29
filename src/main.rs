extern crate csv;
extern crate gl;
extern crate glutin;

mod linear_regression;

use linear_regression::LinearRegression;

fn main() {
    let mut reader = csv::Reader::from_file("./src/data.csv").unwrap().has_headers(false);

    let mut points: Vec<(f64, f64)> = Vec::new();
    for record in reader.decode() {
        // Test score, number of hours studied
        let p: (f64, f64) = record.unwrap();
        points.push(p);
    }

    let lin_reg = LinearRegression::from_records(&points).iterations(3);
    lin_reg.run(0.0, 0.0);

    // OpenGL drawing
    let window = glutin::Window::new().unwrap();
    window.set_inner_size(200, 200);

    let mut vao = 0;
    unsafe {
        window.make_current();
        gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);
        gl::ClearColor(1.0, 0.0, 0.0, 1.0);
        let mut vao = 0;
        let mut vbo = 0;
        // Create Vertex Array Object
        gl::GenVertexArrays(1, &mut vao);
        gl::BindVertexArray(vao);
    }
    for event in window.wait_events() {
        unsafe { gl::Clear(gl::COLOR_BUFFER_BIT) };
        window.swap_buffers();

        match event {
            glutin::Event::Closed => break,
            _ => ()
        }
    }
}
