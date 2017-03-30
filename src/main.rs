extern crate csv;
extern crate gl;
extern crate glutin;

mod linear_regression;

use linear_regression::LinearRegression;
use gl::types::{GLfloat, GLenum, GLuint, GLint, GLchar, GLsizeiptr, GLboolean};
use std::mem;
use std::ffi::CString;
use std::ptr;
use std::str;

static VERTEX_DATA: [GLfloat; 4] = [ -0.5, 0.0, 0.5, 0.0 ];

static VS_SRC: &'static str =
    "#version 330\n\
    layout (location = 0) in vec2 position;\n\
    void main() {\n\
    gl_Position = vec4(position, 0.0, 1.0);\n\
    }";

static FS_SRC: &'static str =
    "#version 330\n\
    out vec4 out_color;\n\
    void main() {\n\
       out_color = vec4(1.0, 1.0, 1.0, 1.0);\n\
    }";

fn compile_shader(src: &str, ty: GLenum) -> GLuint {
    let shader;
    unsafe {
        shader = gl::CreateShader(ty);
        // Attempt to compile the shader
        let c_str = CString::new(src.as_bytes()).unwrap();
        gl::ShaderSource(shader, 1, &c_str.as_ptr(), ptr::null());
        gl::CompileShader(shader);

        // Get the compile status
        let mut status = gl::FALSE as GLint;
        gl::GetShaderiv(shader, gl::COMPILE_STATUS, &mut status);

        // Fail on error
        if status != (gl::TRUE as GLint) {
            let mut len = 0;
            gl::GetShaderiv(shader, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::new();
            buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
            gl::GetShaderInfoLog(shader, len, ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
            panic!("{}", str::from_utf8(buf.as_slice()).ok().expect("ShaderInfoLog not valid utf8"));
        }
    }
    shader
}

fn link_program(vs: GLuint, fs: GLuint) -> GLuint {
    unsafe {
        let program = gl::CreateProgram();
        gl::AttachShader(program, vs);
        gl::AttachShader(program, fs);
        gl::LinkProgram(program);
        // Get the link status
        let mut status = gl::FALSE as GLint;
        gl::GetProgramiv(program, gl::LINK_STATUS, &mut status);

        // Fail on error
        if status != (gl::TRUE as GLint) {
            let mut len: GLint = 0;
            gl::GetProgramiv(program, gl::INFO_LOG_LENGTH, &mut len);
            let mut buf = Vec::new();
            buf.set_len((len as usize) - 1); // subtract 1 to skip the trailing null character
            gl::GetProgramInfoLog(program, len, ptr::null_mut(), buf.as_mut_ptr() as *mut GLchar);
            panic!("{}", str::from_utf8(buf.as_slice()).ok().expect("ProgramInfoLog not valid utf8"));
        }
        program
    }
}

fn main() {
    let mut reader = csv::Reader::from_file("./src/data.csv").unwrap().has_headers(false);

    let mut points: Vec<(f64, f64)> = Vec::new();
    for record in reader.decode() {
        // Test score, number of hours studied
        let p: (f64, f64) = record.unwrap();
        points.push(p);
    }

    let lin_reg = LinearRegression::from_records(&points).iterations(3);
    //lin_reg.run(0.0, 0.0);

    // Create the window
    let window = glutin::Window::new().unwrap();
    let window_size = window.get_inner_size();
    println!("{:?}", window.get_inner_size_points());

    let mut vao = 0;
    let mut vbo = 0;
    let mut shader_program = 0;

    let mut points_to_draw: Vec<GLfloat> = Vec::new();
    for pt in &points {
        points_to_draw.push(pt.0 as f32 / 400.0);
        points_to_draw.push(pt.1 as f32 / 400.0);
    }

    for pt in &points_to_draw {
        println!("{:?}", pt);
    }

    unsafe {
        window.make_current();
        gl::load_with(|symbol| window.get_proc_address(symbol) as *const _);
        gl::ClearColor(0.0, 0.0, 0.0, 1.0);
        
        // Compile and link shaders
        let vs = compile_shader(VS_SRC, gl::VERTEX_SHADER);
        let fs = compile_shader(FS_SRC, gl::FRAGMENT_SHADER);
        shader_program = link_program(vs, fs);

        // Generate the VAO
        gl::GenVertexArrays(1, &mut vao);
        gl::BindVertexArray(vao);

        // Bind the VBO and copy data
        gl::GenBuffers(1, &mut vbo);
        gl::BindBuffer(gl::ARRAY_BUFFER, vbo);
        gl::BufferData(gl::ARRAY_BUFFER,
                       (points_to_draw.len() * mem::size_of::<GLfloat>()) as GLsizeiptr,
                       mem::transmute(&points_to_draw[0]),
                       gl::STATIC_DRAW);

        // Enable the vertex attribute at location 0 (positions)
        gl::EnableVertexAttribArray(0);
        gl::VertexAttribPointer(0, 2, gl::FLOAT, gl::FALSE, 0, ptr::null());

        // Unbind the VAO
        gl::BindVertexArray(0);
    }
    for event in window.wait_events() {
        unsafe {
            gl::Clear(gl::COLOR_BUFFER_BIT);
            gl::UseProgram(shader_program);
            gl::BindVertexArray(vao);
            gl::DrawArrays(gl::POINTS, 0, points_to_draw.len() as i32);
        };
        window.swap_buffers();

        match event {
            glutin::Event::Closed => break,
            _ => ()
        }
    }
}
