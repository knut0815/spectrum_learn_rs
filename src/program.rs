extern crate gl;

use gl::types::{GLfloat, GLenum, GLuint, GLint, GLchar, GLsizeiptr, GLboolean};
use std::mem;
use std::ffi::CString;
use std::ptr;
use std::str;

use std::error::Error;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;

pub struct Program {
    program_id: GLuint,
}

impl Drop for Program {
    fn drop(&mut self) {
        unsafe { gl::DeleteProgram(self.program_id); }
    }
}

impl Program {
    pub fn compile_shader(file_path: &str, shader_stage: GLenum) -> GLuint {
        // Attempt to open the file that contains the shader
        // source code
        let path = Path::new(file_path);
        let mut file_handle = File::open(&path)
            .expect("Failed to open file");
        let mut contents = String::new();
        file_handle.read_to_string(&mut contents)
            .expect("Failed to read file contents");

        let shader;
        unsafe {
            shader = gl::CreateShader(shader_stage);

            // Attempt to compile the shader
            let c_str = CString::new(contents.as_bytes()).unwrap();
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

    pub fn link_program(vs: GLuint, fs: GLuint) -> GLuint {
        unsafe {
            let program = gl::CreateProgram();
            gl::AttachShader(program, vs);
            gl::AttachShader(program, fs);
            gl::LinkProgram(program);

            // Shaders can be safely deleted
            gl::DeleteShader(vs);
            gl::DeleteShader(fs);

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

    pub fn from_file(path_to_vs: &str, path_to_fs: &str) -> Program {
        let vs = Program::compile_shader(path_to_vs, gl::VERTEX_SHADER);
        let fs = Program::compile_shader(path_to_fs, gl::FRAGMENT_SHADER);

        Program { program_id: Program::link_program(vs, fs) }
    }

    pub fn bind(&self) {
        unsafe { gl::UseProgram(self.program_id); }
    }

    pub fn unbind(&self) {
        unsafe { gl::UseProgram(0); }
    }
}
