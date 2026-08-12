//! field-glass — 場の生き窓。
//! FieldRuntime live: 毎frame tick→cur()→窓。クリック=撃。ESC=閉。
//! journal = proof/field-glass.fldj (非/tmp、replay可).

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::rc::Rc;
use std::time::Instant;

use field::plane::{to_gray, WaveParams};
use field::runtime::FieldRuntime;
use field::FieldConfig;
use tao::dpi::LogicalSize;
use tao::event::{ElementState, Event, MouseButton, WindowEvent};
use tao::event_loop::{ControlFlow, EventLoop};
use tao::keyboard::Key;
use tao::window::WindowBuilder;

const W: usize = 128;
const H: usize = 128;
const SCALE: usize = 5;
const RANGE: f32 = 0.06;

fn main() {
    let cfg = FieldConfig::new(W, H);
    let params = WaveParams { seed: 42, ..WaveParams::default() };
    assert!(params.stable(), "CFL");
    let journal: PathBuf = {
        let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../proof");
        std::fs::create_dir_all(&d).unwrap();
        d.join("field-glass.fldj")
    };
    let mut rt = FieldRuntime::new(cfg, params, 4096, &journal).expect("runtime");
    rt.excite(W / 2, H / 2, 1.0);

    let event_loop = EventLoop::new();
    let window = Rc::new(
        WindowBuilder::new()
            .with_title("場 field-glass — click=撃 esc=閉")
            .with_inner_size(LogicalSize::new((W * SCALE) as f64, (H * SCALE) as f64))
            .build(&event_loop)
            .expect("window"),
    );
    let context = softbuffer::Context::new(window.clone()).expect("sb context");
    let mut surface = softbuffer::Surface::new(&context, window.clone()).expect("sb surface");

    let mut gray: Vec<u8> = Vec::new();
    let mut cursor = (0f64, 0f64);
    let mut frames = 0u64;
    let mut t0 = Instant::now();

    event_loop.run(move |event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    let _ = rt.flush();
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput { event, .. } => {
                    if event.logical_key == Key::Escape {
                        let _ = rt.flush();
                        *control_flow = ControlFlow::Exit;
                    }
                }
                WindowEvent::CursorMoved { position, .. } => {
                    cursor = (position.x, position.y);
                }
                WindowEvent::MouseInput { state: ElementState::Pressed, button: MouseButton::Left, .. } => {
                    let sf = window.scale_factor();
                    let x = ((cursor.0 / sf) as usize / SCALE).min(W - 1);
                    let y = ((cursor.1 / sf) as usize / SCALE).min(H - 1);
                    rt.excite(x, y, 1.0);
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                rt.tick();
                window.request_redraw();
            }
            Event::RedrawRequested(_) => {
                let size = window.inner_size();
                let (pw, ph) = (size.width as usize, size.height as usize);
                if pw == 0 || ph == 0 {
                    return;
                }
                surface
                    .resize(NonZeroU32::new(pw as u32).unwrap(), NonZeroU32::new(ph as u32).unwrap())
                    .unwrap();
                let cur = rt.cur();
                to_gray(&cur, RANGE, &mut gray);
                let mut buf = surface.buffer_mut().unwrap();
                for py in 0..ph {
                    let fy = (py * H / ph).min(H - 1);
                    for px in 0..pw {
                        let fx = (px * W / pw).min(W - 1);
                        let g = gray[fy * W + fx] as u32;
                        buf[py * pw + px] = (g << 16) | (g << 8) | g;
                    }
                }
                buf.present().unwrap();
                frames += 1;
                if frames % 240 == 0 {
                    let dt = t0.elapsed().as_secs_f64();
                    println!("[field-glass] {:.1} fps · digest {:#018x}", 240.0 / dt, rt.digest());
                    t0 = Instant::now();
                }
            }
            _ => {}
        }
    });
}
