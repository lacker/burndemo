use burn::backend::Wgpu;
use model::ModelConfig;

mod data;
mod model;

fn main() {
    type MyBackend = Wgpu<f32, i32>;

    let device = Default::default();
    let model = ModelConfig::new(10, 512).init::<MyBackend>(&device);

    println!("{}", model);
}
