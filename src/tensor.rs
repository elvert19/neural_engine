use ndarray::Array2;
use rand_distr::{Distribution, Normal};

pub type Tensor = Array2<f64>;

pub fn zeros(rows: usize, cols: usize) -> Tensor {
    Array2::zeros((rows, cols))
}

pub fn ones(rows: usize, cols: usize) -> Tensor {
    Array2::ones((rows, cols))
}

pub fn random_normal(rows: usize, cols: usize) -> Tensor {
    let mut rng = rand::rng();
    let normal = Normal::new(0.0, 1.0).unwrap();
    Array2::from_shape_fn((rows, cols), |_| normal.sample(&mut rng))
}

pub fn xavier_init(rows: usize, cols: usize) -> Tensor {
    let scale = (2.0 / (rows + cols) as f64).sqrt();
    random_normal(rows, cols) * scale
}

pub fn he_init(rows: usize, cols: usize) -> Tensor {
    let scale = (2.0 / rows as f64).sqrt();
    random_normal(rows, cols) * scale
}