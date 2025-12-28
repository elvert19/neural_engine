use ndarray::Array2; //defines the array in 2 dimensional

pub type Tensor = Array2<f32>;

pub fn multiply(a: &Tensor, b: &Tensor) -> Tensor{
    a.dot(b) // this is where matrix multiplication is done
}
