use crate::tensor::{Tensor, he_init, zeros};
use ndarray::Axis;

pub trait Layer: Send + Sync {
    fn forward(&mut self, input: Tensor) -> Tensor;
    fn backward(&mut self, output_gradient: Tensor) -> Tensor;
    fn get_params_mut(&mut self) -> Vec<&mut Tensor>;
    fn get_grads(&self) -> Vec<&Tensor>;
}

pub struct Dense {
    pub weights: Tensor,
    pub biases: Tensor,
    input: Option<Tensor>,
    grad_weights: Tensor,
    grad_biases: Tensor,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: he_init(input_size, output_size),
            biases: zeros(1, output_size),
            input: None,
            grad_weights: zeros(input_size, output_size),
            grad_biases: zeros(1, output_size),
        }
    }
}

impl Layer for Dense {
    fn forward(&mut self, input: Tensor) -> Tensor {
        self.input = Some(input.clone());
        input.dot(&self.weights) + &self.biases
    }

    fn backward(&mut self, output_gradient: Tensor) -> Tensor {
        let input = self.input.as_ref().unwrap();
        self.grad_weights = input.t().dot(&output_gradient);
        self.grad_biases = output_gradient.sum_axis(Axis(0)).insert_axis(Axis(0));
        output_gradient.dot(&self.weights.t())
    }

    fn get_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.biases]
    }

    fn get_grads(&self) -> Vec<&Tensor> {
        vec![&self.grad_weights, &self.grad_biases]
    }
}