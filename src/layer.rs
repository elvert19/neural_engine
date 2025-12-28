use crate::tensor::{Tensor, random_normal, zeros};
use ndarray::Axis;

pub trait Layer {
    fn forward(&mut self, input: Tensor) -> Tensor;
    fn backward(&mut self, output_gradient: Tensor) -> Tensor;
    
    // For the optimizer to access parameters
    fn get_params_mut(&mut self) -> Option<(&mut Tensor, &mut Tensor)>; // weights, biases
    fn get_grads(&self) -> Option<(&Tensor, &Tensor)>; // grad_weights, grad_biases
}

pub struct Dense {
    pub weights: Tensor,
    pub biases: Tensor,
    pub input: Option<Tensor>, // Cache input for backprop
    pub grad_weights: Tensor,
    pub grad_biases: Tensor,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: random_normal(input_size, output_size),
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
        // Y = X . W + B
        input.dot(&self.weights) + &self.biases
    }

    fn backward(&mut self, output_gradient: Tensor) -> Tensor {
        let input = self.input.as_ref().expect("Forward pass must run before backward");

        self.grad_weights = input.t().dot(&output_gradient);
        
 
        self.grad_biases = output_gradient.sum_axis(Axis(0)).insert_axis(Axis(0));

        output_gradient.dot(&self.weights.t())
    }

    fn get_params_mut(&mut self) -> Option<(&mut Tensor, &mut Tensor)> {
        Some((&mut self.weights, &mut self.biases))
    }

    fn get_grads(&self) -> Option<(&Tensor, &Tensor)> {
        Some((&self.grad_weights, &self.grad_biases))
    }
}