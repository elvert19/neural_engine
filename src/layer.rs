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
    // pre-allocated output buffer   reused every forward pass
    // avoids a heap allocation per batch per layer
    output_buf: Option<Tensor>,
}

impl Dense {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        Self {
            weights: he_init(input_size, output_size),
            biases: zeros(1, output_size),
            input: None,
            grad_weights: zeros(input_size, output_size),
            grad_biases: zeros(1, output_size),
            output_buf: None,
        }
    }
}

impl Layer for Dense {
    #[inline(always)]
    fn forward(&mut self, input: Tensor) -> Tensor {
        let batch = input.shape()[0];
        let out   = self.weights.shape()[1];

        // reuse the output buffer if the shape matches (same batch size),
        // otherwise allocate once and cache it for future batches
        let mut output = match self.output_buf.take() {
            Some(buf) if buf.shape() == [batch, out] => buf,
            _ => Tensor::zeros((batch, out)),
        };

        // BLAS dgemm: output = input · weights  (zero-copy, in-place)
        ndarray::linalg::general_mat_mul(
            1.0,
            &input,
            &self.weights,
            0.0,
            &mut output,
        );
        // add bias row-wise (broadcast)
        output += &self.biases;

        self.input = Some(input);

        let ret = output.clone();
        self.output_buf = Some(output);
        ret
    }

    #[inline(always)]
    fn backward(&mut self, output_gradient: Tensor) -> Tensor {
        let input = self.input.as_ref().unwrap();
        self.grad_weights = input.t().dot(&output_gradient);
        self.grad_biases  = output_gradient.sum_axis(Axis(0)).insert_axis(Axis(0));
        output_gradient.dot(&self.weights.t())
    }

    fn get_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![&mut self.weights, &mut self.biases]
    }

    fn get_grads(&self) -> Vec<&Tensor> {
        vec![&self.grad_weights, &self.grad_biases]
    }
}