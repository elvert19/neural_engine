use crate::tensor::Tensor;
use crate::layer::Layer;

pub struct ReLU {
    input: Option<Tensor>,
}

impl ReLU {
    pub fn new() -> Self {
        Self { input: None }
    }
}

impl Layer for ReLU {
    fn forward(&mut self, input: Tensor) -> Tensor {
        self.input = Some(input.clone());
        input.mapv(|x| x.max(0.0))
    }

    fn backward(&mut self, output_gradient: Tensor) -> Tensor {
        let input = self.input.as_ref().unwrap();
        output_gradient * input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 })
    }

    fn get_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn get_grads(&self) -> Vec<&Tensor> {
        vec![]
    }
}

pub struct Sigmoid {
    output: Option<Tensor>,
}

impl Sigmoid {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Layer for Sigmoid {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let output = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, output_gradient: Tensor) -> Tensor {
        let output = self.output.as_ref().unwrap();
        output_gradient * output * (1.0 - output)
    }

    fn get_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn get_grads(&self) -> Vec<&Tensor> {
        vec![]
    }
}

pub struct Tanh {
    output: Option<Tensor>,
}

impl Tanh {
    pub fn new() -> Self {
        Self { output: None }
    }
}

impl Layer for Tanh {
    fn forward(&mut self, input: Tensor) -> Tensor {
        let output = input.mapv(|x| x.tanh());
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, output_gradient: Tensor) -> Tensor {
        let output = self.output.as_ref().unwrap();
        output_gradient * (1.0 - output * output)
    }

    fn get_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn get_grads(&self) -> Vec<&Tensor> {
        vec![]
    }
}
