use crate::layer::Layer;
use crate::tensor::Tensor;


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
        
        let derivative = input.mapv(|x| if x > 0.0 { 1.0 } else { 0.0 });
        output_gradient * derivative
    }

    fn get_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![] // No learnable parameters
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
        // Sigmoid formula: 1 / (1 + e^-x)
        let output = input.mapv(|x| 1.0 / (1.0 + (-x).exp()));
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, output_gradient: Tensor) -> Tensor {
        let output = self.output.as_ref().unwrap();
        
        let derivative = output * (1.0 - output);
        output_gradient * derivative
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
        // Derivative: 1 - output^2
        let derivative = 1.0 - (output * output);
        output_gradient * derivative
    }

    fn get_params_mut(&mut self) -> Vec<&mut Tensor> {
        vec![]
    }

    fn get_grads(&self) -> Vec<&Tensor> {
        vec![]
    }
}