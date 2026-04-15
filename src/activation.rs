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
pub struct Softmax  {
    output:  Option<Tensor>,

}


impl Softmax {
    pub fn new ()  -> Self {
        Self {output:None} 
    }
}


impl Layer for Softmax {
    fn forward(&mut self, input: Tensor) -> Tensor {
        // subtract row-max to prevent e^huge = Inf
        let max_vals = input.fold_axis(
            ndarray::Axis(1),
            f64::NEG_INFINITY,
            |&acc, &x| acc.max(x),
        );
        let shifted = &input - &max_vals.insert_axis(ndarray::Axis(1));
        let exp_vals = shifted.mapv(f64::exp);
        let row_sums = exp_vals.sum_axis(ndarray::Axis(1));
        let output = &exp_vals / &row_sums.insert_axis(ndarray::Axis(1));
        self.output = Some(output.clone());
        output
    }

    fn backward(&mut self, grad_output: Tensor) -> Tensor {
        // simplified — correct when paired with CCE loss
        grad_output
    }

    fn get_params_mut(&mut self) -> Vec<&mut Tensor> { vec![] }
    fn get_grads(&self) -> Vec<&Tensor> { vec![] }
}