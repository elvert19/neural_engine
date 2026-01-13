use crate::layer::Layer;
use crate::tensor::{Tensor, zeros};

pub trait Optimizer {
    fn update(&mut self, layers: &mut [Box<dyn Layer>]);
}

pub struct SGD {
    learning_rate: f64,
}

impl SGD {
    pub fn new(learning_rate: f64) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for SGD {
    fn update(&mut self, layers: &mut [Box<dyn Layer>]) {
        for layer in layers {
            let grads = layer.get_grads().iter().map(|&g| g.clone()).collect::<Vec<_>>();
            let params = layer.get_params_mut();
            
            for (param, grad) in params.into_iter().zip(grads.iter()) {
                *param -= &(grad * self.learning_rate);
            }
        }
    }
}

pub struct Adam {
    learning_rate: f64,
    beta1: f64,
    beta2: f64,
    epsilon: f64,
    t: usize,
    m: Vec<Vec<Tensor>>,
    v: Vec<Vec<Tensor>>,
}

impl Adam {
    pub fn new(learning_rate: f64) -> Self {
        Self {
            learning_rate,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            t: 0,
            m: Vec::new(),
            v: Vec::new(),
        }
    }
}

impl Optimizer for Adam {
    fn update(&mut self, layers: &mut [Box<dyn Layer>]) {
        self.t += 1;
        
        if self.m.is_empty() {
            for layer in layers.iter() {
                let grads = layer.get_grads();
                let layer_m: Vec<Tensor> = grads.iter()
                    .map(|g| zeros(g.shape()[0], g.shape()[1]))
                    .collect();
                let layer_v: Vec<Tensor> = grads.iter()
                    .map(|g| zeros(g.shape()[0], g.shape()[1]))
                    .collect();
                self.m.push(layer_m);
                self.v.push(layer_v);
            }
        }
        
        for (i, layer) in layers.iter_mut().enumerate() {
            let grads = layer.get_grads().iter().map(|&g| g.clone()).collect::<Vec<_>>();
            let params = layer.get_params_mut();
            
            for (j, (param, grad)) in params.into_iter().zip(grads.iter()).enumerate() {
                self.m[i][j] = self.beta1 * &self.m[i][j] + (1.0 - self.beta1) * grad;
                self.v[i][j] = self.beta2 * &self.v[i][j] + (1.0 - self.beta2) * (grad * grad);
                
                let m_hat = &self.m[i][j] / (1.0 - self.beta1.powi(self.t as i32));
                let v_hat = &self.v[i][j] / (1.0 - self.beta2.powi(self.t as i32));
                
                *param -= &(self.learning_rate * m_hat / (v_hat.mapv(|x| x.sqrt()) + self.epsilon));
            }
        }
    }
}
