pub mod tensor;
pub mod layer;
pub mod activation;
pub mod loss;
pub mod optimizer;
pub mod utilis;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{IntoPyArray, PyArray2, PyReadonlyArray2};
use ndarray::Array2;
use layer::{Layer, Dense};
use activation::{ReLU, Sigmoid, Tanh};
use optimizer::{Optimizer, SGD, Adam};
use loss::{Loss, MSE, BinaryCrossEntropy};

type Tensor = Array2<f64>;

pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
}

impl Sequential {
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    pub fn add(&mut self, layer: impl Layer + 'static) {
        self.layers.push(Box::new(layer));
    }

    pub fn forward(&mut self, input: Tensor) -> Tensor {
        let mut current_output = input;
        for layer in &mut self.layers {
            current_output = layer.forward(current_output);
        }
        current_output
    }

    pub fn train(
        &mut self,
        inputs: &Tensor,
        targets: &Tensor,
        epochs: usize,
        loss_fn: &dyn Loss,
        optimizer: &mut dyn Optimizer,
    ) -> Vec<f64> {
        let mut loss_history = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let output = self.forward(inputs.clone());
            let cost = loss_fn.compute(&output, targets);
            loss_history.push(cost);

            let mut gradient = loss_fn.derivative(&output, targets);

            for layer in self.layers.iter_mut().rev() {
                gradient = layer.backward(gradient);
            }

            optimizer.update(&mut self.layers);

            if epoch % 100 == 0 {
                println!("Epoch {}: Loss = {:.4}", epoch, cost);
            }
        }
        loss_history
    }

    pub fn predict(&mut self, input: &Tensor) -> Tensor {
        self.forward(input.clone())
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass(name = "Sequential")]
struct PySequential {
    inner: Sequential,
    optimizer: Option<PyOptimizer>,
    loss_fn: Option<PyLoss>,
}

#[pymethods]
impl PySequential {
    #[new]
    fn new() -> Self {
        PySequential {
            inner: Sequential::new(),
            optimizer: None,
            loss_fn: None,
        }
    }

    fn add_dense(&mut self, input_size: usize, output_size: usize) {
        self.inner.add(Dense::new(input_size, output_size));
    }

    fn add_relu(&mut self) {
        self.inner.add(ReLU::new());
    }

    fn add_sigmoid(&mut self) {
        self.inner.add(Sigmoid::new());
    }

    fn add_tanh(&mut self) {
        self.inner.add(Tanh::new());
    }

    fn set_optimizer(&mut self, optimizer_type: &str, learning_rate: f64) -> PyResult<()> {
        let opt = match optimizer_type.to_lowercase().as_str() {
            "sgd" => PyOptimizer::SGD(SGD::new(learning_rate)),
            "adam" => PyOptimizer::Adam(Adam::new(learning_rate)),
            _ => return Err(PyValueError::new_err("Unknown optimizer. Use 'sgd' or 'adam'")),
        };
        self.optimizer = Some(opt);
        Ok(())
    }

    fn set_loss(&mut self, loss_type: &str) -> PyResult<()> {
        let loss = match loss_type.to_lowercase().as_str() {
            "mse" => PyLoss::MSE(MSE),
            "bce" | "binary_crossentropy" => PyLoss::BCE(BinaryCrossEntropy),
            _ => return Err(PyValueError::new_err("Unknown loss. Use 'mse' or 'bce'")),
        };
        self.loss_fn = Some(loss);
        Ok(())
    }

    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        let input_tensor = input.as_array().to_owned();
        let output_tensor = self.inner.forward(input_tensor);
        output_tensor.into_pyarray(py)
    }

    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<f64>,
    ) -> &'py PyArray2<f64> {
        let input_tensor = input.as_array().to_owned();
        let output_tensor = self.inner.predict(&input_tensor);
        output_tensor.into_pyarray(py)
    }

    fn train<'py>(
        &mut self,
        py: Python<'py>,
        inputs: PyReadonlyArray2<f64>,
        targets: PyReadonlyArray2<f64>,
        epochs: usize,
    ) -> PyResult<&'py PyArray2<f64>> {
        let inputs_tensor = inputs.as_array().to_owned();
        let targets_tensor = targets.as_array().to_owned();

        let optimizer = self.optimizer.as_mut()
            .ok_or_else(|| PyValueError::new_err("Optimizer not set. Call set_optimizer() first"))?;
        
        let loss_fn = self.loss_fn.as_ref()
            .ok_or_else(|| PyValueError::new_err("Loss function not set. Call set_loss() first"))?;

        let loss_history = self.inner.train(
            &inputs_tensor,
            &targets_tensor,
            epochs,
            loss_fn.as_loss(),
            optimizer.as_optimizer_mut(),
        );

        let loss_array = Array2::from_shape_vec((loss_history.len(), 1), loss_history)
            .map_err(|e| PyValueError::new_err(format!("Failed to create loss array: {}", e)))?;

        Ok(loss_array.into_pyarray(py))
    }

    fn layer_count(&self) -> usize {
        self.inner.layers.len()
    }
}

enum PyOptimizer {
    SGD(SGD),
    Adam(Adam),
}

impl PyOptimizer {
    fn as_optimizer_mut(&mut self) -> &mut dyn Optimizer {
        match self {
            PyOptimizer::SGD(opt) => opt,
            PyOptimizer::Adam(opt) => opt,
        }
    }
}

enum PyLoss {
    MSE(MSE),
    BCE(BinaryCrossEntropy),
}

impl PyLoss {
    fn as_loss(&self) -> &dyn Loss {
        match self {
            PyLoss::MSE(loss) => loss,
            PyLoss::BCE(loss) => loss,
        }
    }
}

#[pymodule]
fn neural_engine(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySequential>()?;
    Ok(())
}