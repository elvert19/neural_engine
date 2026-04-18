extern crate blas_src;
pub mod tensor;
pub mod layer;
pub mod activation;
pub mod loss;
pub mod optimizer;
pub mod utilis;

use pyo3::prelude::*;
use pyo3::exceptions::PyValueError;
use numpy::{PyArray2, PyReadonlyArray2};
use ndarray::{Array2, Axis};
use layer::{Layer, Dense};
use activation::{ReLU, Sigmoid, Tanh, Softmax};
use optimizer::{Optimizer, SGD, Adam};
use loss::{Loss, MSE, BinaryCrossEntropy, CategoricalCrossEntropy};


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

    pub fn predict(&mut self, input: &Tensor) -> Tensor {
        self.forward(input.clone())
    }
}

impl Default for Sequential {
    fn default() -> Self {
        Self::new()
    }
}

#[pyclass(name = "Sequential", unsendable)]
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

    fn add_softmax(&mut self) {
        self.inner.add(Softmax::new());
    }

    fn set_optimizer(&mut self, optimizer_type: &str, learning_rate: f64) -> PyResult<()> {
        let opt = match optimizer_type.to_lowercase().as_str() {
            "sgd"  => PyOptimizer::SGD(SGD::new(learning_rate)),
            "adam" => PyOptimizer::Adam(Adam::new(learning_rate)),
            _ => return Err(PyValueError::new_err(
                "Unknown optimizer. Use 'sgd' or 'adam'",
            )),
        };
        self.optimizer = Some(opt);
        Ok(())
    }

    fn set_loss(&mut self, loss_type: &str) -> PyResult<()> {
        let loss = match loss_type.to_lowercase().as_str() {
            "mse"                              => PyLoss::MSE(MSE),
            "bce" | "binary_crossentropy"      => PyLoss::BCE(BinaryCrossEntropy),
            "cce" | "categorical_crossentropy" => PyLoss::CCE(CategoricalCrossEntropy),
            _ => return Err(PyValueError::new_err(
                "Unknown loss. Use 'mse', 'bce', or 'cce'",
            )),
        };
        self.loss_fn = Some(loss);
        Ok(())
    }

    fn forward<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let input_tensor = input.as_array().to_owned();
        let output_tensor = self.inner.forward(input_tensor);
        PyArray2::from_array(py, &output_tensor)
    }

    fn predict<'py>(
        &mut self,
        py: Python<'py>,
        input: PyReadonlyArray2<f64>,
    ) -> Bound<'py, PyArray2<f64>> {
        let input_tensor = input.as_array().to_owned();
        let output_tensor = self.inner.predict(&input_tensor);
        PyArray2::from_array(py, &output_tensor)
    }

    fn train<'py>(
        &mut self,
        py: Python<'py>,
        inputs: PyReadonlyArray2<f64>,
        targets: PyReadonlyArray2<f64>,
        epochs: usize,
        batch_size: usize,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let inputs_t = inputs.as_array().to_owned();
        let targets_t = targets.as_array().to_owned();
        let n = inputs_t.shape()[0];

        let optimizer = self.optimizer.as_mut()
            .ok_or_else(|| PyValueError::new_err("Call set_optimizer() first"))?;
        let loss_fn = self.loss_fn.as_ref()
            .ok_or_else(|| PyValueError::new_err("Call set_loss() first"))?;

        let mut loss_history = Vec::with_capacity(epochs);

        for epoch in 0..epochs {
            let mut indices: Vec<usize> = (0..n).collect();
            for i in (1..n).rev() {
                let j = (i as f64
                    * (epoch as f64 * 6_364_136_223_846_793_005.0
                        + 1_442_695_040_888_963_407.0)
                        .abs()
                    / u64::MAX as f64) as usize
                    % (i + 1);
                indices.swap(i, j);
            }

            let mut epoch_loss = 0.0_f64;
            let mut num_batches = 0_usize;

            for chunk in indices.chunks(batch_size) {
                let batch_x = inputs_t.select(Axis(0), chunk);
                let batch_y = targets_t.select(Axis(0), chunk);

                let output = self.inner.forward(batch_x);
                let cost = loss_fn.as_loss().compute(&output, &batch_y);
                epoch_loss += cost;
                num_batches += 1;

                let mut grad = loss_fn.as_loss().derivative(&output, &batch_y);
                for layer in self.inner.layers.iter_mut().rev() {
                    grad = layer.backward(grad);
                }
                optimizer.as_optimizer_mut().update(&mut self.inner.layers);
            }

            let avg_loss = epoch_loss / num_batches as f64;
            loss_history.push(avg_loss);
            if epoch % 100 == 0 {
                println!("Epoch {}: Loss = {:.4}", epoch, avg_loss);
            }
        }

        let arr = Array2::from_shape_vec((loss_history.len(), 1), loss_history)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(PyArray2::from_array(py, &arr))
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
    CCE(CategoricalCrossEntropy),
}

impl PyLoss {
    fn as_loss(&self) -> &dyn Loss {
        match self {
            PyLoss::MSE(loss) => loss,
            PyLoss::BCE(loss) => loss,
            PyLoss::CCE(loss) => loss,
        }
    }
}

#[pymodule]
fn neural_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySequential>()?;
    Ok(())
}