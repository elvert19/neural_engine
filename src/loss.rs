use crate::tensor::Tensor;

pub trait Loss {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64;
    fn derivative(&self, predictions: &Tensor, targets: &Tensor) -> Tensor;
}

pub struct MSE;

impl Loss for MSE {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        let diff = predictions - targets;
        (&diff * &diff).mean().unwrap()
    }

    fn derivative(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let n = predictions.shape()[0] as f64;
        2.0 * (predictions - targets) / n
    }
}

pub struct BinaryCrossEntropy;

impl Loss for BinaryCrossEntropy {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        let epsilon = 1e-15;
        let pred_clipped = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
        let loss = targets * pred_clipped.mapv(|x| x.ln()) + 
                   (1.0 - targets) * pred_clipped.mapv(|x| (1.0 - x).ln());
        -loss.mean().unwrap()
    }

    fn derivative(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        let epsilon = 1e-15;
        let pred_clipped = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
        let n = predictions.shape()[0] as f64;
        (&pred_clipped - targets) / (&pred_clipped * (1.0 - &pred_clipped) * n)
    }
}

pub struct CategoricalCrossEntropy;

impl Loss for CategoricalCrossEntropy {
    fn compute(&self, predictions: &Tensor, targets: &Tensor) -> f64 {
        let epsilon = 1e-15;
        let clipped = predictions.mapv(|x| x.max(epsilon).min(1.0 - epsilon));
        // sum over classes, mean over samples
        -(targets * clipped.mapv(|x| x.ln()))
            .sum_axis(ndarray::Axis(1))
            .mean()
            .unwrap()
    }

    fn derivative(&self, predictions: &Tensor, targets: &Tensor) -> Tensor {
        // combined CCE + Softmax gradient = predicted - true
        // divide by batch size for correct scaling
        let n = predictions.shape()[0] as f64;
        (predictions - targets) / n
    }
}