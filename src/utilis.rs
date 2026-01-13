use ndarray::{Array2, s};

pub fn train_test_split(
    data: &Array2<f64>,
    labels: &Array2<f64>,
    test_ratio: f64,
) -> (Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>) {
    let n = data.shape()[0];
    let split_idx = ((1.0 - test_ratio) * n as f64) as usize;
    
    let train_data = data.slice(s![0..split_idx, ..]).to_owned();
    let test_data = data.slice(s![split_idx.., ..]).to_owned();
    let train_labels = labels.slice(s![0..split_idx, ..]).to_owned();
    let test_labels = labels.slice(s![split_idx.., ..]).to_owned();
    
    (train_data, test_data, train_labels, test_labels)
}

pub fn normalize(data: &mut Array2<f64>) {
    let mean = data.mean().unwrap();
    let std = data.std(0.0);
    *data = (data.clone() - mean) / (std + 1e-8);
}