use ndarray::{Array2};

pub fn tanh(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|a| a.tanh())
}

pub fn tanh_prime(x: &Array2<f32>) -> Array2<f32>  {
    x.mapv(|a| 1.0 - (a.tanh().powi(2)))
}

pub fn mse(y_true: &Array2<f32>, y_pred: &Array2<f32>) -> f32 {
    let diff = y_true - y_pred;
    let square = diff.mapv(|x| x.powi(2));
    square.mean().unwrap()
}

pub fn mse_prime(y_true: &Array2<f32>, y_pred: &Array2<f32>) -> Array2<f32> {
    let diff = y_pred - y_true;
    let two = Array2::from_elem(diff.dim(), 2.0);
    diff * two / (y_true.len() as f32)
}

