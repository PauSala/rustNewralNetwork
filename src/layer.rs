use ndarray::Array2;

pub trait Layer{
    fn forward_propagation(&mut self, input: Array2<f64>) -> Array2<f64>;
    fn backward_propagation(&mut self, output_error: Array2<f64>, learning_rate: f64) -> Array2<f64>;
}
