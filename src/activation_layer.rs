use ndarray::Array2;

use crate::layer::Layer;

pub struct ActivationLayer {
    input: Array2<f64>,
    activation: fn(&Array2<f64>) -> Array2<f64>,
    activation_prime: fn(&Array2<f64>) -> Array2<f64>
}

impl ActivationLayer{
    pub fn new(activation: fn(&Array2<f64>) -> Array2<f64>, activation_prime:fn(&Array2<f64>) -> Array2<f64>) -> ActivationLayer{
        ActivationLayer{
            input: Array2::default((1, 1)),
            activation,
            activation_prime
        }
    }
}

impl Layer for ActivationLayer {
    fn forward_propagation(&mut self, input: Array2<f64>) -> Result<Array2<f64>, String> {
        self.input = input;
        Ok((self.activation)(&self.input))
    }

    fn backward_propagation(
        &mut self,
        output_error: Array2<f64>,
        _learning_rate: f64,
    ) -> Result<Array2<f64>, String> {
        Ok((self.activation_prime)(&self.input) * output_error)
    }
}
