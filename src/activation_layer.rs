use ndarray::Array2;

use crate::layer::{Layer, LayerType, SerializedLayer};

pub struct ActivationLayer {
    input: Array2<f32>,
    activation: fn(&Array2<f32>) -> Array2<f32>,
    activation_prime: fn(&Array2<f32>) -> Array2<f32>
}

impl ActivationLayer{
    pub fn new(activation: fn(&Array2<f32>) -> Array2<f32>, activation_prime:fn(&Array2<f32>) -> Array2<f32>) -> ActivationLayer{
        ActivationLayer{
            input: Array2::default((1, 1)),
            activation,
            activation_prime
        }
    }
}

impl Layer for ActivationLayer {
    fn forward_propagation(&mut self, input: Array2<f32>) -> Result<Array2<f32>, String> {
        self.input = input;
        Ok((self.activation)(&self.input))
    }

    fn backward_propagation(
        &mut self,
        output_error: Array2<f32>,
        _learning_rate: f32,
    ) -> Result<Array2<f32>, String> {
        Ok((self.activation_prime)(&self.input) * output_error)
    }
    fn get_type(&mut self) -> crate::layer::LayerType {
        LayerType::ActivationLayer
    }
    fn to_serialized(&mut self) -> Option<SerializedLayer> {
        let serialized = SerializedLayer{
            layer_name: String::from("ActivationLayer"),
            weights: Vec::new(),
            bias: Vec::new()
        };
        Some(serialized)
    }
}
