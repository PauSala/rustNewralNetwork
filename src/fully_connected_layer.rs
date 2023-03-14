use std::ops::Add;

use ndarray::Array2;
use rand::Rng;

use crate::layer::Layer;

pub struct FullyConnectedLayer {
    weights: Array2<f64>,
    bias: Array2<f64>,
    input: Array2<f64>,
    //output: Array2<f64>,
}

impl FullyConnectedLayer {
    pub fn new(input_size: usize, output_size: usize) -> FullyConnectedLayer {
        let mut rng = rand::thread_rng();
        
        // Define the shape of the array
        let shape = (input_size, output_size);
        // Create a new array with the specified shape and fill it with random values
        let weights: Array2<f64> = Array2::from_shape_fn(shape, |_| (rng.gen::<f64>() - 0.5));
        // Define the shape of the array
        let shape = (1, output_size);
        let bias: Array2<f64> = Array2::from_shape_fn(shape, |_| rng.gen::<f64>() - 0.5);

        FullyConnectedLayer {
            weights,
            bias,
            input: Array2::default((1, 1)),
            //output: Array2::default((1, 1)),
        }
    }
}

impl Layer for FullyConnectedLayer {
    fn forward_propagation(&mut self, input: Array2<f64>) -> Result<Array2<f64>, String> {
     
        self.input = input;
        let mid = self.input.dot(&self.weights);
        Ok(mid.add(&self.bias))
    
    }
    fn backward_propagation(
        &mut self,
        output_error: Array2<f64>,
        learning_rate: f64,
    ) -> Result<Array2<f64>, String> {

        let input_error = output_error.dot(&self.weights.t());
        let weights_error = self.input.t().dot(&output_error);
        self.weights = &self.weights - (learning_rate * weights_error);
        self.bias = &self.bias - learning_rate * output_error;
        Ok(input_error)
    }
}
