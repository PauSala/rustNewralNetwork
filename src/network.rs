use ndarray::Array2;

use crate::layer::Layer;

pub struct Network {
    pub loss: Box<dyn FnMut(&Array2<f64>, &Array2<f64>) -> f64>,
    pub loss_prime: Box<dyn FnMut(&Array2<f64>, &Array2<f64>) -> Array2<f64>>,
    pub layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input_data: Array2<f64>) -> Vec<Array2<f64>> {
        let mut result: Vec<Array2<f64>> = Vec::new();

        for item in input_data.rows().into_iter().by_ref() {
            let mut output: Array2<f64> = item.to_owned().insert_axis(ndarray::Axis(0));
            for layer in self.layers.iter_mut() {
                output = layer.forward_propagation(output);
            }
            result.push(output);
        }
        result
    }

    pub fn fit(
        &mut self,
        x_train: &Array2<f64>,
        y_train: &Array2<f64>,
        epochs: usize,
        learning_rate: f64,
    ) {
        for i in 0..epochs {
            let mut err: f64 = 0.0;

            for (index, item) in x_train.rows().into_iter().by_ref().enumerate() {
                let mut output = item.to_owned().insert_axis(ndarray::Axis(0));
                for layer in self.layers.iter_mut() {
                    output = layer.forward_propagation(output);
                }
                let y = y_train.row(index).to_owned().insert_axis(ndarray::Axis(0));
                err += (self.loss)(&y, &output);

                let mut error = (self.loss_prime)(&y, &output);

                for layer in self.layers.iter_mut().rev() {
                    error = layer.backward_propagation(error, learning_rate);
                }
            }

            err = err / (x_train.len() as f64);
            print!("\nEpoch: {} Error: {}", i, err);
        }

    }
}
