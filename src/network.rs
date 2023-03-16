use crate::layer::{Layer, SerializedLayer};
use ndarray::Array2;
use serde::{ser::SerializeStruct, Serialize, Serializer};
use std::{fs::File, io::Write};

pub struct Network {
    pub loss: fn(&Array2<f32>, &Array2<f32>) -> f32,
    pub loss_prime: fn(&Array2<f32>, &Array2<f32>) -> Array2<f32>,
    pub layers: Vec<Box<dyn Layer>>,
}

impl Network {
    pub fn add(&mut self, layer: Box<dyn Layer>) {
        self.layers.push(layer);
    }

    pub fn predict(&mut self, input_data: Array2<f32>) -> Option<Vec<Array2<f32>>> {
        let mut result: Vec<Array2<f32>> = Vec::new();

        for item in input_data.rows().into_iter().by_ref() {
            let mut output: Array2<f32> = item.to_owned().insert_axis(ndarray::Axis(0));
            for layer in self.layers.iter_mut() {
                match layer.forward_propagation(output) {
                    Ok(data) => output = data,
                    Err(e) => {
                        print!("Error occurred: {}", e);
                        return None;
                    }
                }
            }
            result.push(output);
        }
        Some(result)
    }

    pub fn fit(
        &mut self,
        x_train: &Array2<f32>,
        y_train: &Array2<f32>,
        epochs: usize,
        learning_rate: f32,
    ) -> Option<()> {
        for i in 0..epochs {
            let mut err: f32 = 0.0;

            for (index, item) in x_train.rows().into_iter().enumerate() {
                let mut output = item.to_owned().insert_axis(ndarray::Axis(0));
                for layer in self.layers.iter_mut() {
                    match layer.forward_propagation(output) {
                        Ok(data) => output = data,
                        Err(e) => {
                            print!("Error occurred: {}", e);
                            return None;
                        }
                    }
                }
                let y = y_train.row(index).to_owned().insert_axis(ndarray::Axis(0));
                err += (self.loss)(&y, &output);

                let mut error = (self.loss_prime)(&y, &output);

                for layer in self.layers.iter_mut().rev() {
                    match layer.backward_propagation(error, learning_rate) {
                        Ok(data) => error = data,
                        Err(e) => {
                            print!("Error occurred: {}", e);
                            return None;
                        }
                    }
                }
            }

            err = err / (x_train.len() as f32);
            print!("\nEpoch: {} Error: {}", i, err);
        }
        Some(())
    }

    pub fn serialize(&mut self) {
        let mut serialized_layers: Vec<SerializedLayer> = Vec::new();
        for layer in self.layers.iter_mut() {
            let data = layer.to_serialized();
            serialized_layers.push(data.unwrap());
        }
        let serialized_model = SerializedModel {
            layers: serialized_layers,
        };
        let data = serde_json::to_string(&serialized_model);
        let mut f = File::create("model.json").expect("Unable to create file");
        f.write(data.unwrap().as_bytes())
            .expect("Unable to write data");
    }
}

struct SerializedModel {
    layers: Vec<SerializedLayer>,
}

impl Serialize for SerializedModel {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 3 is the number of fields in the struct.
        let mut state = serializer.serialize_struct("SerializedModel", 1)?;
        state.serialize_field("Layers", &self.layers)?;
        state.end()
    }
}
