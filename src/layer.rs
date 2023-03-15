use ndarray::Array2;
use serde::{Serialize, Serializer, ser::SerializeStruct};

pub trait Layer{
    fn forward_propagation(&mut self, input: Array2<f64>) -> Result<Array2<f64>, String>;
    fn backward_propagation(&mut self, output_error: Array2<f64>, learning_rate: f64) -> Result<Array2<f64>, String>;
    fn get_type(&mut self) -> LayerType;
    fn to_serialized(&mut self) -> Option<SerializedLayer>;
}

pub enum LayerType {
    ActivationLayer,
    FullyConnectedLayer,
}

pub struct SerializedLayer {
    pub layer_name: String, 
    pub weights: Vec<f64>,
    pub bias: Vec<f64>,
}

impl Serialize for SerializedLayer {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        // 3 is the number of fields in the struct.
        let mut state = serializer.serialize_struct("SerializedLayer", 3)?;
        state.serialize_field("Type", &self.layer_name)?;
        state.serialize_field("weigths", &self.weights)?;
        state.serialize_field("bias", &self.bias)?;
        state.end()
    }
}
