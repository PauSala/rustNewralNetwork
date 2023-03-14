use mnist::*;
use ndarray::{s, Array2};
use newral_network::{
    activation_layer::ActivationLayer,
    fully_connected_layer::FullyConnectedLayer,
    network::Network,
    rnumpy::{mse, mse_prime, tanh, tanh_prime},
};

fn main() {
    // Deconstruct the returned Mnist struct.
    let Mnist {
        trn_img,
        trn_lbl,
        tst_img,
        tst_lbl,
        ..
    } = MnistBuilder::new()
        .label_format_one_hot()
        .training_set_length(50_000)
        .validation_set_length(10_000)
        .test_set_length(10_000)
        .finalize();

    let image_num = 0;
    // Can use an Array2 or Array3 here (Array3 for visualization)
    let train_data = Array2::from_shape_vec((50_000, 28 * 28), trn_img)
        .expect("Error converting images to Array3 struct")
        .map(|x| *x as f64 / 256.0);

    // Convert the returned Mnist struct to Array2 format
    let train_labels: Array2<f64> = Array2::from_shape_vec((50_000, 10), trn_lbl)
        .expect("Error converting training labels to Array2 struct")
        .map(|x| *x as f64);

    let _test_data = Array2::from_shape_vec((10_000, 28 * 28), tst_img)
        .expect("Error converting images to Array2 struct")
        .map(|x| *x as f64 / 256.0);

    let _test_labels: Array2<f64> = Array2::from_shape_vec((10_000, 10), tst_lbl)
        .expect("Error converting testing labels to Array2 struct")
        .map(|x| *x as f64);

    //Build network
    let layer = FullyConnectedLayer::new(28 * 8, 40);
    let second_layer = ActivationLayer::new(tanh, tanh_prime);
    let third_layer = FullyConnectedLayer::new(40, 20);
    let fourth_layer = ActivationLayer::new(tanh, tanh_prime);
    let fiveth_layer = FullyConnectedLayer::new(20, 10);
    let sixth_layer = ActivationLayer::new(tanh, tanh_prime);

    let loss = mse;
    let loss_prime = mse_prime;

    let mut network = Network {
        loss,
        loss_prime,
        layers: Vec::new(),
    };
    network.add(Box::new(layer));
    network.add(Box::new(second_layer));
    network.add(Box::new(third_layer));
    network.add(Box::new(fourth_layer));
    network.add(Box::new(fiveth_layer));
    network.add(Box::new(sixth_layer));

    let fit = network.fit(&train_data, &train_labels, 50, 0.01);
    if fit.is_some() {
        match network.predict(_test_data) {

            Some(result) => {
                print!("\nResult: {:#.1}", result[0]);
                print!("\nResult: {:#.1}", result[1]);
                print!("\nResult: {:#.1}", result[2]);
                print!("\nResult: {:#.1}", result[3]);
                print!("\nResult: {:#.1}", result[4]);
                println!("\n{:#.1?}\n", _test_labels.slice(s![image_num..5, ..]));
            }
            None => print!("\nSome error occured"),
        }
    }
}
