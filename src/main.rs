use rand::Rng;
use std::io;

const TRAINING_SET_SIZE: usize = 5;
const TESTING_SET_SIZE: usize = 5;
const LEARNING_RATE: f64 = 0.1;
const EPOCHS: usize = 10;

fn main() {
    loop {
        let mut rng = rand::thread_rng();


        // Initialize weights and biases
        let mut weights_ih: Vec<f64> = (0..4).map(|_| rng.gen_range(0.0..1.0)).collect(); // 2x2
        let mut weights_ho: Vec<f64> = (0..2).map(|_| rng.gen_range(0.0..1.0)).collect(); // 2x1
        let mut bias_h: Vec<f64> = (0..2).map(|_| rng.gen_range(0.0..1.0)).collect(); // 1x2
        let mut bias_o = rng.gen_range(0.0..1.0); // 1x1

        // Inputs and target
        let train_array: Vec<Vec<f64>> = (0..TRAINING_SET_SIZE).map(|_| vec![rng.gen_range(-2.5..2.5),rng.gen_range(-2.5..2.5)]).collect();
        let train_target: Vec<f64> = (0..TRAINING_SET_SIZE).map(|i| 0.5 * train_array[i][0] + 1.5 * train_array[i][1]).collect();

        


        for _ in 0..EPOCHS { // Training for EPOCHS epochs
            for i in 0..train_array.len() {
                let inputs = &train_array[i];
                let target = train_target[i];

                // Forward pass _________________________________________________________
                let hidden_input: Vec<f64> = mat_vec_mul(&weights_ih, &inputs, &bias_h);
                let hidden_output: Vec<f64> = hidden_input.iter().cloned().collect();
                let final_input = dot_product(&weights_ho, &hidden_output) + bias_o;
                let final_output = final_input;

                // Compute the error
                let error = target - final_output;

                // Backward pass ________________________________________________________

                // Output to hidden
                let d_weights_ho: Vec<f64> = hidden_output.iter().map(|&x| LEARNING_RATE * error * x).collect();
                let d_bias_o = LEARNING_RATE * error;

                // Hidden to input
                let hidden_errors: Vec<f64> = weights_ho.iter().map(|&x| x * error).collect();
                let d_weights_ih: Vec<f64> = (0..4).map(|i| LEARNING_RATE * hidden_errors[i/2] * inputs[i%2]).collect();
                let d_bias_h: Vec<f64> = hidden_errors.iter().map(|&x| LEARNING_RATE * x).collect();

                // Update weights and biases
                update_vec(&mut weights_ho, &d_weights_ho);
                bias_o += d_bias_o;
                update_vec(&mut weights_ih, &d_weights_ih);
                update_vec(&mut bias_h, &d_bias_h);
        }
    }

        // Final output after training
        let test_array: Vec<Vec<f64>> = (0..TESTING_SET_SIZE).map(|_| vec![rng.gen_range(-2.5..2.5),rng.gen_range(-2.5..2.5)]).collect();
        let test_target: Vec<f64> = (0..TESTING_SET_SIZE).map(|i| 0.5 * test_array[i][0] + 1.5 * test_array[i][1]).collect();

        let mut train_total_error: f64 = 0.0;
        let mut test_total_error: f64 = 0.0;

        println!("-----------------Training set----------------");
        for i in 0..train_array.len() {
            let inputs = &train_array[i];
            let target = train_target[i];
            let final_output = feed_forward(&weights_ih, &weights_ho, &bias_h, bias_o, &inputs);
            if i < test_target.len() {
                println!("Target: {:.5}, Output: {:.5}", target, final_output);
            }
            train_total_error += (target - final_output).powi(2);
        }

        println!("-----------------Testing set-----------------");
        for i in 0..test_array.len() {
            let inputs = &test_array[i];
            let target = test_target[i];
            let final_output = feed_forward(&weights_ih, &weights_ho, &bias_h, bias_o, &inputs);
            println!("Target: {:.5}, Output: {:.5}", target, final_output);
            test_total_error += (target - final_output).powi(2);
        }

        println!("-------------------Info set------------------");
        let train_mse = train_total_error / train_target.len() as f64;
        let test_mse = test_total_error / test_target.len() as f64;
        println!("Training set count: {}", TRAINING_SET_SIZE);
        println!("Testing set count: {}", TESTING_SET_SIZE);
        println!("Learning rate: {}", LEARNING_RATE);
        println!("Epochs: {}", EPOCHS);
        println!("Training MSE: {:.5}", train_mse);
        println!("Testing MSE: {:.5}", test_mse);

        println!("-------------Press Enter to Retry-------------");
        let mut input = String::new();
        io::stdin().read_line(&mut input).unwrap();
        print!("{}[2J", 27 as char);
    }
}

fn mat_vec_mul(matrix: &Vec<f64>, vector: &Vec<f64>, bias: &Vec<f64>) -> Vec<f64> {
    let mut result = vec![0.0; bias.len()];
    for i in 0..result.len() {
        result[i] = matrix[i*2] * vector[0] + matrix[i*2 + 1] * vector[1] + bias[i];
    }
    result
}

fn dot_product(vec1: &Vec<f64>, vec2: &Vec<f64>) -> f64 {
    vec1.iter().zip(vec2.iter()).map(|(&x, &y)| x * y).sum()
}

fn update_vec(vec: &mut Vec<f64>, delta: &Vec<f64>) {
    for (v, d) in vec.iter_mut().zip(delta.iter()) {
        *v += *d;
    }
}

fn feed_forward(weights_ih: &Vec<f64>, weights_ho: &Vec<f64>, bias_h: &Vec<f64>, bias_o: f64, inputs: &Vec<f64>) -> f64 {
    let hidden_input = mat_vec_mul(&weights_ih, &inputs, &bias_h);
    let hidden_output: Vec<f64> = hidden_input.iter().cloned().collect();
    let final_input = dot_product(&weights_ho, &hidden_output) + bias_o;
    final_input
}