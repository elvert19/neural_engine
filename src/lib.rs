pub mod tensor;
pub mod layer;
pub mod activation;
pub mod loss;
pub mod optimizer;
pub mod utilis;

use layer:: Layer;
use optimizer::Optimizer;
use loss::Loss;
use ndarray::Array2;

pub struct Sequential {
    layers: Vec <Box<dyn Layer>>,

    
}

impl Sequential  {
    pub  fn new () -> Self {
        self  { layers: Vec ::new ()}
    }

    pub fn add (&mut self, layer : impl Layer + 'static ) {
        self.layers.push (Box::new(layer));

    }

    pub fn forward (&mut self, input : Array2<f64>) -> Array2<f64> {
        let mut current_output  = input;
        for layer in &mut self.layers {
            current_output = layer.forward (current_output);

        }
        current_output
    }




    pub fn train (
        &mut self,
        inputs: & Array2<f64>,
        targets:&Array2<f64>,
        epochs: usize,
        loss_fn: &dyn Loss,
        optimizer: & mut dyn Optimizer,


    ) -> Vec <f64> {
        let mut loss_history =Vec:: with_capacity(epochs);

        for epoch in 0.. epochs {
            
            let output = self.forward(inputs.clone() );

            let cost = loss_fn.compute(&output, targets);
            loss_history.push(cost);


            let mut gradient  = loss_fn.derivative(&output, targets);

            for layer  in self.layers.iter_mut().rev(){
                gradient =layer.backward(gradient);



            }

            optimizer.update (&mut self.layers);

            if epoch %100 ==0 {
                println!("Epoch{}:Loss = {:4}",epoch, cost );
            }
        }
        loss_history
    }


    pub fn predict (&self , input : &Array2<f64>) -> Array2<f64> {

        unimplemented!("Add inference -only forward pass")
    }
} impl Default for Sequential {
    fn default () -> self {
        self::new()
    }
}