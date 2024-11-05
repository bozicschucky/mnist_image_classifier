
//to see the status of loading tensorflow library
 const TensorImportStatus = document.getElementById('status');
 TensorImportStatus.innerText = 'Loaded TensorFlow.js - version: ' + tf.version.tfjs;
 TensorImportStatus.classList.remove('loading');
TensorImportStatus.classList.add('ready');
 

// TODO: Save the model to the local storage so that it can be resused later.
// TODO: get the user drawn input from the canvas and use it to predict the digit.


import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/mnist.js";

//get the reference to the  mnist input values (pixel data).
const inputs = TRAINING_DATA.inputs;
//get the reference to the  mnist input values (pixel data).
const outputs = TRAINING_DATA.outputs;

tf.util.shuffleCombo(inputs, outputs);

// input feature Array is 2 dimensional
const inputs_tensor = tf.tensor2d(inputs);

// output feature Array is 1 dimensional
const outputs_tensor = tf.oneHot(tf.tensor1d(outputs, 'int32'), 10);


// we define the model architecture

const model = tf.sequential();

model.add(tf.layers.dense({
    inputShape: [784],
    units: 32,
    activation: 'relu'
}));

model.add(
    tf.layers.dense({
        units: 16,
        activation: 'relu'
    })
)

model.add(
    tf.layers.dense({
        units: 10,
        activation: 'softmax'
    })
)

model.summary();
const PREDICTION_ELEMENT = document.getElementById('prediction');

train();


async function train() { 

  // compile the model with the defined optimizer and loss function to use

  model.compile({
    optimizer: 'adam',
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  let results = await model.fit(
    inputs_tensor,
    outputs_tensor,
    {
      shuffle:true,  // Ensure the data is shuffled before each epoch
      batchSize: 512, // update weights after 512 samples
      epochs: 50,  // go over the data 50 times
      validationSplit: 0.2,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          console.log('Epoch: ' + epoch + ' Loss: ' + logs.loss + ' Accuracy: ' + logs.acc);
        }
      }
    }
  )

  outputs_tensor.dispose();
  inputs_tensor.dispose();
  evaluate();  // evaluate the model once training is done

}


function evaluate() {
  const OFFSET = Math.floor((Math.random() * inputs.length));

  let answer = tf.tidy(() => {

    let newInput = tf.tensor1d(inputs[OFFSET]).expandDims();

    let output = model.predict(newInput);
    output.print();
    return output.squeeze().argMax();
  });
  
  answer.array().then(index => {
    PREDICTION_ELEMENT.innerText = 'Prediction: ' + index;
    PREDICTION_ELEMENT.setAttribute('class', (index == outputs[OFFSET]) ? 'correct' : 'wrong');
    answer.dispose();
    drawImage(inputs[OFFSET]);
  })
}

const CANVAS = document.getElementById('canvas');
const CTX = CANVAS.getContext('2d');

function drawImage(digit) {
  var imgData = CTX.getImageData(0, 0, 28, 28);
  
  for (let i = 0; i < digit.length; i++) {
    imgData.data[i * 4] = digit[i] * 255; // RED channel
    imgData.data[i * 4 + 1] = digit[i] * 255; // GREEN channel
    imgData.data[i * 4 + 2] = digit[i] * 255; // BLUE channel
    imgData.data[i * 4 + 3] = 255; // ALPHA channel
  }

  // render the update array of data to the canvas itself
  CTX.putImageData(imgData, 0, 0);

  // perform a new classification after a certain interval
  setTimeout(evaluate, 2000);
}

