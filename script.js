// TODO: Save the model to the local storage so that it can be resused later.
// TODO: get the user drawn input from the canvas and use it to predict the digit.

import { TRAINING_DATA } from "https://storage.googleapis.com/jmstore/TensorFlowJS/EdX/TrainingData/fashion-mnist.js";

//to see the status of loading tensorflow library
const TensorImportStatus = document.getElementById("status");
TensorImportStatus.innerText =
  "Loaded TensorFlow.js - version: " + tf.version.tfjs;
TensorImportStatus.classList.remove("loading");
TensorImportStatus.classList.add("ready");

var interval = 2000;
const RANGER = document.getElementById("ranger");
const DOM_SPEED = document.getElementById("domSpeed");

RANGER.addEventListener("input", function () {
  interval = 2000 / this.value;
  DOM_SPEED.innerText =
    "Change interval between classifications in Currently " + this.value + "ms";
});

// map the output to the actual label
const LOOKUP = [
  "T-shirt",
  "Trouser",
  "Pullover",
  "Dress",
  "Coat",
  "Sandal",
  "Shirt",
  "Sneaker",
  "Bag",
  "Ankle boot",
];
function normalize(tensor, min, max) {
  const result = tf.tidy(() => {
    const MIN_VALUES = tf.scalar(min);
    const MAX_VALUES = tf.scalar(max);
    const TENSOR_SUBTRACT_MIN_VALUE = tf.sub(tensor, MIN_VALUES);
    const RANGE_SIZE = tf.sub(MAX_VALUES, MIN_VALUES);
    const NORMALIZED_VALUES = tf.div(TENSOR_SUBTRACT_MIN_VALUE, RANGE_SIZE);
    return NORMALIZED_VALUES;
  });
  return result;
}
//get the reference to the  mnist input
const inputs = TRAINING_DATA.inputs;
//get the reference to the  mnist input values (pixel data).
const outputs = TRAINING_DATA.outputs;

tf.util.shuffleCombo(inputs, outputs);

// input feature Array is 2 dimensional
const inputs_tensor = normalize(tf.tensor2d(inputs), 0, 255);

// output feature Array is 1 dimensional
const outputs_tensor = tf.oneHot(tf.tensor1d(outputs, "int32"), 10);

// we define the model architecture

const model = tf.sequential();

model.add(
  tf.layers.conv2d({
    inputShape: [28, 28, 1],
    kernelSize: 3,
    filters: 16,
    strides: 1,
    padding: "same",
    activation: "relu",
  })
);

model.add(
  tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2,
  })
);

model.add(
  tf.layers.conv2d({
    kernelSize: 3,
    filters: 32,
    strides: 1,
    padding: "same",
    activation: "relu",
  })
);

model.add(
  tf.layers.maxPooling2d({
    poolSize: 2,
    strides: 2,
  })
);

model.add(tf.layers.flatten());
model.add(tf.layers.dense({ units: 128, activation: "relu" }));
model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

model.summary();
const PREDICTION_ELEMENT = document.getElementById("prediction");

train();

async function train() {
  // compile the model with the defined optimizer and loss function to use

  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });

  const RESHAPED_INPUTS = inputs_tensor.reshape([inputs.length, 28, 28, 1]);

  let results = await model.fit(RESHAPED_INPUTS, outputs_tensor, {
    shuffle: true, // Ensure the data is shuffled before each epoch
    batchSize: 256, // 256 samples per batch
    epochs: 30, // go over the data 30 times
    validationSplit: 0.15,
    callbacks: {
      onEpochEnd: async (epoch, logs) => {
        console.log(
          "Epoch: " + epoch + " Loss: " + logs.loss + " Accuracy: " + logs.acc
        );
      },
    },
  });

  RESHAPED_INPUTS.dispose();
  outputs_tensor.dispose();
  inputs_tensor.dispose();
  evaluate(); // evaluate the model once training is done
}

function evaluate() {
  const OFFSET = Math.floor(Math.random() * inputs.length);

  let answer = tf.tidy(() => {
    // let newInput = tf.tensor1d(inputs[OFFSET]).expandDims();
    let newInput = normalize(tf.tensor1d(inputs[OFFSET]), 0, 255);
    let output = model.predict(newInput.reshape([1, 28, 28, 1]));
    output.print();
    return output.squeeze().argMax();
  });

  answer.array().then((index) => {
    PREDICTION_ELEMENT.innerText = "Prediction: " + LOOKUP[index];
    PREDICTION_ELEMENT.setAttribute(
      "class",
      index == outputs[OFFSET] ? "correct" : "wrong"
    );
    answer.dispose();
    drawImage(inputs[OFFSET]);
  });
}

const CANVAS = document.getElementById("canvas");
const CTX = CANVAS.getContext("2d");

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
