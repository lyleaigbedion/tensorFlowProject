import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier'
import { VehicleEfficiency } from './models/VehicleEfficiency';
import { models } from '@tensorflow/tfjs';

const webcamElement = document.getElementById('webcam');
let net;
const classifier = knnClassifier.create();


async function app() {
  console.log('Loading mobilenet..');

  // Load the model
  net = await mobilenet.load();
  console.log('Successfully loaded model');

  // Create an object from Tensorflow.js data API which could capture image
  // from the web camera as Tensor.
  const webcam = await tf.data.webcam(webcamElement);

  // Reads an image from the webcam and associates it with a specific class
  // index.
  const addExample = async classId => {
    // Capture an image from the web camera.
    const img = await webcam.capture();

    // Get the intermediate activation of MobileNet 'conv_preds' and pass that
    // to the KNN classifier.
    const activation = net.infer(img, true);

    // Pass the intermediate activation to the classifier.
    classifier.addExample(activation, classId);

    // Dispose the tensor to release the memory.
    img.dispose();
  };

  // When clicking a button, add an example for that class.
  document.getElementById('class-a').addEventListener('click', () => addExample(0));
  document.getElementById('class-b').addEventListener('click', () => addExample(1));
  document.getElementById('class-c').addEventListener('click', () => addExample(2));

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['A', 'B', 'C'];
      document.getElementById('console').innerText = `
        prediction: ${classes[result.label]}\n
        probability: ${result.confidences[result.label]}
      `;

      // Dispose the tensor to release the memory.
      img.dispose();
    }

    await tf.nextFrame();
  }
}



// async function app() {
//   try {
//   console.log('Loading mobilenet...');

//   //Load the model.
//   net = await mobilenet.load();
//   console.log('Successfully loaded model');

//   // Create an object from Tensorflow.js data API which could capture image
//   // from the web camera as Tensor.
//   const webcam = await tf.data.webcam(webcamElement);


//   //Reads an image from the webcam and associates it with a specific class index
//   const addExample = async classId => {
//     const img = await webcam.capture();

//     // Get the intermediate activation of MobileNet 'conv_preds' and pass that
//     // to the KNN classifier.
//     const activation = net.infer(img, true);

//     //Get the intermediate activation to the classifier.
//     classifier.addExample(activation, classId)
//     //console.log(classifier);
//     // Dispose the tensor to release the memory.
//     img.dispose();
//   }
//   // When clicking a button, add an example for that class.
//   document.getElementById('class-a').addEventListener('click', () => addExample(0));
//   document.getElementById('class-b').addEventListener('click', () => addExample(1));
//   document.getElementById('class-c').addEventListener('click', () => addExample(2));

//   while(true) {
//     if (classifier.getNumClasses() > 0) {
//       const img = await webcam.capture();

//       const activation = net.infer(img, 'conv_preds');

//       const result = await classifier.predictClass(activation);

//       const classes = ['A', 'B', 'C'];
//       document.getElementById('console').innerText = `
//         prediction: ${classes[result.label]}\n
//         probability: ${result.confidences[result.label]}
//       `;

//       // Dispose the tensor to release the memory
//       img.dispose();// net = await mobilenet.load();
//       // console.log('Successfully loaded model');

//       // Give some breathing room by waiting for the next animation frame to
//       // fire.

//     }
//     await tf.nextFrame();
//   }
//   //Make a prdiction through the model on our image.
//   // const imgEl = document.getElementById('img');
//   // const result = await net.classify(imgEl);
//   // console.log(result);
//   } catch (error) {
//     console.log(error);
//   }

// }

app();
console.log('Testing?')








// // // async function run () {
// // //   const MyModel = new VehicleEfficiency();

// // //   await MyModel.init();

// // //   // generate test input data
// // //   const testInputs = tf.linspace(0, 1, 100);

// // //   // make predictions
// // //   MyModel.predict(testInputs);
// // // }

// // // document.addEventListener('DOMContentLoaded', run);

// // const values  = [];

// // for(let i = 0; i < 15; i++){
// //   values[i] = Math.random() * 100
// // }

// // const shapeA = [5,3];
// // const shapeB = [3,5];


// // const a = tf.tensor2d(values, shapeA, "int32");
// // const b = tf.tensor2d(values, shapeB, "int32");

// // const c = a.matMul(b)

// // //const vtense = tf.variable(tense);

// // //data.print();
// // //console.log(c.)
// // //console.log(tense.get(2));

// // // async function logTense(){
// // //   console.log( await tense.data())
// // // }

// // // logTense();
// // console.log(tf.memory().numTensors)


// const model = tf.sequential();
// const configHidden = {
//   units: 4,
//   inputShape: [2],
//   activation: 'sigmoid'

// }

// const configOutput = {
//   units: 1,
//   activation: 'sigmoid'

// }

// const hidden = tf.layers.dense(configHidden);
// const outputs = tf.layers.dense(configOutput);

// model.add(hidden);
// model.add(outputs);

// const sdgOpt = tf.train.sgd(0.5)
// const config = {
//   optimizer: sdgOpt,
//   loss: 'meanSquaredError'
// }
// model.compile(config);

// const xs = tf.tensor2d([
//   [0, 0],
//   [0.5,0.5],
//   [1,1]
// ])


// const ys = tf.tensor2d([
//   [1],
//   [0.5],
//   [0]
// ])



// const train = async () => {
//   for(let i =0; i < 300; i++){
//     const response = await model.fit(xs, ys, {verbose:true, epochs: 5, shuffle:true})
//     console.log(response.history.loss[0])
//   }
//   console.log('trainning complete')
//   complete()
// }
// train()

// const complete = () =>{
//   let output = model.predict(xs);
//   output.print()
// }




