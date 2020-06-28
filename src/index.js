import * as tf from '@tensorflow/tfjs';
import * as mobilenet from '@tensorflow-models/mobilenet';
import * as knnClassifier from '@tensorflow-models/knn-classifier'
import { VehicleEfficiency } from './models/VehicleEfficiency';
import { models } from '@tensorflow/tfjs';





// async function loadMobilenet() {
//   return await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json')
// }
// function loadImage(src) {
//   console.log(src)
//   return new Promise((resolve, reject) => {
//   const img = new Image();
//   img.src = src;
//   img.width = 1000;
//   img.height = 1000;
//     console.log(img)
//   img.onload = async () =>
//   console.log(img)

//      await tf.browser.fromPixels(img);
//     // img.onerror = (err) =>
//     //   reject(err);
//   });
// }

// function cropImage(img) {
//   const width = img.shape[0];
//   const height = img.shape[1];  // use the shorter side as the size to which we will crop
//   const shorterSide = Math.min(img.shape[0], img.shape[1]);  // calculate beginning and ending crop points
//   const startingHeight = (height - shorterSide) / 2;  const startingWidth = (width - shorterSide) / 2;  const endingHeight = startingHeight + shorterSide;  const endingWidth = startingWidth + shorterSide;  // return image data cropped to those points
//   return img.slice([startingWidth, startingHeight, 0], [endingWidth, endingHeight, 3]);
// }

// function resizeImage(image) {
//   return tf.image.resizeBilinear(image, [224, 224]);
// }

// function batchImage(image) {
//   // Expand our tensor to have an additional dimension, whose size is 1
//   const batchedImage = image.expandDims(0);  // Turn pixel data into a float between -1 and 1.
//   return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
// }

// function loadAndProcessImage(image) {
//   const croppedImage = cropImage(image);
//   const resizedImage = resizeImage(croppedImage);
//   const batchedImage = batchImage(resizedImage);
//   return batchedImage;
// }

// import drum from '../data/pretrained-model-data/pen.jpg';
// loadMobilenet().then(pretrainedModel => {
//     loadImage(drum).then(img => {
//       const processedImage = loadAndProcessImage(img);
//       const prediction = pretrainedModel.predict(processedImage);    // Because of the way Tensorflow.js works, you must call print on a Tensor instead of console.log.
//       prediction.as1D().argMax().print();;
//         });
// });

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
  document.getElementById('Paper_Mario').addEventListener('click', () => addExample(0));
  document.getElementById('Pokemon_Stadium_2').addEventListener('click', () => addExample(1));
  document.getElementById('Mario_64').addEventListener('click', () => addExample(2));
  document.getElementById('class-null').addEventListener('click', () => addExample(3));
  document.getElementById('save-model').addEventListener('click', async () =>{await   ('downloads://my-model')} )

  while (true) {
    if (classifier.getNumClasses() > 0) {
      const img = await webcam.capture();

      // Get the activation from mobilenet from the webcam.
      const activation = net.infer(img, 'conv_preds');
      // Get the most likely class and confidence from the classifier module.
      const result = await classifier.predictClass(activation);

      const classes = ['Paper Mario', 'Pokemon Stadium 2', 'Mario 64', 'Not A VideoGame'];
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



// // async function app() {
// //   try {
// //   console.log('Loading mobilenet...');

// //   //Load the model.
// //   net = await mobilenet.load();
// //   console.log('Successfully loaded model');

// //   // Create an object from Tensorflow.js data API which could capture image
// //   // from the web camera as Tensor.
// //   const webcam = await tf.data.webcam(webcamElement);


// //   //Reads an image from the webcam and associates it with a specific class index
// //   const addExample = async classId => {
// //     const img = await webcam.capture();

// //     // Get the intermediate activation of MobileNet 'conv_preds' and pass that
// //     // to the KNN classifier.
// //     const activation = net.infer(img, true);

// //     //Get the intermediate activation to the classifier.
// //     classifier.addExample(activation, classId)
// //     //console.log(classifier);
// //     // Dispose the tensor to release the memory.
// //     img.dispose();
// //   }
// //   // When clicking a button, add an example for that class.
// //   document.getElementById('class-a').addEventListener('click', () => addExample(0));
// //   document.getElementById('class-b').addEventListener('click', () => addExample(1));
// //   document.getElementById('class-c').addEventListener('click', () => addExample(2));

// //   while(true) {
// //     if (classifier.getNumClasses() > 0) {
// //       const img = await webcam.capture();

// //       const activation = net.infer(img, 'conv_preds');

// //       const result = await classifier.predictClass(activation);

// //       const classes = ['A', 'B', 'C'];
// //       document.getElementById('console').innerText = `
// //         prediction: ${classes[result.label]}\n
// //         probability: ${result.confidences[result.label]}
// //       `;

// //       // Dispose the tensor to release the memory
// //       img.dispose();// net = await mobilenet.load();
// //       // console.log('Successfully loaded model');

// //       // Give some breathing room by waiting for the next animation frame to
// //       // fire.

// //     }
// //     await tf.nextFrame();
// //   }
// //   //Make a prdiction through the model on our image.
// //   // const imgEl = document.getElementById('img');
// //   // const result = await net.classify(imgEl);
// //   // console.log(result);
// //   } catch (error) {
// //     console.log(error);
// //   }

// // }

app();
// console.log('Testing?')








// // // // async function run () {
// // // //   const MyModel = new VehicleEfficiency();

// // // //   await MyModel.init();

// // // //   // generate test input data
// // // //   const testInputs = tf.linspace(0, 1, 100);

// // // //   // make predictions
// // // //   MyModel.predict(testInputs);
// // // // }

// // // // document.addEventListener('DOMContentLoaded', run);

// // // const values  = [];

// // // for(let i = 0; i < 15; i++){
// // //   values[i] = Math.random() * 100
// // // }

// // // const shapeA = [5,3];
// // // const shapeB = [3,5];


// // // const a = tf.tensor2d(values, shapeA, "int32");
// // // const b = tf.tensor2d(values, shapeB, "int32");

// // // const c = a.matMul(b)

// // // //const vtense = tf.variable(tense);

// // // //data.print();
// // // //console.log(c.)
// // // //console.log(tense.get(2));

// // // // async function logTense(){
// // // //   console.log( await tense.data())
// // // // }

// // // // logTense();
// // // console.log(tf.memory().numTensors)


// // const model = tf.sequential();
// // const configHidden = {
// //   units: 4,
// //   inputShape: [2],
// //   activation: 'sigmoid'

// // }

// // const configOutput = {
// //   units: 1,
// //   activation: 'sigmoid'

// // }

// // const hidden = tf.layers.dense(configHidden);
// // const outputs = tf.layers.dense(configOutput);

// // model.add(hidden);
// // model.add(outputs);

// // const sdgOpt = tf.train.sgd(0.5)
// // const config = {
// //   optimizer: sdgOpt,
// //   loss: 'meanSquaredError'
// // }
// // model.compile(config);

// // const xs = tf.tensor2d([
// //   [0, 0],
// //   [0.5,0.5],
// //   [1,1]
// // ])


// // const ys = tf.tensor2d([
// //   [1],
// //   [0.5],
// //   [0]
// // ])



// // const train = async () => {
// //   for(let i =0; i < 300; i++){
// //     const response = await model.fit(xs, ys, {verbose:true, epochs: 5, shuffle:true})
// //     console.log(response.history.loss[0])
// //   }
// //   console.log('trainning complete')
// //   complete()
// // }
// // train()

// // const complete = () =>{
// //   let output = model.predict(xs);
// //   output.print()
// // }




