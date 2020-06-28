// import * as tf from '@tensorflow/tfjs';
// import * as mobilenet from '@tensorflow-models/mobilenet';
// import * as knnClassifier from '@tensorflow-models/knn-classifier'
// import { VehicleEfficiency } from './models/VehicleEfficiency';
// import { models } from '@tensorflow/tfjs';

console.log('ml5 version:', ml5.version);

let mobilenet;
let classifier
let video;
let label = '';
let mario64btn;
let paperMariobtn;
let pokemonSta2btn;
let notAGamebtn;
let trainBtn;
let saveBtn;

function whileTraining(loss) {
  if(!loss){
    console.log('Training Complete')
    classifier.classify(gotResults);

  }else{
    console.log(loss)
  }

}

function modelReady() {
  console.log('Model is ready!!!');

}

function videoReady() {
  console.log('Video is ready!!!');

}

function gotResults(error, results){
  if (error) {
    console.error(error)
  } else {

    label = results[0].label
    console.log(results)
    //let prob = results[0].confidence;
    classifier.classify(gotResults)
  }
}


function setup() {
  createCanvas(320, 270);
  video = createCapture(VIDEO);
  video.hide();
  background(0);

  mobilenet = ml5.featureExtractor('MobileNet',modelReady);
  classifier = mobilenet.classification(video, { numLabels:4 });///ADD OPTIONS HERE TO CHANGE LABEL SIZE!!!!

  mario64btn = createButton('Mario 64');
  mario64btn.mousePressed(function(){
    classifier.addImage('Mario 64');
  });


  paperMariobtn = createButton('Paper Mario');
  paperMariobtn.mousePressed(function(){
    classifier.addImage('Paper Mario');
  });

  pokemonSta2btn = createButton('Pokemon Stadium 2');
  pokemonSta2btn.mousePressed(function(){
    classifier.addImage('Pokemon Stadium 2');
  });

  notAGamebtn = createButton('Not a game');
  notAGamebtn.mousePressed(function(){
    classifier.addImage('Not a game');
  });

  trainBtn = createButton('Train Model');
  trainBtn.mousePressed(function(){
    classifier.train(whileTraining);
  })

  saveBtn = createButton('Save Model');
  saveBtn.mousePressed(function(){
    classifier.save();
  })


}

function draw() {
  background(0);
  image(video,0,0,320,240);
  fill(255);
  textSize(16);
  text(label, 10, height - 20);
}
