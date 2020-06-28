console.log("Testing")
console.log('ml5 version:', ml5.version);

let mobilenet;
let classifier
let video;
let label = 'Loading Model';
let findBtn

//Helper functions
function modelReady() {
  console.log('Model is ready!!!');
  classifier.load('model.json', customModelReady);

}

function customModelReady () {
  console.log('Custom Model is ready!');
  label = 'Model Ready'

}

function videoReady() {
  console.log('Video is ready!!!');
  classifier.classify(gotResults);
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

//setup
function setup() {
  createCanvas(320, 270);
  video = createCapture(VIDEO);
  video.hide();
  background(0);

  mobilenet = ml5.featureExtractor('MobileNet',modelReady);
  classifier = mobilenet.classification(video, { numLabels:4 }, videoReady);///ADD OPTIONS HERE TO CHANGE LABEL SIZE!!!!

  findBtn = createButton('Find');
  findBtn.mousePressed(function(){
    console.log(label);
  });
}
//rendering
function draw() {
  background(0);
  image(video,0,0,320,240);
  fill(255);
  textSize(16);
  text(label, 10, height - 15);
}
