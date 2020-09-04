//todo: create vars
//* VARS
let x_vals = []; // array of values - not a tensor
let y_vals = []; // array of values - not a tensor
let a = []; // ax^n tf trainable variables
let polyDegree; // degree of poly
let optimizer; // optimizer
let learningRate; // learning rate

function setup() {
  createCanvas(400, 400);
  background(0);


  // add slider for degree of polynom
  text1 = createP('degree of polynom');
  text1.position(width / 2 - 60, height);

  slider = createSlider(0, 10, 3, 1);
  slider.position(width / 2 - 65, height + 40);
  polyDegree = slider.value();

  text2 = createP(slider.value());
  text2.position(width / 2, height + 50);

  // add select for optimizer selection
  selector = createSelect();
  selector.position(width / 2, height + 90);
  selector.option('sgd');
  selector.option('adam');
  selector.option('adagrad');
  selector.option('adadelta');
  selector.option('adamax');
  //selector.option('rmsprop');

  selector.selected('sgd');
  selector.changed(changeOptimizer);

  text3 = createP('select optimizer');
  text3.position(width / 3 - 50, height + 75);

  // add slider for learning rate
  selector1 = createSelect();
  selector1.position(width / 2, height + 115);
  selector1.option(2);
  selector1.option(0.8);
  selector1.option(0.5);
  selector1.option(0.3);
  selector1.option(0.1);
  selector1.option(0.03);
  selector1.option(0.01);
  selector1.option(0.003);
  selector1.option(0.001);
  selector1.selected(0.03);
  selector1.changed(changeLR);

  text4 = createP('learning rate');
  text4.position(width / 3 - 50, height + 100);

  polynomCreation();
  changeLR();
  changeOptimizer();

}

function polynomCreation() {
  // reset array coefficient
  a = [];
  // tf.variable
  // a[i] are trainable!!!
  // create new array coefficient
  for (let i = 0; i <= polyDegree; i++) {
    a[i] = tf.variable(tf.scalar(random(-1, 1)));
  }

}

// todo: - 1 - create a datase with mouse
// todo: - 2 - loss function MSE
// todo: - 3 - optimizer
// todo: - 4 - predict function

//* CREATE DATASET
function mouseClicked() {
  // if it's on canvas
  if ((mouseX <= width) && (mouseY <= height)) {
    // create the data
    let x = map(mouseX, 0, width, -1, 1);
    let y = map(mouseY, 0, height, 1, -1);
    x_vals.push(x);
    y_vals.push(y);
  }
}

//* LOSS
function loss(predictions, labels) {
  // mean squared error
  // (predictions - labels)^2
  return predictions.sub(labels).square().mean();
}

//* OPTIMIZER

function changeLR() {
  learningRate = selector1.value()
}
//const learningRate = 0.03;


// stochastic gradient descent
//const optimizer = tf.train.sgd(learningRate); 
// adam
//optimizer = tf.train.adam(learningRate);


function changeOptimizer() {
  if (selector.value() == 'sgd') {
    optimizer = tf.train.sgd(learningRate);
  } else if (selector.value() == 'adam') {
    optimizer = tf.train.adam(learningRate);
  } else if (selector.value() == 'adagrad') {
    optimizer = tf.train.adagrad(learningRate);
  } else if (selector.value() == 'adadelta') {
    optimizer = tf.train.adadelta(learningRate);
  } else if (selector.value() == 'adamax') {
    optimizer = tf.train.adamax(learningRate);
  } //else if (selector.value() == 'rmsprop') {
  //   optimizer = tf.train.rmsprop(learningRate);
  // } // RMS PROP DOESN;T WORK

}

//* PREDICT
function predict(x) {

  //* CHANGE
  const xs = tf.tensor1d(x); // tensor xs
  //     y = a[n]*x^n + ... + a[1]*x^1 + a[0]x^0 -- polynomial function

  let ys = tf.tensor(0);

  for (let i = 0; i <= polyDegree; i++) {
    //cast the exponent in tensor format
    let exponent = tf.tensor(i).toInt();
    // add element by element
    ys = ys.add(xs.pow(exponent).mul(a[i]));
  }
  return ys;
}

function updatePolynomDegree() {

  polyDegree = slider.value();
  polynomCreation();
}

//* DRAW LOOP
function draw() {
  background(0);

  //update the slider
  text2.html(slider.value());

  // recreate function if degree has changed
  if (slider.value() != polyDegree) {
    //take new degree
    polyDegree = slider.value();
    // recreatefunction
    polynomCreation();
  }
  // draw the point from the dataset
  stroke(255);
  strokeWeight(8);
  for (let i = 0; i < x_vals.length; i++) {
    let px = map(x_vals[i], -1, 1, 0, width);
    let py = map(y_vals[i], 1, -1, 0, height);
    point(px, py);
  }

  //* TRAIN
  tf.tidy(() => {
    //* tidy to clear the memory

    if (x_vals.length > 0) {
      // don't train if there are no values

      const ys = tf.tensor1d(y_vals); // tensor to cleanup

      // cleanup also this tensor below
      optimizer.minimize(() => loss(predict(x_vals), ys));
      //* this part above is where the learning happens in tfjs
    }

    // draw the line on screen

    // I have the xs and just need to predict the ys
    const curveX = [];

    for (let x = -1; x < 1; x += 0.05) {
      curveX.push(x);
    }
    const ys = predict(curveX); // predict creates tensors

    // take the values from the tensor
    let curveY = ys.dataSync();

    // draw the curve
    beginShape();
    noFill();
    stroke(255);
    strokeWeight(2);
    for (let i = 0; i < curveX.length; i++) {
      let x = map(curveX[i], -1, 1, 0, width);
      let y = map(curveY[i], 1, -1, 0, height);
      vertex(x, y);
    }
    endShape();

  });
}