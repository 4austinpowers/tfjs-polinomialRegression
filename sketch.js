//todo: create vars
//* VARS
let x_vals = []; // array of values - not a tensor
let y_vals = []; // array of values - not a tensor
let a, b, c; // tf trainable variables



function setup() {

	createCanvas(400, 400);
	background(0);

	// tf.variable because they change over time
	// a b c are trainable!!!
	a = tf.variable(tf.scalar(random(1)));
	b = tf.variable(tf.scalar(random(1)));
	c = tf.variable(tf.scalar(random(1)));

}
// todo: - 1 - create a datase with mouse
// todo: - 2 - loss function MSE
// todo: - 3 - optimizer
// todo: - 4 - predict function

//* CREATE DATASET
function mouseClicked() {

	let x = map(mouseX, 0, width, -1, 1);
	let y = map(mouseY, 0, height, 1, -1);
	x_vals.push(x);
	y_vals.push(y);
};

//* LOSS
function loss(predictions, labels) {
	// mean squared error
	// (predictions - labels)^2
	return predictions.sub(labels).square().mean();
}

//* OPTIMIZER
const learningRate = 0.03;
const optimizer = tf.train.sgd(learningRate);

//* PREDICT
function predict(x) {

	const xs = tf.tensor1d(x); // tensor xs
	//     y = a*x^2 + b*x + c -- parabolic function
	const ys = xs.square().mul(a).add(xs.mul(b)).add(c);
	return ys;
}


//* DRAW LOOP
function draw() {
	background(0);

	// draw the point from the dataset
	stroke(255);
	strokeWeight(8);
	for (let i = 0; i < x_vals.length; i++) {

		let px = map(x_vals[i], -1, 1, 0, width);
		let py = map(y_vals[i], 1, -1, 0, height);
		point(px, py);
	}

	//* TRAIN
	tf.tidy(() => { //* tidy to clear the memory

		if (x_vals.length > 0) { // don't train if there are no values

			const ys = tf.tensor1d(y_vals); // tensor to cleanup

			// cleanup also this tensor below
			optimizer.minimize(() => loss(predict(x_vals), ys));
			//* this part above is where the learning happens in tfjs
		}

		// draw the line on screen

		// I have the xs and just need to predict the ys
		const curveX = [];

		for (let x = -1; x = 1; x += 0.05) {
			curveX.push(x);
		}
		const ys = predict(curveX); // predict creates tensors

		// take the values from the tensor
		let curveY = ys.dataSync();

		beginShape();
		noFill();
		stroke(255);
		strokeWeight(2);
		for (let x = 0; x = curveX.length; x++) {
			let x = map(curveX, -1, 1, 0, width);
			let y = map(curveY, -1, 1, 0, height);
			vertex(x, y);
		}
		endShape();

	});
}

// 	// const values = [];
// 	// for (let i = 0; i < 15; i++) {
// 	// 	values[i] = random(0, 255);
// 	// }
// 	// const shape = [5, 3];


// 	// const a = tf.tensor(values, shape, 'int32');
// 	// const b = tf.tensor(values, shape, 'int32');
// 	// const bb = b.transpose();
// 	// const c = a.matMul(bb);

// 	// c.print();
// 	//const data = tf.tensor([0, 0, 127, 255, 1, 0, 200, 200], [2, 2, 2]);
// 	// tens.print();
// 	// tens.array(3).then(array => console.log(array));

// 	// const vtens = tf.variable(tens);
// 	// console.log(vtens);
// 	// tens.data().then(function (stuff) {
// 	// 	console.log(stuff);
// 	// })
// 	//console.log(data);
// }

// function draw() {
// 	// frameRate(1);
// 	const values = [];
// 	for (let i = 0; i < 150000; i++) {
// 		values[i] = random(0, 255);
// 	}
// 	const shape = [500, 300];


// 	const a = tf.tensor(values, shape, 'int32');
// 	const b = tf.tensor(values, shape, 'int32');
// 	const b_t = b.transpose();
// 	const c = a.matMul(b_t);

// 	c.print();

// 	//* use of TF DISPOSE to clear memory
// 	a.dispose();
// 	b.dispose();
// 	c.dispose();
// 	b_t.dispose();

// 	console.log(tf.memory().numTensors);

// 	//* USE OF TF TIDY to clear memory

// 	tf.tidy(function () {
// 		const a = tf.tensor(values, shape, 'int32');
// 		const b = tf.tensor(values, shape, 'int32');
// 		const b_t = b.transpose();
// 		const c = a.matMul(b_t);

// 		c.print();

// 		a.dispose();
// 		b.dispose();
// 		c.dispose();
// 		b_t.dispose();

// 		console.log(tf.memory().numTensors);
// 	})

// }