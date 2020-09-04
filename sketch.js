//todo: create vars
//* VARS
let x_vals = []; // array of values - not a tensor
let y_vals = []; // array of values - not a tensor
let m, b; // tf trainable variables



function setup() {

	createCanvas(400, 400);
	background(0);

	// tf.variable because they change over time
	// m & b are trainable!!!
	m = tf.variable(tf.scalar(random(1)));
	b = tf.variable(tf.scalar(random(1)));
}
//todo: create a datase with mouse
//todo: loss function MSE
//todo: optimizer
//todo: predict function

//* CREATE DATASET
function mouseClicked() {

	let x = map(mouseX, 0, width, 0, 1);
	let y = map(mouseY, 0, height, 1, 0);
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
	const ys = xs.mul(m).add(b); // tensor ys

	return ys;
}


//* DRAW LOOP
function draw() {
	background(0);

	stroke(255);
	strokeWeight(8);
	for (let i = 0; i < x_vals.length; i++) {

		let px = map(x_vals[i], 0, 1, 0, width);
		let py = map(y_vals[i], 1, 0, 0, height);
		point(px, py);
	}

	//* TRAIN
	tf.tidy(() => { //* tidy to clear the memory

		if (x_vals.length > 0) {
			const ys = tf.tensor1d(y_vals); // tensor to cleanup
			// cleanup also this tensor below
			optimizer.minimize(() => loss(predict(x_vals), ys));
			//* this part above is where the learning happens in tfjs
		}

		// draw the line on screen

		// I have the xs and just need to predict the ys
		const xs = [0, 1];
		const ys = predict(xs); // predict creates tensors

		// remap from predictions to screen coord
		let x1 = map(xs[0], 0, 1, 0, width);
		let x2 = map(xs[1], 0, 1, 0, width);

		// take the values from the tensor
		let lineY = ys.dataSync();

		// remap the ys obtained from the tensor
		let y1 = map(lineY[0], 1, 0, 0, height);
		let y2 = map(lineY[1], 1, 0, 0, height);

		// draw the line
		strokeWeight(2);
		line(x1, y1, x2, y2);
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