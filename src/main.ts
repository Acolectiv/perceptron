import Perceptron from "./Perceptron";

let ai = new Perceptron([5, 3, 1]);

const trainingData = [
    [2, 5, 3, 18, 2],
    [4, 6, 1, 12, 3],
    [1, 3, 5, 10, 1],
];

const trainingLabels = [0, 1, 0];

ai.train(trainingData, trainingLabels);

const input = [2, 5, 3, 18, 2];
const prediction = ai.predict(input);

console.log("Prediction:", prediction);