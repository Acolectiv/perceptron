// @ts-ignore
import { writeFileSync, readFileSync } from "fs";

enum ActivationFunction {
    Sigmoid,
    Tanh,
    Relu
}

enum Initialization {
    Random,
    Xavier,
    He
}

enum Regularization {
    None,
    L1,
    L2
}

import Matrix from "./Matrix";

class Perceptron {
    layers: Matrix[];
    weights: Matrix[];
    biases: Matrix[];
    activationFunction: ActivationFunction;
    learningRate: number;
    momentum: number;
    batchSize: number;
    weightVelocity: Matrix[];
    biasVelocity: Matrix[];

    regularization: Regularization;
    regularizationRate: number;
    dropoutRate: number;
    isDropout: boolean;
    dropoutMask: Matrix[];

    validationData: number[][];
    validationLabels: number[];
    epochs: number;
    earlyStopPatience: number;
    weightLearningRate: number;
    biasLearningRate: number;
    minLoss: number;
    epochsWithoutImprovement: number;

    bnGammas: Matrix[];
    bnBetas: Matrix[];
    bnGammaVelocity: Matrix[];
    bnBetaVelocity: Matrix[];

    constructor(
        layerSizes: number[],
        activationFunction: ActivationFunction = ActivationFunction.Sigmoid,
        learningRate: number = 0.1,
        momentum: number = 0.9,
        batchSize: number = 1,
        initialization: Initialization = Initialization.Random,
        regularization: Regularization = Regularization.None,
        regularizationRate: number = 0.01,
        dropoutRate: number = 0.5,
        epochs: number = 100, 
        earlyStopPatience: number = 10,
        weightLearningRate: number = 0.5, 
        biasLearningRate: number = 0.5
    ) {
        this.layers = layerSizes.map(size => new Matrix(size, 1));

        this.weights = layerSizes.slice(1).map((size, i) => {
            let weightMatrix = new Matrix(size, layerSizes[i]);
            switch (initialization) {
                case Initialization.Xavier:
                    return weightMatrix.fillRandom().multiply(Math.sqrt(1 / layerSizes[i]));
                case Initialization.He:
                    return weightMatrix.fillRandom().multiply(Math.sqrt(2 / layerSizes[i]));
                default:
                    return weightMatrix.fillRandom();
            }
        });

        this.biases = layerSizes.slice(1).map(size => new Matrix(size, 1).fillRandom());
        this.activationFunction = activationFunction;
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.batchSize = batchSize;
        this.weightVelocity = this.weights.map(weight => new Matrix(weight.rows, weight.cols));
        this.biasVelocity = this.biases.map(bias => new Matrix(bias.rows, bias.cols));

        this.regularization = regularization;
        this.regularizationRate = regularizationRate;
        this.dropoutRate = dropoutRate;
        this.isDropout = false;
        this.dropoutMask = [];

        this.epochs = epochs;
        this.earlyStopPatience = earlyStopPatience;
        this.weightLearningRate = weightLearningRate;
        this.biasLearningRate = biasLearningRate;
        this.minLoss = Infinity;
        this.epochsWithoutImprovement = 0;

        this.bnGammas = layerSizes.slice(1).map(size => new Matrix(size, 1).fill(1)); // Initialize gamma to 1
        this.bnBetas = layerSizes.slice(1).map(size => new Matrix(size, 1).fill(0)); // Initialize beta to 0
        this.bnGammaVelocity = this.bnGammas.map(gamma => new Matrix(gamma.rows, gamma.cols));
        this.bnBetaVelocity = this.bnBetas.map(beta => new Matrix(beta.rows, beta.cols));
    }

    setValidationData(data: number[][], labels: number[]): void {
        this.validationData = data;
        this.validationLabels = labels;
    }

    calculateLoss(data: number[][], labels: number[]): number {
        let totalLoss = 0;
        for (let i = 0; i < data.length; i++) {
            let prediction = this.predict(data[i]);
            totalLoss += this.lossFunc(prediction, labels[i]);
        }
        return totalLoss / data.length;
    }

    // predict(inputArray: number[]): number {
    //     this.layers[0] = Matrix.fromArray(inputArray);
    //     for (let i = 1; i < this.layers.length; i++) {
    //         this.layers[i] = Matrix.add(Matrix.dotProduct(this.weights[i-1], this.layers[i-1]), this.biases[i-1]).map(this.activationFunc);
    //     }
    //     return this.layers[this.layers.length-1].toArray()[0];
    // }

    predict(inputArray: number[]): number {
        this.layers[0] = Matrix.fromArray(inputArray);
        for (let i = 1; i < this.layers.length; i++) {
            this.layers[i] = Matrix.add(
                Matrix.dotProduct(this.weights[i - 1], this.layers[i - 1]),
                this.biases[i - 1]
            );

            this.layers[i] = this.normalizeBatch(
                this.layers[i],
                this.bnGammas[i - 1],
                this.bnBetas[i - 1]
            );

            this.layers[i] = this.layers[i].map(this.activationFunc);
        }
        return this.layers[this.layers.length - 1].toArray()[0];
    }

    train(trainingData: number[][], trainingLabels: number[], batchSize: number = 1): void {
        for (let epoch = 0; epoch < this.epochs; epoch++) {
            for (let i = 0; i < trainingData.length; i += batchSize) {
                let batchData = trainingData.slice(i, i + batchSize);
                let batchLabels = trainingLabels.slice(i, i + batchSize);
                for (let j = 0; j < batchData.length; j++) {
                    let labelArray = [batchLabels[j] === 0 ? 1 : 0, batchLabels[j]];
                    this.backpropagate(batchData[j], labelArray);
                }
                this.updateWeightsAndBiases();
            }
            if (this.validationData && this.validationLabels) {
                let validationLoss = this.calculateLoss(this.validationData, this.validationLabels);
                if (validationLoss < this.minLoss) {
                    this.minLoss = validationLoss;
                    this.epochsWithoutImprovement = 0;
                } else {
                    this.epochsWithoutImprovement++;
                    if (this.epochsWithoutImprovement > this.earlyStopPatience) {
                        //console.log(`Early stopping at epoch ${epoch}`);
                        break;
                    }
                }
            }
        }
    }
    

    // backpropagate(inputArray: number[], targetArray: number[]): void {
    //     this.predict(inputArray);
    
    //     for (let i = this.layers.length-2; i >= 0; i--) {
    //         const targetMatrix = Matrix.fromArray(targetArray);
    //         const outputLayer = this.layers[this.layers.length - 1];
            
    //         console.log("Target Matrix dimensions:", targetMatrix.rows, targetMatrix.cols);
    //         console.log("Output Layer dimensions:", outputLayer.rows, outputLayer.cols);

    //         const reshapedTargetMatrix = targetMatrix.reshape(outputLayer.rows, outputLayer.cols);
    //         let outputErrors = Matrix.subtract(reshapedTargetMatrix, outputLayer);
    
    //         let gradients = Matrix.map(this.layers[i+1], this.activationFuncDerivative);
    //         gradients.multiply(outputErrors);
    //         gradients.multiply(this.learningRate);
    
    //         let weightDeltas = Matrix.multiply(gradients, Matrix.transpose(this.layers[i]));
    
    //         this.weights[i].add(weightDeltas);
    //         this.biases[i].add(gradients);
    
    //         outputErrors = Matrix.multiply(Matrix.transpose(this.weights[i]), outputErrors);
    //     }
    // }

    backpropagate(inputArray: number[], targetArray: number[]): void {
        this.predict(inputArray);
      
        let target = Matrix.fromArray(targetArray); // Convert targetArray to matrix
        const output = this.layers[this.layers.length - 1];
      
        target = target.reshape(output.rows, output.cols);

        const outputErrors = Matrix.subtract(target, output);
      
        for (let i = this.layers.length - 2; i >= 0; i--) {
          const gradients = Matrix.map(this.layers[i + 1], this.activationFuncDerivative);
          gradients.multiply(outputErrors);
          gradients.multiply(this.learningRate);
      
          const weightDeltas = Matrix.multiply(gradients, Matrix.transpose(this.layers[i]));
      
          this.weights[i].add(weightDeltas);
          this.biases[i].add(gradients);
      
          const transposedWeights = Matrix.transpose(this.weights[i]);
          outputErrors.setData(Matrix.multiply(transposedWeights, outputErrors).data); // Update the outputErrors with new values
        }
      }

    updateWeightsAndBiases(): void {
        for (let i = 0; i < this.weights.length; i++) {
            if (this.regularization == Regularization.L1) {
                let sign = Matrix.map(this.weights[i], x => x > 0 ? 1 : -1);
                this.weights[i] = this.weights[i].subtract(sign.multiply(this.weightLearningRate * this.regularizationRate));
            } else if (this.regularization == Regularization.L2) {
                this.weights[i] = this.weights[i].multiply(1 - this.weightLearningRate * this.regularizationRate);
            }
            this.weights[i] = this.weights[i].add(this.weightVelocity[i]);
            this.biases[i] = this.biases[i].add(this.biasVelocity[i].multiply(this.biasLearningRate));
        }
        this.weightVelocity = this.weightVelocity.map(weight => new Matrix(weight.rows, weight.cols));
        this.biasVelocity = this.biasVelocity.map(bias => new Matrix(bias.rows, bias.cols));
    }

    mse(targets: number[], inputs: number[][]): number {
        let sum = 0;
        for (let i = 0; i < targets.length; i++) {
            let prediction: number = this.predict(inputs[i]);
            let target: number = targets[i];
            sum += Math.pow(prediction - target, 2);
        }
        return sum / targets.length;
    }
    
    

    activationFunc = (x: number): number => {
        switch (this.activationFunction) {
            case ActivationFunction.Sigmoid:
                return 1 / (1 + Math.exp(-x));
            case ActivationFunction.Tanh:
                return Math.tanh(x);
            case ActivationFunction.Relu:
                return Math.max(0, x);
        }
    }

    activationFuncDerivative = (x: number): number => {
        switch (this.activationFunction) {
            case ActivationFunction.Sigmoid:
                return x * (1 - x);
            case ActivationFunction.Tanh:
                return 1 - x * x;
            case ActivationFunction.Relu:
                return x > 0 ? 1 : 0;
        }
    }

    enableDropout(): void {
        this.isDropout = true;
    }

    disableDropout(): void {
        this.isDropout = false;
    }

    saveModel(filePath: string): void {
        let modelData = {
            weights: this.weights.map(weight => weight.data),
            biases: this.biases.map(bias => bias.data),
            layers: this.layers.map(layer => layer.rows)
        };
        let modelJson = JSON.stringify(modelData);
        writeFileSync(filePath, modelJson, 'utf8');
    }

    loadModel(filePath: string): void {
        let modelJson = readFileSync(filePath, 'utf8');
        let modelData = JSON.parse(modelJson);

        this.weights = modelData.weights.map((weightData: number[][], i: number) => new Matrix(weightData.length, modelData.layers[i]).setData(weightData));
        this.biases = modelData.biases.map((biasData: number[][], i: number) => new Matrix(biasData.length, 1).setData(biasData));
    }

    lossFunc(prediction: number, label: number): number {
        let error = prediction - label;
        return error * error;
    }

    normalizeBatch(layer: Matrix, gamma: Matrix, beta: Matrix): Matrix {
        const epsilon = 1e-10;
        const mean = layer.mean();
        const variance = layer.variance();

        const normalizedLayer = layer.clone();
        for (let i = 0; i < normalizedLayer.rows; i++) {
            for (let j = 0; j < normalizedLayer.cols; j++) {
                normalizedLayer.data[i][j] = (normalizedLayer.data[i][j] - mean) / Math.sqrt(variance + epsilon);
                normalizedLayer.data[i][j] = gamma.data[i][j] * normalizedLayer.data[i][j] + beta.data[i][j];
            }
        }

        return normalizedLayer;
    }
}

export default Perceptron;