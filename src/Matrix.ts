class Matrix {
    data: number[][];
    rows: number;
    cols: number;

    constructor(rows: number, cols: number, data?: number[][]) {
        this.rows = rows;
        this.cols = cols;

        if (data) {
            this.data = data;
        } else {
            this.data = Array(this.rows).fill(0).map(() => Array(this.cols).fill(0));
        }
    }

    static fromArray(arr: number[]): Matrix {
        return new Matrix(arr.length, 1, arr.map(val => [val]));
    }

    fillRandom(): Matrix {
        this.data = this.data.map(row => row.map(() => Math.random()));
        return this;
    }

    fillRandomBinary(prob: number): Matrix {
        this.data = this.data.map(row => row.map(() => Math.random() < prob ? 0 : 1));
        return this;
    }

    static subtract(a: Matrix, b: Matrix): Matrix {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error("Matrices must have the same dimensions for subtraction");
        }

        let result = new Matrix(a.rows, a.cols);
        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] - b.data[i][j];
            }
        }

        return result;
    }

    static multiply(a: Matrix, b: Matrix): Matrix {
        if (a.cols !== b.rows) {
            throw new Error("Number of columns in first matrix must be equal to number of rows in second for multiplication");
        }

        let result = new Matrix(a.rows, b.cols);

        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }

    static add(a: Matrix, b: Matrix): Matrix {
        if (a.rows !== b.rows || a.cols !== b.cols) {
            throw new Error("Both matrices must have the same dimensions for addition");
        }

        let result = new Matrix(a.rows, a.cols);

        for (let i = 0; i < a.rows; i++) {
            for (let j = 0; j < a.cols; j++) {
                result.data[i][j] = a.data[i][j] + b.data[i][j];
            }
        }

        return result;
    }

    add(other: Matrix): Matrix {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new Error("Matrices must have the same dimensions for addition");
        }

        let result = new Matrix(this.rows, this.cols);
        
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j] + other.data[i][j];
            }
        }

        return result;
    }

    subtract(other: Matrix): Matrix {
        if (this.rows !== other.rows || this.cols !== other.cols) {
            throw new Error("Both matrices must have the same dimensions for subtraction");
        }

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] -= other.data[i][j];
            }
        }

        return this;
    }

    multiply(other: Matrix | number): Matrix {
        if (other instanceof Matrix) {
            if (this.rows !== other.rows || this.cols !== other.cols) {
                throw new Error("Both matrices must have the same dimensions for multiplication");
            }

            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] *= other.data[i][j];
                }
            }
        } else {
            for (let i = 0; i < this.rows; i++) {
                for (let j = 0; j < this.cols; j++) {
                    this.data[i][j] *= other;
                }
            }
        }

        return this;
    }

    map(func: (val: number, row: number, col: number) => number): Matrix {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                let val = this.data[i][j];
                this.data[i][j] = func(val, i, j);
            }
        }

        return this;
    }

    static map(matrix: Matrix, func: (val: number, row: number, col: number) => number): Matrix {
        let result = new Matrix(matrix.rows, matrix.cols);

        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                let val = matrix.data[i][j];
                result.data[i][j] = func(val, i, j);
            }
        }

        return result;
    }

    copy(): Matrix {
        let result = new Matrix(this.rows, this.cols);

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                result.data[i][j] = this.data[i][j];
            }
        }

        return result;
    }

    toArray(): number[] {
        let arr = [];

        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                arr.push(this.data[i][j]);
            }
        }

        return arr;
    }

    static transpose(matrix: Matrix): Matrix {
        let result = new Matrix(matrix.cols, matrix.rows);

        for (let i = 0; i < matrix.rows; i++) {
            for (let j = 0; j < matrix.cols; j++) {
                result.data[j][i] = matrix.data[i][j];
            }
        }

        return result;
    }

    setData(data: number[][]): Matrix {
        this.data = data;
        return this;
    }
    
    static dotProduct(a: Matrix, b: Matrix): Matrix {
        if (a.cols !== b.rows) {
            throw new Error('Columns of A must match rows of B.');
        }

        let result = new Matrix(a.rows, b.cols);
        for (let i = 0; i < result.rows; i++) {
            for (let j = 0; j < result.cols; j++) {
                let sum = 0;
                for (let k = 0; k < a.cols; k++) {
                    sum += a.data[i][k] * b.data[k][j];
                }
                result.data[i][j] = sum;
            }
        }

        return result;
    }
    
    fill(value: number): Matrix {
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                this.data[i][j] = value;
            }
        }
        return this;
    }

    clone(): Matrix {
        const clonedMatrix = new Matrix(this.rows, this.cols);
        for (let i = 0; i < this.rows; i++) {
            clonedMatrix.data[i] = [...this.data[i]];
        }

        return clonedMatrix;
    }
    
    mean(): number {
        let sum = 0;
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                sum += this.data[i][j];
            }
        }
        return sum / (this.rows * this.cols);
    }
    
    variance(): number {
        const mean = this.mean();
        let sum = 0;
        for (let i = 0; i < this.rows; i++) {
            for (let j = 0; j < this.cols; j++) {
                const diff = this.data[i][j] - mean;
                sum += diff * diff;
            }
        }
        return sum / (this.rows * this.cols);
    }

    reshape(rows: number, cols: number): Matrix {
        const data: number[][] = [];
      
        for (let i = 0; i < rows; i++) {
          const row: number[] = [];
          for (let j = 0; j < cols; j++) {
            const dataIndex = i * cols + j;
            row.push(this.data[Math.floor(dataIndex / this.cols)][dataIndex % this.cols]);
          }
          data.push(row);
        }
      
        this.rows = rows;
        this.cols = cols;
        this.data = data;
      
        return this;
      }
}

export default Matrix