using System;
using System.Collections.Generic;
using System.Text;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MultiLayerPerceptron
{
	public class NeuralNetwork
	{
		public int Num_Inputs { get; private set; }
		public int Num_Hidden { get; private set; }
		public int Num_Outputs { get; private set; }
		public Matrix<double> Weights_Input_To_Hidden { get; private set; }
		public Matrix<double> Weights_Hidden_To_Output { get; private set; }
		public Matrix<double> Biases_Input_To_Hidden { get; private set; }
		public Matrix<double> Biases_Hidden_To_Output { get; private set; }
		public double LearningRate { get; private set; }
		public NeuralNetwork(int num_inputs, int num_hidden, int num_outputs)
		{
			Random rand = new Random();
			Num_Inputs = num_inputs;
			Num_Hidden = num_hidden;
			Num_Outputs = num_outputs;
			Weights_Input_To_Hidden = Matrix<double>.Build.Dense(num_hidden, num_inputs,(i,j) => rand.NextDouble()*2 - 1);
			Weights_Hidden_To_Output = Matrix<double>.Build.Dense(num_outputs, num_hidden, (i, j) => rand.NextDouble() * 2 - 1);
			Biases_Input_To_Hidden = Matrix<double>.Build.Dense(num_hidden, 1, (i, j) => rand.NextDouble() * 2 - 1);
			Biases_Hidden_To_Output = Matrix<double>.Build.Dense(num_outputs, 1, (i, j) => rand.NextDouble() * 2 - 1);
			LearningRate = 0.1;
		}

		public static Matrix<double> Sigmoid(Matrix<double> x)
		{
			return Matrix<double>.Build.Dense(x.RowCount, x.ColumnCount, (i,j) => 1 / (1 + Math.Exp(-x[i,j])));
		}
		public Matrix<double> FeedForward(Matrix<double> inputs)
		{
			Matrix<double> hidden = Weights_Input_To_Hidden * inputs + Biases_Input_To_Hidden;
			hidden = Sigmoid(hidden);
			Matrix<double> calculatedOutputs = Weights_Hidden_To_Output * hidden + Biases_Hidden_To_Output;
			calculatedOutputs = Sigmoid(calculatedOutputs);
			return calculatedOutputs;
		}

		public void Train(Matrix<double> inputs, Matrix<double> expectedOutputs)
		{
			Matrix<double> hidden = Weights_Input_To_Hidden * inputs + Biases_Input_To_Hidden;
			hidden = Sigmoid(hidden);
			Matrix<double> calculatedOutputs = Weights_Hidden_To_Output * hidden + Biases_Hidden_To_Output;
			calculatedOutputs = Sigmoid(calculatedOutputs);

			Matrix<double> output_errors = expectedOutputs - calculatedOutputs;
			Matrix<double> gradient = Matrix<double>.Build.Dense(calculatedOutputs.RowCount, calculatedOutputs.ColumnCount, (i, j) => output_errors[i,j] * calculatedOutputs[i, j] * (1 - calculatedOutputs[i, j]));

			gradient = gradient * LearningRate;
			Matrix<double> hidden_T = hidden.Transpose();
			Matrix<double> Weights_Hidden_To_Output_Deltas = gradient * hidden_T;

			Weights_Hidden_To_Output = Weights_Hidden_To_Output + Weights_Hidden_To_Output_Deltas;
			Biases_Hidden_To_Output = Biases_Hidden_To_Output + gradient;

			Matrix<double> who_t = Weights_Hidden_To_Output.Transpose();
			Matrix<double> hidden_errors = who_t * output_errors;
			Matrix<double> hidden_gradient = Matrix<double>.Build.Dense(hidden.RowCount, hidden.ColumnCount, (i, j) => hidden_errors[i,j] * hidden[i, j] * (1 - hidden[i, j]));

			hidden_gradient = hidden_gradient * LearningRate;

			Matrix<double> input_T = inputs.Transpose();
			Matrix<double> Weights_Input_To_Hidden_Deltas = hidden_gradient * input_T;

			Weights_Input_To_Hidden = Weights_Input_To_Hidden + Weights_Input_To_Hidden_Deltas;
			Biases_Input_To_Hidden = Biases_Input_To_Hidden + hidden_gradient;
		}
	}
}
