using System;
using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

namespace MultiLayerPerceptron
{
	class Program
	{
		Random rand = new Random();
		static void Main(string[] args)
		{
			NeuralNetwork nn = new NeuralNetwork(2, 4, 2);
			MatrixBuilder<double> M = Matrix<double>.Build;
			for (int i = 0; i < 10000; i++)
			{
				double[][,] data = GetRandomData();
				Matrix<double> inputs = M.DenseOfArray(data[0]);
				Matrix<double> expectedOutputs = M.DenseOfArray(data[1]);
				nn.Train(inputs, expectedOutputs);
				
				
			}


			for (int i = 0; i < 10; i++)
			{
				double[][,] dataToPredict = GetRandomData();
				Matrix<double> inputsToPredict = M.DenseOfArray(dataToPredict[0]);
				Matrix<double> predictionOutput = nn.FeedForward(inputsToPredict);
				if (predictionOutput[0, 0] > predictionOutput[1, 0])
				{
					Console.WriteLine($"{inputsToPredict[0, 0]} XOR {inputsToPredict[1, 0]} = 0");
				}
				else
				{
					Console.WriteLine($"{inputsToPredict[0, 0]} XOR {inputsToPredict[1, 0]} = 1");
				}

			}






		}

		static double[][,] GetRandomData()
		{
			double[][,] Data;
			Random rand = new Random();
			switch (rand.Next(0, 4))
			{
				case 0:
					Data = new double[2][,]
					{
					new double[,] {{ 0 }, { 0 } },
					new double[,] {{ 1 }, { 0 } }
					};
					break;
				case 1:
					Data = new double[2][,]
					{
					new double[,] {{ 0 }, { 1 } },
					new double[,] {{ 0 }, { 1 } }
					};
					break;
				case 2:
					Data = new double[2][,]
					{
					new double[,] {{ 1 }, { 0 } },
					new double[,] {{ 0 }, { 1 } }
					};
					break;
				default:
					Data = new double[2][,]
					{
					new double[,] {{ 1 }, { 1 } },
					new double[,] {{ 1 }, { 0 } }
					};
					break;
			}
			return Data;
		}
	}
}
