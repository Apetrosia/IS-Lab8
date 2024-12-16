using System;
using System.Diagnostics;
using System.IO;
using System.Threading.Tasks;

namespace NeuralNetwork1
{
    public class StudentNetwork : BaseNetwork
    {
        private int[] structure;
        private double[][] neurons;
        private double[][][] weights;
        private double[][] biases;
        private double learningRate = 0.15;
        private Random random;

        public Stopwatch stopWatch = new Stopwatch();

        public StudentNetwork(int[] structure)
        {
            this.structure = structure;
            neurons = new double[structure.Length][];
            weights = new double[structure.Length - 1][][];
            biases = new double[structure.Length - 1][];
            random = new Random();

            for (int i = 0; i < structure.Length; i++)
                neurons[i] = new double[structure[i]];

            for (int i = 0; i < structure.Length - 1; i++)
            {
                int currentLayerSize = structure[i];
                int nextLayerSize = structure[i + 1];

                weights[i] = new double[nextLayerSize][];
                biases[i] = new double[nextLayerSize];

                for (int j = 0; j < nextLayerSize; j++)
                {
                    weights[i][j] = new double[currentLayerSize];
                    for (int k = 0; k < currentLayerSize; k++)
                        weights[i][j][k] = random.NextDouble() * 2 - 1;
                    biases[i][j] = random.NextDouble() * 2 - 1;
                }
            }
        }

        public override int Train(Sample sample, double acceptableError, bool parallel)
        {
            double error;
            int iters = 0;

            do
            {
                error = TrainSample(sample.input, sample.Output);
                iters++;
            } while (error > acceptableError);

            return iters;
        }

        public override double TrainOnDataSet(SamplesSet samplesSet, int epochsCount, double acceptableError, bool parallel)
        {
            double totalError = double.PositiveInfinity;

            stopWatch.Restart();

            for (int epoch = 0; epoch < epochsCount; epoch++)
            {
                totalError = 0;

                foreach (Sample sample in samplesSet.samples)
                    totalError += TrainSample(sample.input, sample.Output);

                totalError /= samplesSet.Count;

                if (totalError <= acceptableError)
                    break;

                OnTrainProgress((double)epoch / epochsCount, totalError, stopWatch.Elapsed);
            }

            OnTrainProgress(1.0, totalError, stopWatch.Elapsed);

            stopWatch.Stop();

            return totalError;
        }

        protected override double[] Compute(double[] input)
        {
            Array.Copy(input, neurons[0], input.Length);

            for (int layer = 0; layer < weights.Length; layer++)
            {
                Parallel.For(0, neurons[layer + 1].Length, neuron =>
                {
                    double sum = biases[layer][neuron];

                    for(int prevNeuron = 0; prevNeuron < neurons[layer].Length; prevNeuron++)
                    {
                        sum += neurons[layer][prevNeuron] * weights[layer][neuron][prevNeuron];
                    }
                    neurons[layer + 1][neuron] = Sigmoid(sum);
                });
            }

            return neurons[neurons.Length - 1];
        }

        private double TrainSample(double[] inputs, double[] expectedOutputs)
        {
            var outputs = Compute(inputs);

            double totalError = 0;
            double[][] errors = new double[structure.Length][];
            for (int i = 0; i < structure.Length; i++)
                errors[i] = new double[structure[i]];

            Parallel.For(0, expectedOutputs.Length, i =>
            {
                errors[errors.Length - 1][i] = expectedOutputs[i] - outputs[i];
                totalError += Math.Pow(errors[errors.Length - 1][i], 2);
            });
            totalError /= 2;

            for (int layer = weights.Length - 1; layer >= 0; layer--)
            {
                Parallel.For(0, weights[layer].Length, neuron =>
                {
                    double delta = 0;

                    if (layer == weights.Length - 1)
                        delta = (expectedOutputs[neuron] - neurons[layer + 1][neuron]) * SigmoidDerivative(neurons[layer + 1][neuron]);
                    else
                        delta = errors[layer + 1][neuron] * SigmoidDerivative(neurons[layer + 1][neuron]);

                    for(int prevNeuron = 0; prevNeuron < neurons[layer].Length; prevNeuron++)
                    {
                        weights[layer][neuron][prevNeuron] += learningRate * delta * neurons[layer][prevNeuron];
                        errors[layer][prevNeuron] += delta * weights[layer][neuron][prevNeuron];
                    }

                    biases[layer][neuron] += learningRate * delta;
                });
            }

            return totalError;
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-x));

        private double SigmoidDerivative(double x) => x * (1.0 - x);

        public void ExportNetwork()
        {
            using (var writer = new BinaryWriter(File.Open("networks\\studNet.bin", FileMode.Create)))
            {
                // Сохраняем структуру
                writer.Write(structure.Length);
                foreach (var layerSize in structure)
                    writer.Write(layerSize);

                // Сохраняем нейроны
                writer.Write(neurons.Length);
                foreach (var layer in neurons)
                {
                    writer.Write(layer.Length);
                    foreach (var neuron in layer)
                        writer.Write(neuron);
                }

                // Сохраняем веса
                writer.Write(weights.Length);
                foreach (var layerWeights in weights)
                {
                    writer.Write(layerWeights.Length);
                    foreach (var neuronWeights in layerWeights)
                    {
                        writer.Write(neuronWeights.Length);
                        foreach (var weight in neuronWeights)
                            writer.Write(weight);
                    }
                }

                // Сохраняем сдвиги (biases)
                writer.Write(biases.Length);
                foreach (var layerBiases in biases)
                {
                    writer.Write(layerBiases.Length);
                    foreach (var bias in layerBiases)
                        writer.Write(bias);
                }
            }
        }
    }
}
