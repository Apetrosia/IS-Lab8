using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.IO;
using System.Text.RegularExpressions;
using Accord.Neuro;
using Accord.Statistics.Kernels;

namespace AForge.WindowsForms
{
    internal class Settings
    {
        private int _border = 20;
        public int border
        {
            get
            {
                return _border;
            }
            set
            {
                if ((value > 0) && (value < height / 3))
                {
                    _border = value;
                    if (top > 2 * _border) top = 2 * _border;
                    if (left > 2 * _border) left = 2 * _border;
                }
            }
        }

        public int width = 640;
        public int height = 640;
        
        /// <summary>
        /// Размер сетки для сенсоров по горизонтали
        /// </summary>
        public int blocksCount = 10;

        /// <summary>
        /// Желаемый размер изображения до обработки
        /// </summary>
        public Size orignalDesiredSize = new Size(500, 500);
        /// <summary>
        /// Желаемый размер изображения после обработки
        /// </summary>
        public Size processedDesiredSize = new Size(500, 500);

        public int margin = 10;
        public int top = 40;
        public int left = 40;

        /// <summary>
        /// Второй этап обработки
        /// </summary>
        public bool processImg = true;

        /// <summary>
        /// Порог при отсечении по цвету 
        /// </summary>
        public byte threshold = 120;
        public float differenceLim = 0.15f;

        public void incTop() { if (top < 2 * _border) ++top; }
        public void decTop() { if (top > 0) --top; }
        public void incLeft() { if (left < 2 * _border) ++left; }
        public void decLeft() { if (left > 0) --left; }
    }

    internal class MagicEye
    {
        /// <summary>
        /// Обработанное изображение
        /// </summary>
        public Bitmap processed;
        /// <summary>
        /// Оригинальное изображение после обработки
        /// </summary>
        public Bitmap original;

        /// <summary>
        /// Класс настроек
        /// </summary>
        public Settings settings = new Settings();

        public ActivationNetwork libNet;

        // studNet
        private int[] structure;
        private double[][] neurons;
        private double[][][] weights;
        private double[][] biases;

        public bool student = false;

        private int time;

        private string lastAnswer = "Обработка";

        public MagicEye()
        {
            ImportNetwork();
            time = DateTime.Now.Second;
        }

        public void ImportNetwork()
        {
            using (FileStream fs = new FileStream("networks\\libNet.bin", FileMode.Open))
            {
                var formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                libNet = (ActivationNetwork)formatter.Deserialize(fs);
            }

            using (var reader = new BinaryReader(File.Open("networks\\studNet.bin", FileMode.Open)))
            {
                // Загружаем структуру
                int structureLength = reader.ReadInt32();
                structure = new int[structureLength];
                for (int i = 0; i < structureLength; i++)
                    structure[i] = reader.ReadInt32();

                // Загружаем нейроны
                int neuronsLength = reader.ReadInt32();
                neurons = new double[neuronsLength][];
                for (int i = 0; i < neuronsLength; i++)
                {
                    int layerSize = reader.ReadInt32();
                    neurons[i] = new double[layerSize];
                    for (int j = 0; j < layerSize; j++)
                        neurons[i][j] = reader.ReadDouble();
                }

                // Загружаем веса
                int weightsLength = reader.ReadInt32();
                weights = new double[weightsLength][][];
                for (int i = 0; i < weightsLength; i++)
                {
                    int layerLength = reader.ReadInt32();
                    weights[i] = new double[layerLength][];
                    for (int j = 0; j < layerLength; j++)
                    {
                        int neuronWeightsLength = reader.ReadInt32();
                        weights[i][j] = new double[neuronWeightsLength];
                        for (int k = 0; k < neuronWeightsLength; k++)
                            weights[i][j][k] = reader.ReadDouble();
                    }
                }

                // Загружаем сдвиги (biases)
                int biasesLength = reader.ReadInt32();
                biases = new double[biasesLength][];
                for (int i = 0; i < biasesLength; i++)
                {
                    int layerBiasesLength = reader.ReadInt32();
                    biases[i] = new double[layerBiasesLength];
                    for (int j = 0; j < layerBiasesLength; j++)
                        biases[i][j] = reader.ReadDouble();
                }
            }
        }

        public bool ProcessImage(Bitmap bitmap, bool stud)
        {
            student = stud;

            //  Минимальная сторона изображения (обычно это высота)
            if (bitmap.Height > bitmap.Width)
                throw new Exception("К такой забавной камере меня жизнь не готовила!");
            //  Можно было, конечено, и не кидаться эксепшенами в истерике, но идите и купите себе нормальную камеру!
            int side = bitmap.Height;

            //  Отпиливаем границы, но не более половины изображения
            if (side < 4 * settings.border) settings.border = side / 4;
            side -= 2 * settings.border;
            
            //  Мы сейчас занимаемся тем, что красиво оформляем входной кадр, чтобы вывести его на форму
            Rectangle cropRect = new Rectangle((bitmap.Width - bitmap.Height) / 2 + settings.left + settings.border, settings.top + settings.border, side, side);
            
            //  Тут создаём новый битмапчик, который будет исходным изображением
            original = new Bitmap(cropRect.Width, cropRect.Height);

            //  Объект для рисования создаём
            Graphics g = Graphics.FromImage(original);
            
            g.DrawImage(bitmap, new Rectangle(0, 0, original.Width, original.Height), cropRect, GraphicsUnit.Pixel);
            Pen p = new Pen(Color.Red);
            p.Width = 1;

            //  Теперь всю эту муть пилим в обработанное изображение
            AForge.Imaging.Filters.Grayscale grayFilter = new AForge.Imaging.Filters.Grayscale(0.2125, 0.7154, 0.0721);
            var uProcessed = grayFilter.Apply(AForge.Imaging.UnmanagedImage.FromManagedImage(original));

            
            int blockWidth = original.Width / settings.blocksCount;
            int blockHeight = original.Height / settings.blocksCount;
            for (int r = 0; r < settings.blocksCount; ++r)
                for (int c = 0; c < settings.blocksCount; ++c)
                {
                    //  Тут ещё обработку сделать
                    g.DrawRectangle(p, new Rectangle(c * blockWidth, r * blockHeight, blockWidth, blockHeight));
                }


            //  Масштабируем изображение до 500x500 - этого достаточно
            AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(settings.orignalDesiredSize.Width, settings.orignalDesiredSize.Height);
            uProcessed = scaleFilter.Apply(uProcessed);
            original = scaleFilter.Apply(original);
            g = Graphics.FromImage(original);
            //  Пороговый фильтр применяем. Величина порога берётся из настроек, и меняется на форме
            AForge.Imaging.Filters.BradleyLocalThresholding threshldFilter = new AForge.Imaging.Filters.BradleyLocalThresholding();
            threshldFilter.PixelBrightnessDifferenceLimit = settings.differenceLim;
            threshldFilter.ApplyInPlace(uProcessed);


            if (settings.processImg)
            {
                string info = processSample(ref uProcessed);
                Font f = new Font(FontFamily.GenericSansSerif, 20);
                g.DrawString(info, f, Brushes.Green, 30, 30);
            }

            //  Получить значения сенсоров из обработанного изображения размера 100x100

            //  Можно вывести информацию на изображение!
            //Font f = new Font(FontFamily.GenericSansSerif, 10);
            //for (int r = 0; r < 4; ++r)
            //    for (int c = 0; c < 4; ++c)
            //        if (currentDeskState[r * 4 + c] >= 1 && currentDeskState[r * 4 + c] <= 16)
            //        {
            //            int num = 1 << currentDeskState[r * 4 + c];
            //            
            //        }


            processed = uProcessed.ToManagedImage();

            return true;
        }

        /// <summary>
        /// Обработка одного сэмпла
        /// </summary>
        /// <param name="index"></param>
        private string processSample(ref Imaging.UnmanagedImage unmanaged)
        {
            int centerX = unmanaged.Width / 2;
            int centerY = unmanaged.Height / 2;
            int lx, rx, ly, ry;

            int a = 100;

            lx = centerX - a;
            rx = centerX + a;
            ly = centerY - a;
            ry = centerY + a;

            // Обрезаем края, оставляя только центральные блобчики
            AForge.Imaging.Filters.Crop cropFilter = new AForge.Imaging.Filters.Crop(new Rectangle(lx, ly, rx - lx, ry - ly));
            unmanaged = cropFilter.Apply(unmanaged);

            //  Масштабируем до 200x200
            AForge.Imaging.Filters.ResizeBilinear scaleFilter = new AForge.Imaging.Filters.ResizeBilinear(200, 200);
            unmanaged = scaleFilter.Apply(unmanaged);

            if (time != DateTime.Now.Second)
            {
                time = DateTime.Now.Second;
                lastAnswer = GetAuto(unmanaged.ToManagedImage());
            }

            return lastAnswer;
        }

        public double[] GetNetworkInput(Bitmap bmp)
        {
            double[] inputs = new double[400];

            for (int i = 0; i < bmp.Height; i++) // по строкам
            {
                int prev = int.MinValue;
                Color c = Color.White;
                for (int j = 0; j < bmp.Width; j++)
                {
                    if (j == 0)
                    {
                        inputs[i] = 0;
                        c = bmp.GetPixel(i, j);
                    }
                    else
                    {
                        if (c != bmp.GetPixel(i, j) && j - prev < 5)
                        {
                            c = bmp.GetPixel(i, j);
                            inputs[i]++;
                            prev = j;
                        }
                    }
                }
            }
            for (int j = 0; j < bmp.Width; j++) // по столбцам
            {
                Color c = Color.White;
                int prev = int.MinValue;
                for (int i = 0; i < bmp.Height; i++)
                {
                    if (i == 0)
                    {
                        inputs[j + 200] = 0;
                        c = bmp.GetPixel(i, j);
                    }
                    else
                    {
                        if (c != bmp.GetPixel(i, j) && j - prev < 5)
                        {
                            c = bmp.GetPixel(i, j);
                            inputs[j + 200]++;
                            prev = j;
                        }
                    }
                }
            }

            return inputs;
        }

        public string GetAuto(Bitmap bmp)
        {
            double[] outputs;

            if (student)
                outputs = StudentCompute(GetNetworkInput(bmp));
            else
                outputs = libNet.Compute(GetNetworkInput(bmp));

            int ind = 0;
            for (int i = 1; i < 10; i++)
                if (outputs[i] > outputs[ind])
                    ind = i;

            switch (ind)
            { // Citroen = 0, Ford, Hyundai, Infiniti, Mercedes, MM, Opel, Renault, Toyota, VW
                case 0:
                    return "citroen";
                case 1:
                    return "ford";
                case 2:
                    return "hyundai";
                case 3:
                    return "infiniti";
                case 4:
                    return "mercedes";
                case 5:
                    return "MM";
                case 6:
                    return "opel";
                case 7:
                    return "renault";
                case 8:
                    return "toyota";
                case 9:
                    return "VW";
                default:
                    return "unknown";
            }
        }

        public double[] StudentCompute(double[] input)
        {
            Array.Copy(input, neurons[0], input.Length);

            for (int layer = 0; layer < weights.Length; layer++)
            {
                Parallel.For(0, neurons[layer + 1].Length, neuron =>
                {
                    double sum = biases[layer][neuron];

                    for (int prevNeuron = 0; prevNeuron < neurons[layer].Length; prevNeuron++)
                    {
                        sum += neurons[layer][prevNeuron] * weights[layer][neuron][prevNeuron];
                    }
                    neurons[layer + 1][neuron] = Sigmoid(sum);
                });
            }

            return neurons[neurons.Length - 1];
        }

        private double Sigmoid(double x) => 1.0 / (1.0 + System.Math.Exp(-x));
    }
}

