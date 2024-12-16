using System;
using System.Collections.Generic;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.Text.RegularExpressions;
using MathNet.Numerics;
using Accord.Neuro;

namespace NeuralNetwork1
{
    public partial class NeuralNetworksStand : Form
    {
        /// <summary>
        /// Генератор изображений (образов)
        /// </summary>
        GenerateImage generator = new GenerateImage();

        //  Создаём новую обучающую выборку
        SamplesSet samples = new SamplesSet();

        public AutoType GetClassNum(string fileName)
        {
            switch (Regex.Split(fileName, @"[\d\\]+")[1])
            { // Citroen = 0, Ford, Hyundai, Infiniti, Mercedes, MM, Opel, Renault, Toyota, VW
                case "citroen":
                    return (AutoType)0;
                case "ford":
                    return (AutoType)1;
                case "hyundai":
                    return (AutoType)2;
                case "infiniti":
                    return (AutoType)3;
                case "mercedes":
                    return (AutoType)4;
                case "MM":
                    return (AutoType)5;
                case "opel":
                    return (AutoType)6;
                case "renault":
                    return (AutoType)7;
                case "toyota":
                    return (AutoType)8;
                case "VW":
                    return (AutoType)9;
                default:
                    return (AutoType)10;
            }
        }

        /// <summary>
        /// Текущая выбранная через селектор нейросеть
        /// </summary>
        public BaseNetwork Net
        {
            get
            {
                var selectedItem = (string) netTypeBox.SelectedItem;
                if (!networksCache.ContainsKey(selectedItem))
                    networksCache.Add(selectedItem, CreateNetwork(selectedItem));

                return networksCache[selectedItem];
            }
        }

        private readonly Dictionary<string, Func<int[], BaseNetwork>> networksFabric;
        private Dictionary<string, BaseNetwork> networksCache = new Dictionary<string, BaseNetwork>();

        /// <summary>
        /// Конструктор формы стенда для работы с сетями
        /// </summary>
        /// <param name="networksFabric">Словарь функций, создающих сети с заданной структурой</param>
        public NeuralNetworksStand(Dictionary<string, Func<int[], BaseNetwork>> networksFabric)
        {
            InitializeComponent();

            string[] imageFiles = Directory.GetFiles("imgs", "*.*", SearchOption.TopDirectoryOnly).ToArray();

            foreach (string file in imageFiles)
            {
                double[] inputs = new double[400];
                Bitmap bmp = new Bitmap(file);
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
                samples.AddSample(new Sample(bmp, inputs, 10, GetClassNum(file)));
            }
            samples.Shuffle();

            this.Height = 580;
            this.Width = 850;
            this.networksFabric = networksFabric;
            netTypeBox.Items.AddRange(this.networksFabric.Keys.Select(s => (object) s).ToArray());
            netTypeBox.SelectedIndex = 0;
            generator.FigureCount = (int) classCounter.Value;
            button3_Click(this, null);
            pictureBox1.Image = Properties.Resources.Title;
        }

        public void UpdateLearningInfo(double progress, double error, TimeSpan elapsedTime)
        {
            if (progressBar1.InvokeRequired)
            {
                progressBar1.Invoke(new TrainProgressHandler(UpdateLearningInfo), progress, error, elapsedTime);
                return;
            }

            StatusLabel.Text = "Ошибка: " + error;
            int progressPercent = (int) Math.Round(progress * 100);
            progressPercent = Math.Min(100, Math.Max(0, progressPercent));
            elapsedTimeLabel.Text = "Затраченное время : " + elapsedTime.Duration().ToString(@"hh\:mm\:ss\:ff");
            progressBar1.Value = progressPercent;
        }


        private void set_result(Sample figure)
        {
            label1.ForeColor = figure.Correct() ? Color.Green : Color.Red;

            label1.Text = "Распознано : " + figure.recognizedClass;

            label8.Text = string.Join("\n", figure.Output.Select(d => d.ToString(CultureInfo.InvariantCulture)));
            pictureBox1.Image = figure.bitmap;
            pictureBox1.Invalidate();
        }

        private void pictureBox1_MouseClick(object sender, MouseEventArgs e)
        {
            string[] imageFiles = Directory.GetFiles("imgs", "*.*", SearchOption.TopDirectoryOnly).ToArray();
            Random r = new Random();
            Sample fig = samples[r.Next(samples.Count)];

            Net.Predict(fig);
            set_result(fig);
        }

        private async Task<double> train_networkAsync(int training_size, int epoches, double acceptable_error,
            bool parallel = true)
        {
            //  Выключаем всё ненужное
            label1.Text = "Выполняется обучение...";
            label1.ForeColor = Color.Red;
            groupBox1.Enabled = false;
            pictureBox1.Enabled = false;
            trainOneButton.Enabled = false;

            try
            {
                //  Обучение запускаем асинхронно, чтобы не блокировать форму
                var curNet = Net;
                double f = await Task.Run(() => curNet.TrainOnDataSet(samples, epoches, acceptable_error, parallel));

                label1.Text = "Щелкните на картинку для теста нового образа";
                label1.ForeColor = Color.Green;
                groupBox1.Enabled = true;
                pictureBox1.Enabled = true;
                trainOneButton.Enabled = true;
                StatusLabel.Text = "Ошибка: " + f;
                StatusLabel.ForeColor = Color.Green;
                return f;
            }
            catch (Exception e)
            {
                label1.Text = $"Исключение: {e.Message}";
            }

            return 0;
        }

        private void button1_Click(object sender, EventArgs e)
        {
#pragma warning disable CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed
            train_networkAsync((int) TrainingSizeCounter.Value, (int) EpochesCounter.Value,
                (100 - AccuracyCounter.Value) / 100.0, parallelCheckBox.Checked);
#pragma warning restore CS4014 // Because this call is not awaited, execution of the current method continues before the call is completed

            ExportNetwork(netTypeBox.SelectedIndex == 1);
        }

        public void ExportNetwork(bool student)
        {
            if (student)
            {
                (Net as StudentNetwork).ExportNetwork();
            }
            else
            {
                using (FileStream fs = new FileStream("networks\\libNet.bin", FileMode.Create))
                {
                    var formatter = new System.Runtime.Serialization.Formatters.Binary.BinaryFormatter();
                    formatter.Serialize(fs, (Net as AccordNet).network);
                }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            Enabled = false;
            //  Тут просто тестирование новой выборки
            //  Создаём новую обучающую выборку
            SamplesSet samples = new SamplesSet();

            //for (int i = 0; i < (int) TrainingSizeCounter.Value; i++)
                //samples.AddSample(generator.GenerateFigure());

            double accuracy = samples.TestNeuralNetwork(Net);

            StatusLabel.Text = $"Точность на тестовой выборке : {accuracy * 100,5:F2}%";
            StatusLabel.ForeColor = accuracy * 100 >= AccuracyCounter.Value ? Color.Green : Color.Red;

            Enabled = true;
        }

        private void button3_Click(object sender, EventArgs e)
        {
            //  Проверяем корректность задания структуры сети
            int[] structure = CurrentNetworkStructure();
            if (structure.Length < 2 || structure[0] != 400 ||
                structure[structure.Length - 1] != generator.FigureCount)
            {
                MessageBox.Show(
                    $"В сети должно быть более двух слоёв, первый слой должен содержать 400 нейронов, последний - ${generator.FigureCount}",
                    "Ошибка", MessageBoxButtons.OK);
                return;
            }

            // Чистим старые подписки сетей
            foreach (var network in networksCache.Values)
                network.TrainProgress -= UpdateLearningInfo;
            // Пересоздаём все сети с новой структурой
            networksCache = networksCache.ToDictionary(oldNet => oldNet.Key, oldNet => CreateNetwork(oldNet.Key));
        }

        private int[] CurrentNetworkStructure()
        {
            return netStructureBox.Text.Split(';').Select(int.Parse).ToArray();
        }

        private void classCounter_ValueChanged(object sender, EventArgs e)
        {
            generator.FigureCount = (int) classCounter.Value;
            var vals = netStructureBox.Text.Split(';');
            if (!int.TryParse(vals.Last(), out _)) return;
            vals[vals.Length - 1] = classCounter.Value.ToString();
            netStructureBox.Text = vals.Aggregate((partialPhrase, word) => $"{partialPhrase};{word}");
        }

        private void btnTrainOne_Click(object sender, EventArgs e)
        {
            if (Net == null) return;
            Random r = new Random();
            Sample fig = samples[r.Next(samples.Count)];
            pictureBox1.Invalidate();
            Net.Train(fig, 0.00005, parallelCheckBox.Checked);
            set_result(fig);
        }

        private BaseNetwork CreateNetwork(string networkName)
        {
            var network = networksFabric[networkName](CurrentNetworkStructure());
            network.TrainProgress += UpdateLearningInfo;
            return network;
        }

        private void recreateNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Заново пересоздаёт сеть с указанными параметрами";
        }

        private void netTrainButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Обучить нейросеть с указанными параметрами";
        }

        private void testNetButton_MouseEnter(object sender, EventArgs e)
        {
            infoStatusLabel.Text = "Тестировать нейросеть на тестовой выборке такого же размера";
        }
    }
}