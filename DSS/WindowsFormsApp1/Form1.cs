using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using System.IO;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using weka.core;

namespace WindowsFormsApp1
{
    public partial class Form1 : Form

    {
        List<object> list;
        string file;
        const int percentSplit = 66;

        static weka.classifiers.Classifier J48cl = null;
        static weka.classifiers.Classifier NaiveBayescl = null;
        static weka.classifiers.Classifier RandomForestcl = null;
        static weka.classifiers.Classifier RandomTreecl = null;
        static weka.classifiers.Classifier _1IBKcl = null;
        static weka.classifiers.Classifier _3IBKcl = null;
        static weka.classifiers.Classifier _5IBKcl = null;
        static weka.classifiers.Classifier _7IBKcl = null;
        static weka.classifiers.Classifier _9IBKcl = null;
        static weka.classifiers.Classifier LogRegressioncl = null;
        static weka.classifiers.Classifier SupportVectorMachine = null;
        static weka.classifiers.Classifier ArtNeuralNetwork = null;


        static weka.classifiers.Classifier model = null;

        public static double J48classifyTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                J48cl = new weka.classifiers.trees.J48();

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                J48cl.buildClassifier(train);
                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                     double predictedClass = J48cl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double NaiveBayesTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);


                NaiveBayescl = new weka.classifiers.bayes.NaiveBayes();
                

                //discretize
                weka.filters.Filter myDiscretize = new weka.filters.unsupervised.attribute.Discretize();
                myDiscretize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDiscretize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                NaiveBayescl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = NaiveBayescl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double RandomForestTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                RandomForestcl = new weka.classifiers.trees.RandomForest();

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                RandomForestcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                   double predictedClass = RandomForestcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double RandomTreeTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                RandomTreecl = new weka.classifiers.trees.RandomTree();

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                RandomTreecl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                   double predictedClass = RandomTreecl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double _1IBkTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                _1IBKcl = new weka.classifiers.lazy.IBk(1);

                weka.filters.Filter myDummy = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummy.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummy);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.instance.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                _1IBKcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = _1IBKcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double _3IBkTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                _3IBKcl = new weka.classifiers.lazy.IBk(3);

                weka.filters.Filter myDummy = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummy.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummy);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.instance.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                _3IBKcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = _3IBKcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double _5IBkTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                _5IBKcl = new weka.classifiers.lazy.IBk(5);

                weka.filters.Filter myDummy = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummy.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummy);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.instance.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                _5IBKcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = _5IBKcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double _7IBkTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                _7IBKcl = new weka.classifiers.lazy.IBk(7);

                weka.filters.Filter myDummy = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummy.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummy);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.instance.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                _7IBKcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = _7IBKcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double _9IBkTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                _9IBKcl = new weka.classifiers.lazy.IBk(9);

                weka.filters.Filter myDummy = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummy.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummy);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.instance.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                _9IBKcl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = _9IBKcl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double LogRegressionTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                LogRegressioncl = new weka.classifiers.functions.Logistic();

                weka.filters.Filter myDummy = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummy.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummy);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.instance.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                LogRegressioncl.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = LogRegressioncl.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double SupportVectorMachineTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);

                
                SupportVectorMachine= new weka.classifiers.functions.SMO();

               weka.filters.Filter myDummy = new weka.filters.unsupervised.attribute.NominalToBinary();
               
                myDummy.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummy);
 

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.instance.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                
                SupportVectorMachine.buildClassifier(train);


                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = SupportVectorMachine.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }

        public static double ArtNeuralNetworkTest(weka.core.Instances insts)
        {
            try
            {
                //weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader("iris.arff"));

                insts.setClassIndex(insts.numAttributes() - 1);
                ArtNeuralNetwork = new weka.classifiers.functions.MultilayerPerceptron();
                weka.filters.Filter myDummy = new weka.filters.unsupervised.attribute.NominalToBinary();
                
                myDummy.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummy);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.instance.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);

                weka.filters.Filter myRandom = new weka.filters.unsupervised.instance.Randomize();
                myRandom.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myRandom);

                int trainSize = insts.numInstances() * percentSplit / 100;
                int testSize = insts.numInstances() - trainSize;
                weka.core.Instances train = new weka.core.Instances(insts, 0, trainSize);

                ArtNeuralNetwork.buildClassifier(train);
                int numCorrect = 0;
                for (int i = trainSize; i < insts.numInstances(); i++)
                {
                    weka.core.Instance currentInst = insts.instance(i);
                    double predictedClass = ArtNeuralNetwork.classifyInstance(currentInst);
                    if (predictedClass == insts.instance(i).classValue())
                        numCorrect++;
                }
                return (double)numCorrect / (double)testSize * 100.0;
            }
            catch (java.lang.Exception ex)
            {
                ex.printStackTrace();
                return 0;
            }
        }
        public Form1()
        {

            list = new List<object>();
            InitializeComponent();
        }

        private void Browse_Click(object sender, EventArgs e)
        {
            OpenFileDialog ofd = new OpenFileDialog();
            ofd.ShowDialog();
            file = ofd.SafeFileName;
            label2.Text = "Wait process in progress";

            weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader(file));
            double max_value = J48classifyTest(insts);
            model = J48cl;
            string name = "J48cl";


            double NBvalue = NaiveBayesTest(insts);
            if (NBvalue > max_value)
            {
                max_value = NBvalue;
                model = NaiveBayescl;
                name = "NaiveBayes";
            }
            double RFvalue = RandomForestTest(insts);
            if (RFvalue > max_value)
            {
                max_value = RFvalue;
                model = RandomForestcl;
                name = "RandomForest";
            }
            double RTvalue = RandomTreeTest(insts);
            if (RTvalue > max_value)
            {
                max_value = RTvalue;
                model = RandomTreecl;
                name = "RandomTree";
            }
            double _5IBKvalue = _5IBkTest(insts);
            if (_5IBKvalue > max_value)
            {
                max_value = _5IBKvalue;
                model = _5IBKcl;
                name = " _5IBK";
            }
            double _7IBKvalue = _7IBkTest(insts);
            if (_7IBKvalue > max_value)
            {
                max_value = _7IBKvalue;
                model = _7IBKcl;
                name = " _7IBk";
            }
            double _9IBKvalue = _9IBkTest(insts);
            if (_9IBKvalue > max_value)
            {
                max_value = _9IBKvalue;
                model = _9IBKcl;
                name = " _9IBk";
            }
            double LogRegressionvalue = LogRegressionTest(insts);
            if (LogRegressionvalue > max_value)
            {
                max_value = LogRegressionvalue;
                model = LogRegressioncl;
                name = "LogRegression";
            }
            double SVM = SupportVectorMachineTest(insts);
            if (SVM > max_value)
            {
                max_value = SVM;
                model = SupportVectorMachine;
                name = "SupportVectorMachine";
            }
            double ArtNN = ArtNeuralNetworkTest(insts);
            if (ArtNN > max_value)
            {
                max_value = ArtNN;
                model = ArtNeuralNetwork;
                name = "ArtNeuralNetwork";
            }

            label2.Text= name + " is the most successful algorithm for this data set " + "(%" + Math.Round(max_value, 2) + ")";
    
            for(int i=0;i<insts.numAttributes()-1;i++)
            {
                if (insts.attribute(i).isNominal())
                {
                   
                    Label l = new Label();
                    flowLayoutPanel2.Controls.Add(l);

                    l.Top = i * 30 + 175;
                    l.Left = 100;
                    l.Text = insts.attribute(i).name().ToString()+": ";
                 

                    ComboBox mybox = new ComboBox();
                    for(int j=0;j<insts.attribute(i).numValues();j++)
                    {
                        mybox.Items.Add(insts.attribute(i).value(j));
                    }
                    // Creating and setting the properties of comboBox 

                    mybox.DropDownStyle = ComboBoxStyle.DropDownList;
                    mybox.Size = new Size(100,30);
                    mybox.Top = i * 30 + 175;
                    l.Left = 200;
                    mybox.Tag = i;
                    flowLayoutPanel2.Controls.Add(mybox);
                    list.Add(mybox);
                  

                }
                else
                {
                    Label l = new Label();
                    flowLayoutPanel2.Controls.Add(l);

                    l.Text = insts.attribute(i).name().ToString() + ": ";
                    TextBox txt = new TextBox();
                    txt.Tag = i;                  
                    list.Add(txt);
                    flowLayoutPanel2.Controls.Add(txt);
                }
            }

            Button button = new Button();
            button.Name = "Discover";
            button.Text = "Find";
            button.Location = new Point(468, 72);
            button.Size = new Size(60, 30);
            button.BackColor = Color.Red;
            button.Font = new Font(button.Font.Name, button.Font.Size, FontStyle.Bold);
            button.Click += new EventHandler(button1_Click);

            Controls.Add(button);


        }
        private void button1_Click(object sender, EventArgs e)
        {
            weka.core.Instances insts = new weka.core.Instances(new java.io.FileReader(file));
            double[] Data = new double[insts.numAttributes()];
            for (int i = 0; i < list.Count; i++)
            {
                if (list[i].GetType() == typeof(TextBox))
                {
                    TextBox txt = (TextBox)list[i];
                    string value = txt.Text.Replace('.', ',');
                    Data[i] = Convert.ToDouble(value);
                }
                else
                {
                    ComboBox combobox = (ComboBox)list[i];
                    Data[i] = Convert.ToDouble(combobox.SelectedIndex);
                }
            }
            // Data[(insts.numAttributes() - 1] = 0;
            insts.setClassIndex(insts.numAttributes() - 1);
            Instance newInsts = new Instance(1.0, Data);
            insts.add(newInsts);
            string type = model.GetType().ToString();

           if (type == "weka.classifiers.bayes.NaiveBayes")
            {
                weka.filters.Filter myDiscretize = new weka.filters.unsupervised.attribute.Discretize();
                myDiscretize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDiscretize);
            }
            else if (type == "weka.classifiers.lazy.IBk")
            {
                weka.filters.Filter myDummy = new weka.filters.unsupervised.attribute.NominalToBinary();
                myDummy.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myDummy);

                weka.filters.Filter myNormalize = new weka.filters.unsupervised.instance.Normalize();
                myNormalize.setInputFormat(insts);
                insts = weka.filters.Filter.useFilter(insts, myNormalize);
            }
            double index = model.classifyInstance(insts.lastInstance());

            string result = insts.attribute(insts.numAttributes() - 1).value(Convert.ToInt16(index));

            MessageBox.Show(result);
        }

        private void flowLayoutPanel2_Paint(object sender, PaintEventArgs e)
        {

        }
    }
}
