using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;

namespace SentimentAnalysis
{
    class Program
    {
        //Global variables set to the path of the data
        const string _datapath = @"..\..\..\data\imdb_labelled.txt";
        const string _testDataPath = @"..\..\..\data\yelp_labelled.txt";

        static void Main(string[] args)
        {
            var model = TrainAndPredict();
        }

        public static PredictionModel<SentimentData, SentimentPrediction> TrainAndPredict()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader<SentimentData>(_datapath, useHeader: false, separator: "tab"),
                new TextFeaturizer("Features", "SentimentText"),
                new FastTreeBinaryClassifier() { NumLeaves = 5, NumTrees = 5, MinDocumentsInLeafs = 2 }
            };

            PredictionModel<SentimentData, SentimentPrediction> model = pipeline.Train<SentimentData, SentimentPrediction>();

            IEnumerable<SentimentData> sentiments = new[]
            {
                new SentimentData
                {
                    SentimentText = "Contoso's 11 is wonderful experience",
                    Sentiment = 0
                },
                new SentimentData
                {
                    SentimentText = "Really bad",
                    Sentiment = 0
                },
                new SentimentData
                {
                    SentimentText = "Joe versus the Volcano Coffee Company is a great film.",
                    Sentiment = 0
                }
            };

            IEnumerable<SentimentPrediction> predictions = model.Predict(sentiments);

            Console.WriteLine();
            Console.WriteLine("Sentiment Predictions");
            Console.WriteLine("---------------------");
        }
    }
}
