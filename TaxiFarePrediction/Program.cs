using System;
using Microsoft.ML.Models;
using Microsoft.ML.Runtime;
using Microsoft.ML.Runtime.Api;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML;
using System.Threading.Tasks;

namespace TaxiFarePrediction
{
    class Program
    {
        const string _dataPath = @"..\..\Data\taxi-fare-train.csv";
        const string _testDataPath = @"..\..\Data\taxi-fare-test.csv";
        const string _modelPath = @"..\..\Data\Model.zip";

        public static async Task Main(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
            Evaluate(model);
            var prediction = model.Predict(TestTrips.Trip1);
            Console.WriteLine("Predicted fare: {0}, actual fare: 29.5", prediction.fare_amount);
            Console.ReadLine();
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline
            {
                new TextLoader<TaxiTrip>(_dataPath, useHeader: true, separator: ","),
                new ColumnCopier(("fare_amount", "Label")),
                new CategoricalOneHotVectorizer("vendor_id", "rate_code", "payment_type"),
                new ColumnConcatenator("Features",
                                        "vendor_id",
                                        "rate_code",
                                        "passenger_count",
                                        "trip_distance",
                                        "payment_type"),
                new FastTreeRegressor()
            };

            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = pipeline.Train<TaxiTrip, TaxiTripFarePrediction>();

            await model.WriteAsync(_modelPath);

            return model;
        }

        public static void Evaluate(PredictionModel<TaxiTrip, TaxiTripFarePrediction> model)
        {
            var testData = new TextLoader<TaxiTrip>(_testDataPath, useHeader: true, separator: ",");
            var evaluator = new RegressionEvaluator();
            RegressionMetrics metrics = evaluator.Evaluate(model, testData);
            Console.WriteLine("Rms = " + metrics.Rms);
            Console.WriteLine("RSquared = " + metrics.RSquared);
        }
    }

    static class TestTrips
    {
        internal static readonly TaxiTrip Trip1 = new TaxiTrip
        {
            vendor_id = "VTS",
            rate_code = "1",
            passenger_count = 1,
            trip_distance = 10.33f,
            payment_type = "CSH",
            fare_amount = 0 //Should be 29.5
        };
    }
}
