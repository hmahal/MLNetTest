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
        const string _dataPath = @".\Data\taxi-fare-train.csv";
        const string _testDataPath = @".\Data\taxi-fare-test.csv";
        const string _modelPath = @".\Data\Model.zip";

        static async Task MainAsync(string[] args)
        {
            PredictionModel<TaxiTrip, TaxiTripFarePrediction> model = await Train();
        }

        public static async Task<PredictionModel<TaxiTrip, TaxiTripFarePrediction>> Train()
        {
            var pipeline = new LearningPipeline()
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
    }
}
